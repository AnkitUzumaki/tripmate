"""The orchestration loop.

One loop, provider-agnostic: the model chooses tools, the registry executes them,
results go back as tool messages, and the model synthesises a grounded answer.
"""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol

from tripmate.config import Settings, get_settings
from tripmate.core.cache import SemanticCache
from tripmate.core.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from tripmate.core.trace import EventType, Tracer
from tripmate.db import SessionStore
from tripmate.models import (
    AgentResponse, Citation, LLMResponse, ToolCall, ToolResult,
)
from tripmate.models import CITATION_RE
from tripmate.tools.registry import ToolRegistry

FORCED_SYNTHESIS_NOTE = (
    "You have gathered enough tool output. Answer the user's question now, "
    "using only what the tools returned. Do not request any more tools."
)

# Ruling F2: a forced-synthesis turn can still come back with empty content
# (e.g. the model just repeats a tool call attempt). Never surface "" as an answer.
EMPTY_ANSWER_FALLBACK = (
    "I wasn't able to put together an answer from the available information. "
    "Could you rephrase the question?"
)


class InvalidQuery(Exception):
    """Raised when input fails validation at the boundary."""


class SupportsComplete(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse: ...


def validate_query(query: str, max_chars: int) -> str:
    cleaned = (query or "").strip()
    if not cleaned:
        raise InvalidQuery("Your message looks empty. Ask me about Tokyo, Reykjavik, "
                           "Bangkok or Barcelona.")
    if len(cleaned) > max_chars:
        raise InvalidQuery(f"That message is too long ({len(cleaned)} characters). "
                           f"Please keep it under {max_chars}.")
    return cleaned


def extract_citations(answer: str) -> list[Citation]:
    return [
        Citation(city=match.group(1).strip().lower(),
                 section=match.group(2).strip().upper())
        for match in CITATION_RE.finditer(answer or "")
    ]


def validate_citations(
    answer: str, citations: list[Citation], allowed_refs: set[str]
) -> tuple[str, list[Citation], list[str]]:
    """Strip any citation that does not correspond to a chunk actually retrieved."""
    kept: list[Citation] = []
    rejected: list[str] = []
    cleaned = answer

    for citation in citations:
        if citation.ref in allowed_refs:
            if citation.ref not in {c.ref for c in kept}:
                kept.append(citation)
            continue
        rejected.append(citation.ref)
        cleaned = cleaned.replace(f"[{citation.ref}]", "")
        cleaned = cleaned.replace(f"[{citation.city}/{citation.section.title()}]", "")

    return cleaned.replace("  ", " ").strip(), kept, rejected


def _refs_from(result: ToolResult) -> set[str]:
    if result.status != "ok" or not result.data:
        return set()
    return {chunk["ref"] for chunk in result.data.get("chunks", []) if "ref" in chunk}


class Agent:
    def __init__(
        self,
        llm: SupportsComplete,
        registry: ToolRegistry,
        tracer_factory: Callable[[str], Tracer] | None = None,
        store: SessionStore | None = None,
        cache: SemanticCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._settings = settings or get_settings()
        self._store = store
        self._cache = cache
        self._tracer_factory = tracer_factory or (lambda sid: Tracer(session_id=sid))

    def chat(self, query: str, session_id: str | None = None) -> AgentResponse:
        session_id = session_id or uuid.uuid4().hex[:12]
        tracer = self._tracer_factory(session_id)
        started = time.perf_counter()

        try:
            cleaned = validate_query(query, self._settings.max_query_chars)
        except InvalidQuery as exc:
            tracer.record(EventType.QUERY_RECEIVED, valid=False, reason=str(exc))
            return self._finish(str(exc), [], tracer, session_id, started)

        tracer.record(EventType.QUERY_RECEIVED, query=cleaned,
                      prompt_version=PROMPT_VERSION)

        cached = self._cache.lookup(cleaned) if self._cache else None
        if cached is not None:
            tracer.record(EventType.CACHE_HIT, similarity=round(cached.similarity, 4))
            return self._finish(cached.answer, cached.citations, tracer, session_id,
                                started, was_cached=True)

        answer, citations = self._run_loop(cleaned, session_id, tracer)

        if self._cache:
            self._cache.store(cleaned, answer, citations)

        return self._finish(answer, citations, tracer, session_id, started)

    def _run_loop(
        self, query: str, session_id: str, tracer: Tracer
    ) -> tuple[str, list[Citation]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self._store:
            messages.extend(self._store.history(session_id))
        messages.append({"role": "user", "content": query})

        allowed_refs: set[str] = set()
        schemas = self._registry.schemas()

        for iteration in range(self._settings.max_tool_iterations):
            is_last = iteration == self._settings.max_tool_iterations - 1
            if is_last:
                messages.append({"role": "system", "content": FORCED_SYNTHESIS_NOTE})

            response = self._call_llm(messages, None if is_last else schemas, tracer)

            if not response.tool_calls:
                return self._finalise(response.content or "", allowed_refs, tracer)

            messages.append(self._assistant_message(response))
            results = self._dispatch_all(response.tool_calls, tracer)

            for call, result in zip(response.tool_calls, results):
                allowed_refs |= _refs_from(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(result.model_dump(), default=str),
                })

        final = self._call_llm(messages, None, tracer)
        return self._finalise(final.content or "", allowed_refs, tracer)

    def _call_llm(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
        tracer: Tracer,
    ) -> LLMResponse:
        started = time.perf_counter()
        response = self._llm.complete(messages, tools=tools)
        tracer.record(
            EventType.LLM_CALL,
            duration_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=response.cost_usd,
            requested_tools=[c.name for c in response.tool_calls],
        )
        return response

    @staticmethod
    def _assistant_message(response: LLMResponse) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name,
                                 "arguments": json.dumps(call.arguments)},
                }
                for call in response.tool_calls
            ],
        }

    def _dispatch_all(
        self, calls: list[ToolCall], tracer: Tracer
    ) -> list[ToolResult]:
        """Run independent tool calls concurrently; a tool never raises into the loop."""
        for call in calls:
            tracer.record(EventType.TOOL_CALL, tool=call.name, arguments=call.arguments)

        def run(call: ToolCall) -> tuple[ToolResult, float]:
            started = time.perf_counter()
            result = self._registry.dispatch(call.name, call.arguments)
            return result, (time.perf_counter() - started) * 1000

        if len(calls) == 1:
            outcomes = [run(calls[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                outcomes = list(pool.map(run, calls))

        results: list[ToolResult] = []
        for call, (result, duration) in zip(calls, outcomes):
            event_type = (
                EventType.TOOL_ERROR if result.status == "error" else EventType.TOOL_RESULT
            )
            # RAGAS hook: carry retrieved chunk text on the trace so faithfulness /
            # context-precision scoring never needs to re-run retrieval later.
            chunks = (result.data or {}).get("chunks", []) if result.status == "ok" else []
            texts_kwarg = {"texts": [chunk["text"] for chunk in chunks]} if chunks else {}
            tracer.record(event_type, duration_ms=duration, tool=call.name,
                          status=result.status, reason=result.reason, **texts_kwarg)
            if (result.data or {}).get("source") == "mock_fallback":
                tracer.record(EventType.FALLBACK_USED, tool=call.name)
            results.append(result)
        return results

    @staticmethod
    def _finalise(
        raw_answer: str, allowed_refs: set[str], tracer: Tracer
    ) -> tuple[str, list[Citation]]:
        answer, kept, rejected = validate_citations(
            raw_answer, extract_citations(raw_answer), allowed_refs
        )
        if rejected:
            tracer.record(EventType.CITATION_REJECTED, refs=rejected)

        is_empty = not answer.strip()
        final_answer = EMPTY_ANSWER_FALLBACK if is_empty else answer
        tracer.record(EventType.ANSWER_SYNTHESIZED,
                      citations=[c.ref for c in kept], length=len(final_answer),
                      used_fallback=is_empty)
        return final_answer, kept

    def _finish(
        self, answer: str, citations: list[Citation], tracer: Tracer,
        session_id: str, started: float, was_cached: bool = False,
    ) -> AgentResponse:
        totals = tracer.totals()
        latency_ms = (time.perf_counter() - started) * 1000
        tracer.flush()
        return AgentResponse(
            answer=answer, citations=citations, trace=tracer.events,
            prompt_tokens=totals.prompt_tokens,
            completion_tokens=totals.completion_tokens,
            cost_usd=totals.cost_usd, latency_ms=latency_ms,
            session_id=session_id, was_cached=was_cached,
        )
