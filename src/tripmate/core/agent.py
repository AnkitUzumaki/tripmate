"""Agent facade over the compiled LangGraph workflow.

Orchestration lives in `tripmate.graph` as a StateGraph. This module keeps the public
surface (`Agent.chat`) and the pure helpers that operate on answers — query validation
and citation checking — which are called from graph nodes and tested directly.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Callable

from tripmate.config import Settings, get_settings
from tripmate.core.cache import SemanticCache
from tripmate.core.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from tripmate.core.trace import EventType, Tracer
from tripmate.db import SessionStore
from tripmate.models import AgentResponse, Citation
from tripmate.models import CITATION_RE
from tripmate.tools.registry import ToolRegistry

# Ruling F2: a forced-synthesis turn can still come back with empty content
# (e.g. the model just repeats a tool call attempt). Never surface "" as an answer.
EMPTY_ANSWER_FALLBACK = (
    "I wasn't able to put together an answer from the available information. "
    "Could you rephrase the question?"
)


class InvalidQuery(Exception):
    """Raised when input fails validation at the boundary."""




def validate_query(query: str, max_chars: int) -> str:
    cleaned = (query or "").strip()
    if not cleaned:
        raise InvalidQuery("Your message looks empty. Ask me about Tokyo, Reykjavik, "
                           "Bangkok or Barcelona.")
    if len(cleaned) > max_chars:
        raise InvalidQuery(f"That message is too long ({len(cleaned)} characters). "
                           f"Please keep it under {max_chars}.")
    return cleaned


DANGLING_LEAD_IN_RE = re.compile(r"\s*(?:Sources?|Refs?|References?)\s*:\s*(?=$|\n)", re.I)


def extract_citations(answer: str) -> list[Citation]:
    return [
        Citation(city=match.group(1).strip().lower(),
                 section=match.group(2).strip().upper())
        for match in CITATION_RE.finditer(answer or "")
    ]


def validate_citations(
    answer: str, citations: list[Citation], allowed_refs: set[str]
) -> tuple[str, list[Citation], list[str]]:
    """Strip any citation that does not correspond to a chunk actually retrieved.

    Matching is done on parsed refs via CITATION_RE, not literal text, so any casing
    the model emits is handled identically. `citations` stays in the signature for
    call-site compatibility, but refs are re-derived from `answer` during the
    substitution pass so a mismatched casing can never slip through.
    """
    kept: list[Citation] = []
    rejected: list[str] = []
    kept_refs: set[str] = set()
    rejected_refs: set[str] = set()

    def _replace(match: re.Match[str]) -> str:
        citation = Citation(
            city=match.group(1).strip().lower(),
            section=match.group(2).strip().upper(),
        )
        if citation.ref in allowed_refs:
            if citation.ref not in kept_refs:
                kept_refs.add(citation.ref)
                kept.append(citation)
            return match.group(0)
        if citation.ref not in rejected_refs:
            rejected_refs.add(citation.ref)
            rejected.append(citation.ref)
        return ""

    cleaned = CITATION_RE.sub(_replace, answer or "")
    # A stripped citation can orphan the phrase that introduced it ("... Source:").
    cleaned = DANGLING_LEAD_IN_RE.sub("", cleaned)
    return " ".join(cleaned.split()), kept, rejected


def _refs_from(result: ToolResult) -> set[str]:
    if result.status != "ok" or not result.data:
        return set()
    return {chunk["ref"] for chunk in result.data.get("chunks", []) if "ref" in chunk}


class Agent:
    """Public surface over the compiled graph.

    Deliberately thin: input validation and response mapping live here, everything
    between them is the graph's business. Keeping `chat()`'s signature is what lets the
    CLI, the API and the whole eval harness work against the graph unchanged.
    """

    def __init__(
        self,
        graph,
        registry: ToolRegistry,
        tracer_factory: Callable[[str], Tracer] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._graph = graph
        self._registry = registry
        self._settings = settings or get_settings()
        self._tracer_factory = tracer_factory or (lambda sid: Tracer(session_id=sid))
        self._tracer: Tracer | None = None

    @property
    def graph(self):
        """The compiled StateGraph, exposed for inspection and visualisation."""
        return self._graph

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def settings(self) -> Settings:
        return self._settings

    def current_tracer(self) -> Tracer:
        """The tracer for the turn in flight.

        Nodes receive this as a zero-argument callable so they need no knowledge of
        session handling.
        """
        if self._tracer is None:
            self._tracer = self._tracer_factory("detached")
        return self._tracer

    def chat(self, query: str, session_id: str | None = None) -> AgentResponse:
        session_id = session_id or uuid.uuid4().hex[:12]
        self._tracer = self._tracer_factory(session_id)
        tracer = self._tracer
        started = time.perf_counter()

        # Validation stays outside the graph so malformed input costs zero LLM calls.
        try:
            cleaned = validate_query(query, self._settings.max_query_chars)
        except InvalidQuery as exc:
            tracer.record(EventType.QUERY_RECEIVED, valid=False, reason=str(exc))
            return self._finish(str(exc), [], tracer, session_id, started)

        tracer.record(EventType.QUERY_RECEIVED, query=cleaned,
                      prompt_version=PROMPT_VERSION)

        from langchain_core.messages import HumanMessage
        from langgraph.errors import GraphRecursionError

        config = {
            "configurable": {"thread_id": session_id},
            # Each tool round costs several node visits; the ceiling is expressed in
            # tool iterations, so convert rather than leaking a graph detail into config.
            "recursion_limit": self._settings.max_tool_iterations * 4 + 4,
        }

        try:
            final = self._graph.invoke(
                {"messages": [HumanMessage(content=cleaned)], "query": cleaned},
                config=config,
            )
        except GraphRecursionError:
            tracer.record(EventType.ANSWER_SYNTHESIZED, used_fallback=True,
                          reason="recursion limit reached")
            return self._finish(EMPTY_ANSWER_FALLBACK, [], tracer, session_id, started)

        # State holds plain refs; AgentResponse's contract is Citation objects.
        citations = [Citation.parse(f"[{ref}]") for ref in final.get("citations", [])]
        return self._finish(
            final.get("answer", ""),
            [c for c in citations if c is not None],
            tracer, session_id, started,
            was_cached=bool(final.get("was_cached")),
        )

    def _finish(
        self, answer: str, citations: list[Citation], tracer: Tracer,
        session_id: str, started: float, was_cached: bool = False,
    ) -> AgentResponse:
        totals = tracer.totals()
        latency_ms = (time.perf_counter() - started) * 1000
        tracer.flush()
        return AgentResponse(
            answer=answer or EMPTY_ANSWER_FALLBACK,
            citations=citations,
            trace=tracer.events,
            prompt_tokens=totals.prompt_tokens,
            completion_tokens=totals.completion_tokens,
            cost_usd=totals.cost_usd,
            latency_ms=latency_ms,
            session_id=session_id,
            was_cached=was_cached,
        )
