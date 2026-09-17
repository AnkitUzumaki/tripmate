"""Graph nodes.

Each is a pure function of AgentState returning a partial update, which is what makes
them independently unit-testable — an improvement over the previous private methods,
which could only be exercised through `chat()`.

Nodes are built by factory functions so their dependencies (LLM, cache, tracer) are
injected rather than imported, keeping the same testability the previous design had.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from tripmate.core.cache import SemanticCache
from tripmate.core.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from tripmate.core.trace import EventType, Tracer
from tripmate.graph.state import AgentState
from tripmate.models import Citation

CACHE_HIT = "cache_hit"
PROCEED = "proceed"


def _estimate_cost(model_name: str, usage: dict) -> float:
    """Price a call from its token counts.

    LangChain does not report cost, so look it up from LiteLLM's price map using the
    counts the provider returned. No network call: this is table lookup only. Returns
    0.0 for a model the map does not price, which is normal for local and open-weight
    models rather than an error.
    """
    if not model_name or not usage:
        return 0.0
    try:
        from litellm import cost_per_token

        prompt_cost, completion_cost = cost_per_token(
            model=model_name,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
        )
        return float(prompt_cost + completion_cost)
    except Exception:
        return 0.0


def make_check_cache(cache: SemanticCache | None, tracer_of: Callable[[], Tracer]):
    """Semantic cache as a routing decision rather than an early return.

    Only consulted when the conversation has no prior turns: a follow-up such as
    "What should I pack?" is not a standalone question, and replaying an answer cached
    for those same words in another session loses the context entirely.
    """

    def check_cache(state: AgentState) -> dict[str, Any]:
        query = state["query"]
        is_first_turn = len(state.get("messages", [])) <= 1
        if cache is None or not is_first_turn:
            return {"was_cached": False, "is_first_turn": is_first_turn}

        hit = cache.lookup(query)
        if hit is None:
            return {"was_cached": False, "is_first_turn": True}

        tracer_of().record(EventType.CACHE_HIT, similarity=round(hit.similarity, 4))
        return {
            "answer": hit.answer,
            "citations": [c.ref for c in hit.citations],
            "was_cached": True,
            "is_first_turn": True,
            "messages": [AIMessage(content=hit.answer)],
        }

    return check_cache


def route_after_cache(state: AgentState) -> str:
    """Conditional edge: a cache hit skips the LLM and the tools entirely."""
    return CACHE_HIT if state.get("was_cached") else PROCEED


def make_agent_node(model, tracer_of: Callable[[], Tracer], model_name: str = ""):
    """Call the LLM with the tool schemas bound.

    Whether it emits tool calls is what drives the next edge, so tool selection is a
    property of the graph rather than a branch inside a method.
    """

    def agent(state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

        started = time.perf_counter()
        response = model.invoke(messages)
        usage = getattr(response, "usage_metadata", None) or {}
        cost = _estimate_cost(model_name, usage)

        tracer_of().record(
            EventType.LLM_CALL,
            duration_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            cost_usd=cost,
            prompt_version=PROMPT_VERSION,
            requested_tools=[c["name"] for c in (response.tool_calls or [])],
        )

        # The product requirement is a trace showing which tools were called and with
        # what arguments. ToolNode does not emit that, and the arguments only exist on
        # this message, so record it here rather than losing it.
        for call in response.tool_calls or []:
            tracer_of().record(EventType.TOOL_CALL,
                               tool=call["name"], arguments=call.get("args", {}))

        return {"messages": [response]}

    return agent


def make_collect_refs(tracer_of: Callable[[], Tracer]):
    """Record tool outcomes and accumulate the citation whitelist.

    Runs straight after ToolNode. ToolNode itself only executes tools and appends
    ToolMessages; the trace events and the `allowed_refs` accumulation that the
    anti-fabrication check depends on happen here.
    """

    def collect_refs(state: AgentState) -> dict[str, Any]:
        tracer = tracer_of()
        refs = list(state.get("allowed_refs", []))

        for message in reversed(state["messages"]):
            if not isinstance(message, ToolMessage):
                break
            try:
                payload = json.loads(message.content)
            except (json.JSONDecodeError, TypeError):
                # ToolNode emits a plain-text ToolMessage when the model names a tool
                # that does not exist. Not our ToolResult JSON, but still a tool
                # failure the trace must show rather than silently drop.
                tracer.record(EventType.TOOL_ERROR, tool=message.name,
                              status="error", reason=str(message.content)[:200])
                continue

            status = payload.get("status")
            data = payload.get("data") or {}
            chunks = data.get("chunks", []) if status == "ok" else []

            event = EventType.TOOL_ERROR if status == "error" else EventType.TOOL_RESULT
            extra = {"texts": [c["text"] for c in chunks]} if chunks else {}
            tracer.record(event, tool=message.name, status=status,
                          reason=payload.get("reason"), **extra)

            if data.get("source") == "mock_fallback":
                tracer.record(EventType.FALLBACK_USED, tool=message.name)

            refs.extend(c["ref"] for c in chunks if "ref" in c)

        return {"allowed_refs": sorted(set(refs))}

    return collect_refs


def make_validate_citations(tracer_of: Callable[[], Tracer]):
    """Strip any citation not backed by a chunk retrieved this turn."""

    def validate(state: AgentState) -> dict[str, Any]:
        from tripmate.core.agent import (
            EMPTY_ANSWER_FALLBACK,
            extract_citations,
            validate_citations,
        )

        last = state["messages"][-1]
        raw = last.content if isinstance(last, AIMessage) else ""
        allowed = set(state.get("allowed_refs", []))

        answer, kept, rejected = validate_citations(raw, extract_citations(raw), allowed)
        tracer = tracer_of()
        if rejected:
            tracer.record(EventType.CITATION_REJECTED, refs=rejected)

        is_empty = not answer.strip()
        final = EMPTY_ANSWER_FALLBACK if is_empty else answer
        tracer.record(EventType.ANSWER_SYNTHESIZED,
                      citations=[c.ref for c in kept], length=len(final),
                      used_fallback=is_empty)
        return {"answer": final, "citations": [c.ref for c in kept]}

    return validate


def make_store_cache(cache: SemanticCache | None):
    """Populate the cache, first turn only — mirroring the lookup rule."""

    def store(state: AgentState) -> dict[str, Any]:
        if cache is None or state.get("was_cached"):
            return {}
        if state.get("is_first_turn"):
            citations = [Citation.parse(f"[{ref}]") for ref in state.get("citations", [])]
            cache.store(state["query"], state["answer"],
                        [c for c in citations if c is not None])
        return {}

    return store
