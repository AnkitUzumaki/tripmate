"""Unit tests for graph nodes.

Each node is a pure function of state, so these exercise them directly rather than
through `chat()` — which is the main testability gain from the port.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tripmate.core.trace import EventType, Tracer
from tripmate.graph.nodes import (
    CACHE_HIT,
    PROCEED,
    _estimate_cost,
    make_collect_refs,
    make_validate_citations,
    route_after_cache,
)


def _tracer() -> Tracer:
    return Tracer(session_id="test")


# --- routing ---

def test_route_sends_a_cache_hit_straight_to_the_end():
    assert route_after_cache({"was_cached": True}) == CACHE_HIT


def test_route_proceeds_to_the_agent_on_a_miss():
    assert route_after_cache({"was_cached": False}) == PROCEED
    assert route_after_cache({}) == PROCEED


# --- collect_refs ---

def test_collect_refs_gathers_chunk_refs_from_tool_messages():
    tracer = _tracer()
    state = {
        "messages": [
            AIMessage(content=""),
            ToolMessage(
                name="search_destination_guide", tool_call_id="1",
                content='{"status": "ok", "data": {"chunks": ['
                        '{"ref": "tokyo/PACKING TIPS", "text": "Layers."}]}}',
            ),
        ],
        "allowed_refs": [],
    }
    out = make_collect_refs(lambda: tracer)(state)
    assert out["allowed_refs"] == ["tokyo/PACKING TIPS"]


def test_collect_refs_traces_an_unparseable_tool_result_as_an_error():
    """ToolNode emits plain text when the model names a tool that does not exist."""
    tracer = _tracer()
    state = {
        "messages": [
            AIMessage(content=""),
            ToolMessage(name="book_flight", tool_call_id="1",
                        content="Error: book_flight is not a valid tool"),
        ],
        "allowed_refs": [],
    }
    make_collect_refs(lambda: tracer)(state)
    assert any(e.event_type == EventType.TOOL_ERROR for e in tracer.events)


def test_collect_refs_records_a_weather_fallback():
    tracer = _tracer()
    state = {
        "messages": [
            AIMessage(content=""),
            ToolMessage(name="get_weather_forecast", tool_call_id="1",
                        content='{"status": "ok", "data": {"source": "mock_fallback"}}'),
        ],
        "allowed_refs": [],
    }
    make_collect_refs(lambda: tracer)(state)
    assert any(e.event_type == EventType.FALLBACK_USED for e in tracer.events)


# --- validate_citations ---

def test_validate_keeps_a_citation_that_was_retrieved():
    tracer = _tracer()
    state = {
        "messages": [AIMessage(content="Pack layers [tokyo/PACKING TIPS].")],
        "allowed_refs": ["tokyo/PACKING TIPS"],
    }
    out = make_validate_citations(lambda: tracer)(state)
    assert out["citations"] == ["tokyo/PACKING TIPS"]
    assert "[tokyo/PACKING TIPS]" in out["answer"]


def test_validate_strips_a_citation_that_was_never_retrieved():
    tracer = _tracer()
    state = {
        "messages": [AIMessage(content="Pack layers [paris/PACKING TIPS].")],
        "allowed_refs": ["tokyo/PACKING TIPS"],
    }
    out = make_validate_citations(lambda: tracer)(state)
    assert out["citations"] == []
    assert "paris" not in out["answer"]
    assert any(e.event_type == EventType.CITATION_REJECTED for e in tracer.events)


def test_validate_never_returns_an_empty_answer():
    tracer = _tracer()
    out = make_validate_citations(lambda: tracer)(
        {"messages": [AIMessage(content="")], "allowed_refs": []}
    )
    assert out["answer"].strip()


# --- cost estimation ---

def test_cost_is_estimated_for_a_priced_model():
    cost = _estimate_cost("gpt-4o-mini",
                          {"input_tokens": 1000, "output_tokens": 500})
    assert cost > 0


def test_cost_is_zero_for_an_unpriced_model_rather_than_raising():
    assert _estimate_cost("some/unpriced-local-model",
                          {"input_tokens": 100, "output_tokens": 50}) == 0.0


def test_cost_is_zero_without_usage_data():
    assert _estimate_cost("gpt-4o-mini", {}) == 0.0
