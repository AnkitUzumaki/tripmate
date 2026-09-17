"""Agent facade and its pure helpers.

The orchestration itself is a LangGraph StateGraph, so behaviour that used to be
reached through private methods is now tested either at node level
(`test_graph_nodes.py`) or through a real compiled graph here.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from tripmate.core.agent import (
    Agent,
    EMPTY_ANSWER_FALLBACK,
    InvalidQuery,
    extract_citations,
    validate_citations,
    validate_query,
)
from tripmate.core.cache import SemanticCache
from tripmate.core.trace import EventType, Tracer
from tripmate.db import Database
from tripmate.graph.builder import bind_model, build_graph
from tripmate.models import ToolResult
from tripmate.tools.registry import ToolRegistry, tool

from tests.fakes import FakeChatModel, ai


# --- stub tools: fast and deterministic, so these tests exercise the graph ---

@tool
def fake_guide(
    query: Annotated[str, "topic"],
    city: Annotated[str | None, "city"] = None,
) -> ToolResult:
    """Search the guide."""
    return ToolResult.ok("fake_guide", {"chunks": [
        {"ref": "tokyo/PACKING TIPS", "city": "tokyo", "section": "PACKING TIPS",
         "text": "Layered clothing works well.", "score": 0.8}
    ]})


@tool
def fake_weather(
    city: Annotated[str, "city"],
    date_or_month: Annotated[str, "month"],
) -> ToolResult:
    """Get weather."""
    return ToolResult.ok("fake_weather", {
        "city": "Tokyo", "period": "December", "source": "climate_normal",
        "conditions": "cold, mostly dry", "temp_range_c": [3.0, 12.0],
        "precip_days": 4,
    })


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(fake_guide)
    registry.register(fake_weather)
    return registry


def _agent(script, cache=None, checkpointer=None) -> tuple[Agent, FakeChatModel]:
    registry = _registry()
    fake = FakeChatModel(script)
    holder: dict = {}
    graph = build_graph(
        model=bind_model(fake, registry),
        registry=registry,
        tracer_of=lambda: holder["agent"].current_tracer(),
        cache=cache,
        checkpointer=checkpointer,
    )
    agent = Agent(graph=graph, registry=registry)
    holder["agent"] = agent
    return agent, fake



class _StubEmbedder:
    """Maps known phrases to fixed orthogonal vectors.

    Deliberately explicit rather than hashing: a modulo hash collides across different
    queries, which silently turns a cache miss into a hit and makes these tests assert
    the opposite of what they intend.
    """

    VECTORS = {
        "what should i pack?": [1.0, 0.0, 0.0],
        "i am going to tokyo in december.": [0.0, 1.0, 0.0],
        "do i need a visa?": [0.0, 0.0, 1.0],
    }

    def embed(self, texts):
        return [self.VECTORS.get(t.strip().lower(), [0.5, 0.5, 0.5]) for t in texts]


def _tools_called(response) -> list[str]:
    return [e.payload["tool"] for e in response.trace
            if e.event_type == EventType.TOOL_CALL]


# --- input validation (pure) ---

def test_validate_query_strips_surrounding_whitespace():
    assert validate_query("  hello  ", max_chars=100) == "hello"


def test_validate_query_rejects_empty_input():
    with pytest.raises(InvalidQuery, match="empty"):
        validate_query("   ", max_chars=100)


def test_validate_query_rejects_overlong_input():
    with pytest.raises(InvalidQuery, match="too long"):
        validate_query("x" * 101, max_chars=100)


# --- citations (pure) ---

def test_extract_citations_finds_every_reference():
    refs = extract_citations("See [tokyo/PACKING TIPS] and [tokyo/SAFETY & HEALTH].")
    assert [c.ref for c in refs] == ["tokyo/PACKING TIPS", "tokyo/SAFETY & HEALTH"]


def test_validate_citations_keeps_references_that_were_retrieved():
    answer = "Pack layers [tokyo/PACKING TIPS]."
    cleaned, kept, rejected = validate_citations(
        answer, extract_citations(answer), {"tokyo/PACKING TIPS"}
    )
    assert kept[0].ref == "tokyo/PACKING TIPS"
    assert rejected == []


def test_validate_citations_strips_references_never_retrieved():
    answer = "Pack layers [osaka/PACKING TIPS]."
    cleaned, kept, rejected = validate_citations(
        answer, extract_citations(answer), {"tokyo/PACKING TIPS"}
    )
    assert kept == []
    assert rejected == ["osaka/PACKING TIPS"]
    assert "[osaka/PACKING TIPS]" not in cleaned


def test_rejected_citations_are_stripped_regardless_of_casing():
    for answer in ("Layers [paris/PACKING TIPS].",
                   "Layers [Paris/Packing Tips].",
                   "Layers [paris/packing tips]."):
        cleaned, kept, rejected = validate_citations(
            answer, extract_citations(answer), {"tokyo/PACKING TIPS"}
        )
        assert rejected == ["paris/PACKING TIPS"]
        assert "[" not in cleaned


def test_stripping_a_citation_removes_the_phrase_that_introduced_it():
    answer = "Barcelona is mild. Source: [paris/BEST TIME TO VISIT]"
    cleaned, _, _ = validate_citations(
        answer, extract_citations(answer), {"barcelona/BEST TIME TO VISIT"}
    )
    assert cleaned == "Barcelona is mild."


def test_invented_citation_markers_are_stripped_not_left_in_the_answer():
    for answer in ("Cold in January 【reykjavik/climate_normal】.",
                   "Layers [Tokyo/climate_normal]."):
        cleaned, kept, _ = validate_citations(
            answer, extract_citations(answer), {"tokyo/PACKING TIPS"}
        )
        assert "[" not in cleaned and "【" not in cleaned
        assert kept == []


def test_a_valid_citation_survives_alongside_an_invented_one():
    answer = "Layers [Tokyo/climate_normal] and shoes [ tokyo/PACKING TIPS ]."
    cleaned, kept, _ = validate_citations(
        answer, extract_citations(answer), {"tokyo/PACKING TIPS"}
    )
    assert [c.ref for c in kept] == ["tokyo/PACKING TIPS"]
    assert "climate_normal" not in cleaned


# --- the graph, end to end ---

def test_no_tool_query_returns_the_model_answer_directly():
    agent, _ = _agent([ai("I help with travel questions.")])
    assert agent.chat("what can you do?").answer == "I help with travel questions."


def test_no_tool_query_makes_exactly_one_model_call():
    agent, fake = _agent([ai("hello")])
    agent.chat("hi")
    assert len(fake.calls) == 1


def test_single_tool_query_calls_that_tool_then_answers():
    agent, _ = _agent([
        ai(tool_calls=[{"name": "fake_guide",
                        "args": {"query": "visa", "city": "tokyo"}, "id": "1"}]),
        ai("No visa needed [tokyo/PACKING TIPS]."),
    ])
    assert _tools_called(agent.chat("visa for Tokyo?")) == ["fake_guide"]


def test_multi_tool_query_calls_both_tools_in_one_turn():
    agent, _ = _agent([
        ai(tool_calls=[
            {"name": "fake_guide", "args": {"query": "packing", "city": "tokyo"}, "id": "1"},
            {"name": "fake_weather", "args": {"city": "Tokyo", "date_or_month": "December"}, "id": "2"},
        ]),
        ai("Pack warm layers [tokyo/PACKING TIPS]."),
    ])
    assert set(_tools_called(agent.chat("what to pack?"))) == {"fake_guide", "fake_weather"}


def test_tool_results_reach_the_model_as_tool_messages():
    agent, fake = _agent([
        ai(tool_calls=[{"name": "fake_weather",
                        "args": {"city": "Tokyo", "date_or_month": "December"}, "id": "1"}]),
        ai("Cold."),
    ])
    agent.chat("weather?")
    assert any(isinstance(m, ToolMessage) for m in fake.calls[1])


def test_tool_call_ids_stay_matched_to_their_own_result():
    """pool ordering must not misattribute a result to the wrong call."""
    agent, fake = _agent([
        ai(tool_calls=[
            {"name": "fake_guide", "args": {"query": "packing", "city": "tokyo"}, "id": "call_a"},
            {"name": "fake_weather", "args": {"city": "Tokyo", "date_or_month": "December"}, "id": "call_b"},
        ]),
        ai("done"),
    ])
    agent.chat("what to pack?")

    tool_messages = [m for m in fake.calls[1] if isinstance(m, ToolMessage)]
    by_id = {m.tool_call_id: m for m in tool_messages}
    assert "fake_guide" in by_id["call_a"].content
    assert "climate_normal" in by_id["call_b"].content


def test_valid_citations_are_returned_on_the_response():
    agent, _ = _agent([
        ai(tool_calls=[{"name": "fake_guide", "args": {"query": "packing"}, "id": "1"}]),
        ai("Layers [tokyo/PACKING TIPS]."),
    ])
    assert [c.ref for c in agent.chat("packing?").citations] == ["tokyo/PACKING TIPS"]


def test_hallucinated_citations_are_stripped_and_traced():
    agent, _ = _agent([
        ai(tool_calls=[{"name": "fake_guide", "args": {"query": "packing"}, "id": "1"}]),
        ai("Layers [paris/PACKING TIPS]."),
    ])
    response = agent.chat("packing for Paris?")
    assert response.citations == []
    assert "[paris/PACKING TIPS]" not in response.answer
    assert any(e.event_type == EventType.CITATION_REJECTED for e in response.trace)


def test_empty_input_is_rejected_before_any_model_call():
    agent, fake = _agent([])
    response = agent.chat("   ")
    assert "empty" in response.answer.lower()
    assert fake.calls == []


def test_trace_records_tokens_from_model_calls():
    agent, _ = _agent([ai("hi", prompt_tokens=100, completion_tokens=20)])
    response = agent.chat("hello")
    assert response.prompt_tokens == 100
    assert response.completion_tokens == 20


def test_recursion_ceiling_returns_a_fallback_rather_than_raising():
    """A model that never stops asking for tools must not hang or crash the turn."""
    looping = [
        ai(tool_calls=[{"name": "fake_weather",
                        "args": {"city": "Tokyo", "date_or_month": "December"},
                        "id": str(i)}])
        for i in range(40)
    ]
    agent, _ = _agent(looping)
    response = agent.chat("weather?")
    assert response.answer == EMPTY_ANSWER_FALLBACK


# --- conversation state via the checkpointer ---

def test_multi_turn_history_reaches_the_model(tmp_path):
    saver = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False))
    agent, fake = _agent([ai("Tokyo is great."), ai("Pack layers.")], checkpointer=saver)

    agent.chat("Tell me about Tokyo", session_id="s1")
    agent.chat("What should I pack?", session_id="s1")

    second_call = fake.calls[1]
    assert any(isinstance(m, AIMessage) and "Tokyo is great" in str(m.content)
               for m in second_call), "prior answer must be in the second turn's context"


def test_follow_up_does_not_replay_a_standalone_cached_answer(tmp_path):
    """A follow-up is not a standalone question; the cache must not answer it."""
    db = Database(url=f"sqlite:///{tmp_path}/c.db")
    db.create_all()

    cache = SemanticCache(db=db, embedder=_StubEmbedder(), threshold=0.95)
    saver = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False))

    standalone, _ = _agent([ai("Which destination?")], cache=cache)
    standalone.chat("What should I pack?", session_id="alone")

    agent, fake = _agent([ai("Noted, Tokyo."), ai("Pack warm layers.")],
                         cache=cache, checkpointer=saver)
    agent.chat("I am going to Tokyo in December.", session_id="ctx")
    response = agent.chat("What should I pack?", session_id="ctx")

    assert not any(e.event_type == EventType.CACHE_HIT for e in response.trace)
    assert response.answer == "Pack warm layers."


def test_first_turn_repeat_still_hits_the_cache(tmp_path):
    db = Database(url=f"sqlite:///{tmp_path}/c2.db")
    db.create_all()

    cache = SemanticCache(db=db, embedder=_StubEmbedder(), threshold=0.95)

    first, _ = _agent([ai("Visa-free for 90 days.")], cache=cache)
    first.chat("Do I need a visa?", session_id="one")

    # Exhausted script: a hit is the only way this can succeed.
    second, fake = _agent([], cache=cache)
    response = second.chat("Do I need a visa?", session_id="two")

    assert any(e.event_type == EventType.CACHE_HIT for e in response.trace)
    assert fake.calls == []
