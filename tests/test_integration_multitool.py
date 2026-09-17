"""Full multi-tool flow: real RAG, real weather (mocked HTTP), real DB, real cache."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from tripmate.core.agent import Agent
from tripmate.core.cache import SemanticCache
from tripmate.core.trace import EventType, Tracer
from tripmate.db import Database, SessionStore
from tripmate.rag.chunker import load_all
from tripmate.rag.store import ChromaStore
from tripmate.tools.destination import reset_store, search_destination_guide, set_store
from tripmate.tools.registry import ToolRegistry
from tripmate.tools.weather import ARCHIVE_URL, GEOCODE_URL, clear_cache
from tripmate.tools.weather import get_weather_forecast

from tests.fakes import FakeChatModel, ai

GEOCODE_TOKYO = {"results": [{"latitude": 35.68, "longitude": 139.75, "name": "Tokyo"}]}
ARCHIVE_DECEMBER = {
    "daily": {
        "time": ["2021-12-01", "2021-12-02", "2022-12-01", "2022-12-02"],
        "temperature_2m_max": [12.0, 11.0, 13.0, 10.0],
        "temperature_2m_min": [4.0, 3.0, 5.0, 2.0],
        "precipitation_sum": [0.0, 0.0, 2.0, 0.0],
    }
}

PACKING_SCRIPT = [
    ai(tool_calls=[
            {"name": "search_destination_guide", "args": {"query": "packing tips", "city": "tokyo"}, "id": "1"},
            {"name": "get_weather_forecast", "args": {"city": "Tokyo", "date_or_month": "December"}, "id": "2"},
        ], prompt_tokens=400, completion_tokens=60),
    ai("Pack warm layers and comfortable walking shoes "
                "[tokyo/PACKING TIPS]. December in Tokyo is typically cold and "
                "mostly dry, roughly 3 to 12 C.",
       prompt_tokens=900, completion_tokens=80),
]


@pytest.fixture()
def agent(tmp_path):
    """Local fixture: real Database, SemanticCache, Tracer, and agent wiring.

    This test needs its own agent configuration with a real SQLite database,
    a real semantic cache, and a tracer writing to disk. We build everything
    fresh here so this test is self-contained.
    """
    # Set up the store for this test
    store = ChromaStore(path=str(tmp_path / "chroma"),
                        embedding_model="BAAI/bge-small-en-v1.5")
    store.add(load_all(Path("data/destinations")))
    set_store(store)
    clear_cache()

    # Real database and session store
    db = Database(url=f"sqlite:///{tmp_path}/it.db")
    db.create_all()

    # Tool registry with both tools
    registry = ToolRegistry()
    registry.register(search_destination_guide)
    registry.register(get_weather_forecast)

    # Build over a REAL compiled graph with a real checkpointer — only the model is
    # scripted. Everything else in this test is the production object.
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3

    from tripmate.graph.builder import bind_model, build_graph

    cache = SemanticCache(db=db, embedder=store.embedder, threshold=0.95)
    fake = FakeChatModel(PACKING_SCRIPT)
    holder: dict = {}
    graph = build_graph(
        model=bind_model(fake, registry),
        registry=registry,
        tracer_of=lambda: holder["agent"].current_tracer(),
        cache=cache,
        checkpointer=SqliteSaver(
            sqlite3.connect(str(tmp_path / "graph.db"), check_same_thread=False)
        ),
    )
    built = Agent(
        graph=graph,
        registry=registry,
        tracer_factory=lambda sid: Tracer(session_id=sid,
                                          trace_dir=str(tmp_path / "traces")),
    )
    holder["agent"] = built

    yield built
    reset_store()
    clear_cache()


@respx.mock
def test_packing_query_calls_both_tools_and_synthesises_one_answer(agent):
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_TOKYO))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=ARCHIVE_DECEMBER))

    response = agent.chat("What should I pack for Tokyo in December?")

    tools = {e.payload["tool"] for e in response.trace
             if e.event_type == EventType.TOOL_CALL}
    assert tools == {"search_destination_guide", "get_weather_forecast"}
    assert "layers" in response.answer


@respx.mock
def test_answer_cites_a_chunk_that_was_actually_retrieved(agent):
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_TOKYO))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=ARCHIVE_DECEMBER))

    response = agent.chat("What should I pack for Tokyo in December?")

    assert [c.ref for c in response.citations] == ["tokyo/PACKING TIPS"]


@respx.mock
def test_cost_and_tokens_accumulate_across_both_llm_calls(agent):
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_TOKYO))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=ARCHIVE_DECEMBER))

    response = agent.chat("What should I pack for Tokyo in December?")

    assert response.prompt_tokens == 1300
    assert response.completion_tokens == 140
    # Cost is derived from the price map rather than injected by the fake, so a
    # nameless test model is legitimately free. _estimate_cost is tested directly
    # in tests/test_graph_nodes.py against a priced model.
    assert response.cost_usd == 0.0


@respx.mock
def test_trace_is_written_to_disk_as_jsonl(agent, tmp_path):
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_TOKYO))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=ARCHIVE_DECEMBER))

    response = agent.chat("What should I pack for Tokyo in December?")

    written = list((tmp_path / "traces").glob("*.jsonl"))
    assert written and response.session_id in written[0].name


@respx.mock
def test_repeating_the_query_is_served_from_the_semantic_cache(agent):
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_TOKYO))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=ARCHIVE_DECEMBER))

    first = agent.chat("What should I pack for Tokyo in December?")
    second = agent.chat("What should I pack for Tokyo in December?")

    assert second.was_cached is True
    assert second.answer == first.answer
    assert any(e.event_type == EventType.CACHE_HIT for e in second.trace)
