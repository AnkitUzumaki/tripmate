"""Shared test fixtures for tool selection and integration tests.

This module provides:
- A real ChromaStore injected into tools for all tests that need it
- A factory fixture to build test agents with scripted responses
- A helper to extract tool names from trace events
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from tripmate.core.agent import Agent
from tripmate.core.trace import EventType
from tripmate.graph.builder import bind_model, build_graph
from tripmate.models import AgentResponse
from tripmate.rag.chunker import load_all
from tripmate.rag.store import ChromaStore
from tripmate.tools.destination import (
    reset_store, search_destination_guide, set_store,
)
from tripmate.tools.registry import ToolRegistry
from tripmate.tools.weather import clear_cache, get_weather_forecast

from langchain_core.messages import AIMessage

from tests.fakes import FakeChatModel


@pytest.fixture(scope="session", autouse=True)
def _real_store(tmp_path_factory):
    """Session-scoped shared ChromaStore injected into destination and weather tools.

    Loads all 20 destination chunks once per test session, then all tests
    that call the tools get a populated store without needing to rebuild it.
    """
    store = ChromaStore(
        path=str(tmp_path_factory.mktemp("chroma_sel")),
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    store.add(load_all(Path("data/destinations")))
    set_store(store)
    clear_cache()
    yield
    reset_store()
    clear_cache()


@pytest.fixture(autouse=True)
def _isolate_weather_cache():
    """Clear the weather disk cache between tests.

    Climate normals cache permanently by design, so without this a test that populates
    Tokyo/December makes a later timeout test unable to time out — there is nothing left
    to call. Function-scoped because the pollution crosses files.
    """
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def build_test_agent() -> Callable[[list[AIMessage]], tuple[Agent, FakeChatModel]]:
    """Build an agent over a REAL compiled graph with a scripted model.

    Only the model is fake: the graph, its nodes, its edges, ToolNode and both real
    tools all execute. That is what makes these tests meaningful rather than mocks
    asserting on mocks.

        agent, fake = build_test_agent([ai(tool_calls=[...]), ai("answer")])
        response = agent.chat("query")

    No checkpointer: each test gets a fresh single-turn graph, so tests cannot leak
    conversation state into one another. Multi-turn tests build their own.
    """
    def _build(script: list[AIMessage]) -> tuple[Agent, FakeChatModel]:
        registry = ToolRegistry()
        registry.register(search_destination_guide)
        registry.register(get_weather_forecast)

        fake = FakeChatModel(script)
        holder: dict[str, Agent] = {}
        graph = build_graph(
            model=bind_model(fake, registry),
            registry=registry,
            tracer_of=lambda: holder["agent"].current_tracer(),
        )
        agent = Agent(graph=graph, registry=registry)
        holder["agent"] = agent
        return agent, fake
    return _build


@pytest.fixture
def tools_called() -> Callable[[AgentResponse], list[str]]:
    """Helper fixture that extracts tool names from a response's trace events.

    Usage:
        response = agent.chat("query")
        names = tools_called(response)  # e.g. ["search_destination_guide"]
    """
    def _extract(response: AgentResponse) -> list[str]:
        # The graph's ToolNode executes tools; what the model *chose* is recorded on
        # the LLM_CALL event, and what actually ran on TOOL_RESULT. Report the union,
        # in first-seen order, so assertions read the same as before the port.
        names: list[str] = []
        for event in response.trace:
            if event.event_type == EventType.LLM_CALL:
                names.extend(event.payload.get("requested_tools") or [])
            elif event.event_type in (EventType.TOOL_RESULT, EventType.TOOL_ERROR):
                tool = event.payload.get("tool")
                if tool:
                    names.append(tool)
        seen: list[str] = []
        for n in names:
            if n not in seen:
                seen.append(n)
        return seen
    return _extract
