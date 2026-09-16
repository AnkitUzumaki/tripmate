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
from tripmate.models import AgentResponse, LLMResponse
from tripmate.rag.chunker import load_all
from tripmate.rag.store import ChromaStore
from tripmate.tools.destination import (
    reset_store, search_destination_guide, set_store,
)
from tripmate.tools.registry import ToolRegistry
from tripmate.tools.weather import clear_cache, get_weather_forecast

from tests.fakes import FakeLLMClient


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


@pytest.fixture
def build_test_agent() -> Callable[[list[LLMResponse]], tuple[Agent, FakeLLMClient]]:
    """Factory fixture that builds a test agent with real tools and a scripted model.

    Each test receives the factory and calls it with an LLMResponse script:
        agent, fake = build_test_agent([LLMResponse(...), ...])
        response = agent.chat("query")

    The agent wires both real tools (search_destination_guide and get_weather_forecast)
    onto a fresh registry, so tool execution is real while the LLM responses are scripted.
    """
    def _build(script: list[LLMResponse]) -> tuple[Agent, FakeLLMClient]:
        registry = ToolRegistry()
        registry.register(search_destination_guide)
        registry.register(get_weather_forecast)
        fake = FakeLLMClient(script)
        return Agent(llm=fake, registry=registry), fake
    return _build


@pytest.fixture
def tools_called() -> Callable[[AgentResponse], list[str]]:
    """Helper fixture that extracts tool names from a response's trace events.

    Usage:
        response = agent.chat("query")
        names = tools_called(response)  # e.g. ["search_destination_guide"]
    """
    def _extract(response: AgentResponse) -> list[str]:
        return [e.payload["tool"] for e in response.trace
                if e.event_type == EventType.TOOL_CALL]
    return _extract
