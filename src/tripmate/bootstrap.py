"""Single wiring point. Both adapters build the agent from here, so they cannot drift."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from tripmate.config import PROVIDER_OPENAI_COMPATIBLE_URLS, Settings, get_settings
from tripmate.core.agent import Agent
from tripmate.core.cache import build_cache
from tripmate.core.trace import Tracer
from tripmate.db import build_database
from tripmate.graph.builder import bind_model, build_graph
from tripmate.rag.ingest import ingest
from tripmate.rag.store import build_store
from tripmate.tools.destination import search_destination_guide, set_store
from tripmate.tools.registry import ToolRegistry
from tripmate.tools.weather import get_weather_forecast


def build_chat_model(settings: Settings | None = None):
    """A LangChain chat model for whichever provider is configured.

    Two paths, tried in order:

    1. `init_chat_model` — LangChain's provider-agnostic factory. Works natively for any
       provider whose integration package is installed (langchain-anthropic,
       langchain-groq, langchain-ollama, ...), giving first-class support rather than
       an OpenAI-shaped approximation.
    2. `ChatOpenAI` pointed at the provider's OpenAI-compatible endpoint. Every provider
       in PROVIDER_OPENAI_COMPATIBLE_URLS exposes one, so this covers the rest without
       requiring an extra dependency per provider.

    Either way `LLM_MODEL` alone selects the provider, which is what keeps the graph
    itself provider-agnostic.
    """
    settings = settings or get_settings()
    settings.export_provider_key()
    provider = settings.provider
    bare_model = settings.llm_model.split("/", 1)[-1]

    try:
        from langchain.chat_models import init_chat_model

        return init_chat_model(
            model=bare_model,
            model_provider=provider,
            temperature=settings.llm_temperature,
        )
    except Exception:
        # No integration package for this provider — fall back to its
        # OpenAI-compatible endpoint.
        pass

    base_url = settings.llm_base_url or PROVIDER_OPENAI_COMPATIBLE_URLS.get(provider)
    return ChatOpenAI(
        model=bare_model,
        base_url=base_url,
        api_key=settings.llm_api_key or "not-needed",
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
    )


def build_checkpointer(settings: Settings | None = None) -> SqliteSaver:
    """Conversation state, owned by LangGraph rather than hand-written turn writes.

    `check_same_thread=False` because the graph may touch the connection from a worker
    thread during concurrent tool dispatch.
    """
    settings = settings or get_settings()
    path = Path(settings.database_url.replace("sqlite:///", "")).with_suffix(".graph.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def build_agent(settings: Settings | None = None) -> Agent:
    settings = settings or get_settings()

    ingest(settings=settings)          # idempotent: a no-op once the pack is loaded
    set_store(build_store(settings))

    registry = ToolRegistry()
    registry.register(search_destination_guide)
    registry.register(get_weather_forecast)

    database = build_database(settings)
    cache = build_cache(database, settings)

    agent_ref: dict[str, Agent] = {}
    graph = build_graph(
        model=bind_model(build_chat_model(settings), registry),
        registry=registry,
        # Nodes ask the agent for the tracer of the turn in flight, so they need no
        # knowledge of sessions and the agent stays the owner of per-turn state.
        tracer_of=lambda: agent_ref["agent"].current_tracer(),
        cache=cache,
        checkpointer=build_checkpointer(settings),
        model_name=settings.llm_model,
    )

    agent = Agent(
        graph=graph,
        registry=registry,
        tracer_factory=lambda sid: Tracer(session_id=sid, trace_dir=settings.trace_dir),
        settings=settings,
    )
    agent_ref["agent"] = agent
    return agent
