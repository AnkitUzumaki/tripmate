"""Single wiring point. Both adapters build the agent from here, so they cannot drift."""

from __future__ import annotations

from tripmate.config import Settings, get_settings
from tripmate.core.agent import Agent
from tripmate.core.cache import build_cache
from tripmate.core.trace import Tracer
from tripmate.db import SessionStore, build_database
from tripmate.llm.client import build_llm_client
from tripmate.rag.ingest import ingest
from tripmate.rag.store import build_store
from tripmate.tools.destination import search_destination_guide, set_store
from tripmate.tools.registry import ToolRegistry
from tripmate.tools.weather import get_weather_forecast


def build_agent(settings: Settings | None = None) -> Agent:
    """Build and configure the agent with all components."""
    settings = settings or get_settings()

    ingest(settings=settings)          # idempotent: a no-op once the pack is loaded
    set_store(build_store(settings))

    registry = ToolRegistry()
    registry.register(search_destination_guide)
    registry.register(get_weather_forecast)

    database = build_database(settings)

    return Agent(
        llm=build_llm_client(settings),
        registry=registry,
        tracer_factory=lambda sid: Tracer(session_id=sid, trace_dir=settings.trace_dir),
        store=SessionStore(database),
        cache=build_cache(database, settings),
        settings=settings,
    )
