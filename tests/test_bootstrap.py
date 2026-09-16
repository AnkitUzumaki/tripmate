"""Tests for the shared bootstrap wiring point."""

from tripmate.bootstrap import build_agent
from tripmate.config import Settings


def test_build_agent_registers_both_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = Settings(
        llm_api_key="test-key",
        chroma_path=str(tmp_path / "chroma"),
        database_url=f"sqlite:///{tmp_path}/boot.db",
        trace_dir=str(tmp_path / "traces"),
        semantic_cache_enabled=False,
    )

    agent = build_agent(settings=settings)

    assert set(agent._registry.names()) == {
        "search_destination_guide", "get_weather_forecast"
    }


def test_build_agent_ingests_the_data_pack(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = Settings(
        llm_api_key="test-key",
        chroma_path=str(tmp_path / "chroma2"),
        database_url=f"sqlite:///{tmp_path}/boot2.db",
        trace_dir=str(tmp_path / "traces"),
        semantic_cache_enabled=False,
    )

    build_agent(settings=settings)

    from tripmate.rag.store import build_store
    assert build_store(settings).count() == 20
