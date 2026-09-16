"""Tests for the FastAPI adapter."""

from fastapi.testclient import TestClient
from pydantic import BaseModel

from tripmate.adapters.api import create_app
from tripmate.config import Settings
from tests.fakes import FakeLLMClient
from tripmate.models import LLMResponse
from tripmate.core.agent import Agent
from tripmate.tools.registry import ToolRegistry


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


def test_health_ok(tmp_path, monkeypatch):
    """GET /health returns 200 with status and model info."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = Settings(
        llm_api_key="test-key",
        chroma_path=str(tmp_path / "chroma"),
        database_url=f"sqlite:///{tmp_path}/health.db",
        trace_dir=str(tmp_path / "traces"),
        semantic_cache_enabled=False,
    )

    registry = ToolRegistry()
    fake = FakeLLMClient([])
    agent = Agent(llm=fake, registry=registry, settings=settings)

    app = create_app(agent=agent)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_chat_empty_query_rejected(tmp_path, monkeypatch):
    """POST /chat with empty query returns 422."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = Settings(
        llm_api_key="test-key",
        chroma_path=str(tmp_path / "chroma"),
        database_url=f"sqlite:///{tmp_path}/chat.db",
        trace_dir=str(tmp_path / "traces"),
        semantic_cache_enabled=False,
    )

    registry = ToolRegistry()
    fake = FakeLLMClient([])
    agent = Agent(llm=fake, registry=registry, settings=settings)

    app = create_app(agent=agent)
    client = TestClient(app)

    response = client.post("/chat", json={"query": ""})
    assert response.status_code == 422


def test_chat_success(tmp_path, monkeypatch):
    """POST /chat with valid query returns 200 with agent response."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = Settings(
        llm_api_key="test-key",
        chroma_path=str(tmp_path / "chroma"),
        database_url=f"sqlite:///{tmp_path}/chat2.db",
        trace_dir=str(tmp_path / "traces"),
        semantic_cache_enabled=False,
    )

    registry = ToolRegistry()
    script = [LLMResponse(content="Hello there!")]
    fake = FakeLLMClient(script)
    agent = Agent(llm=fake, registry=registry, settings=settings)

    app = create_app(agent=agent)
    client = TestClient(app)

    response = client.post("/chat", json={"query": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Hello there!"
    assert "session_id" in data


def test_sessions_no_store_returns_404(tmp_path, monkeypatch):
    """/sessions/{id} returns 404 when no store configured."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = Settings(
        llm_api_key="test-key",
        chroma_path=str(tmp_path / "chroma"),
        database_url=f"sqlite:///{tmp_path}/session.db",
        trace_dir=str(tmp_path / "traces"),
        semantic_cache_enabled=False,
    )

    registry = ToolRegistry()
    fake = FakeLLMClient([])
    agent = Agent(llm=fake, registry=registry, settings=settings)

    app = create_app(agent=agent, store=None)
    client = TestClient(app)

    response = client.get("/sessions/abc123")
    assert response.status_code == 404


def test_health_no_store_ok(tmp_path, monkeypatch):
    """GET /health returns 200 even when vector store is unavailable."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = Settings(
        llm_api_key="test-key",
        chroma_path=str(tmp_path / "missing_chroma"),
        database_url=f"sqlite:///{tmp_path}/health_no_store.db",
        trace_dir=str(tmp_path / "traces"),
        semantic_cache_enabled=False,
    )

    registry = ToolRegistry()
    fake = FakeLLMClient([])
    agent = Agent(llm=fake, registry=registry, settings=settings)

    app = create_app(agent=agent, store=None)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
