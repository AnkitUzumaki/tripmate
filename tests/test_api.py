"""Tests for the FastAPI adapter."""

import pytest
from fastapi.testclient import TestClient

from tripmate.adapters.api import create_app
from tripmate.core.agent import Agent
from tripmate.tools.registry import ToolRegistry, tool
from tripmate.models import ToolResult
from typing import Annotated

from tests.fakes import FakeChatModel, ai


@tool
def stub_guide(query: Annotated[str, "topic"]) -> ToolResult:
    """Search the guide."""
    return ToolResult.ok("stub_guide", {"chunks": [
        {"ref": "tokyo/VISA & ENTRY", "city": "tokyo", "section": "VISA & ENTRY",
         "text": "Visa-free for many nationalities.", "score": 0.9}
    ]})


def _client(script) -> TestClient:
    """App wired to a real graph with a scripted model and one stub tool."""
    from tripmate.core.agent import Agent
    from tripmate.graph.builder import bind_model, build_graph

    registry = ToolRegistry()
    registry.register(stub_guide)
    fake = FakeChatModel(script)
    holder: dict = {}
    graph = build_graph(
        model=bind_model(fake, registry),
        registry=registry,
        tracer_of=lambda: holder["agent"].current_tracer(),
    )
    agent = Agent(graph=graph, registry=registry)
    holder["agent"] = agent
    return TestClient(create_app(agent=agent))


def test_health_reports_ok():
    response = _client([]).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_the_registered_tools():
    body = _client([]).get("/health").json()
    assert body["tools"] == ["stub_guide"]


def test_chat_returns_the_answer():
    client = _client([ai("Hello from TripMate.")])
    body = client.post("/chat", json={"query": "hello"}).json()

    assert body["answer"] == "Hello from TripMate."


def test_chat_returns_citations_and_trace():
    client = _client([
        ai(tool_calls=[{"name": "stub_guide", "args": {"query": "visa"}, "id": "1"}]),
        ai("Visa-free [tokyo/VISA & ENTRY]."),
    ])
    body = client.post("/chat", json={"query": "visa for Japan?"}).json()

    assert body["citations"] == ["tokyo/VISA & ENTRY"]
    assert len(body["trace"]) > 0


def test_chat_echoes_the_session_id():
    client = _client([ai("hi")])
    body = client.post("/chat", json={"query": "hi", "session_id": "abc123"}).json()

    assert body["session_id"] == "abc123"


def test_chat_reports_cost_and_latency():
    client = _client([ai("hi", prompt_tokens=10, completion_tokens=2)])
    body = client.post("/chat", json={"query": "hi"}).json()

    assert body["prompt_tokens"] == 10
    assert body["latency_ms"] >= 0


def test_missing_query_field_is_rejected_with_422():
    assert _client([]).post("/chat", json={}).status_code == 422


def test_empty_query_string_is_rejected_with_422():
    assert _client([]).post("/chat", json={"query": ""}).status_code == 422


def test_sessions_endpoint_returns_404_without_a_store():
    assert _client([]).get("/sessions/nope").status_code == 404
