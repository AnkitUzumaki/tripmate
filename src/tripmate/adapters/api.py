"""FastAPI adapter for the agent.

Provides HTTP endpoints wrapping the same Agent that the CLI uses.
Imports are deferred to enable lazy agent construction (no side effects on module load).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tripmate.bootstrap import build_agent
from tripmate.config import Settings
from tripmate.core.agent import Agent
from tripmate.db import SessionStore
from tripmate.models import AgentResponse


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    trace: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    session_id: str
    was_cached: bool


class HealthResponse(BaseModel):
    status: str
    model: str | None = None
    tools: list[str] | None = None
    chunk_count: int | None = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    turns: list[dict[str, Any]] = Field(default_factory=list)


_active_agent: Agent | None = None
_active_store: SessionStore | None = None


def create_app(
    agent: Agent | None = None, store: SessionStore | None = None
) -> FastAPI:
    """Create FastAPI app with injected or lazily-built agent.

    If agent is not provided, it is built on first request to avoid side effects
    on module import. The store parameter is optional; if None, session endpoints
    will return 404.
    """
    global _active_agent, _active_store
    _active_agent = agent
    _active_store = store

    app = FastAPI(title="TripMate API")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report service health, optionally with model name and tool list."""
        active = _ensure_agent()
        model = active.settings.llm_model
        tools = active.registry.names()

        chunk_count = None
        if _active_store:
            try:
                chunk_count = _active_store.count()
            except Exception:
                chunk_count = None

        return HealthResponse(status="ok", model=model, tools=tools, chunk_count=chunk_count)

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """Accept a user query and return the agent's response."""
        active = _ensure_agent()
        result = active.chat(request.query, session_id=request.session_id)
        return ChatResponse(
            answer=result.answer,
            citations=[c.ref for c in result.citations],
            trace=[event.model_dump() for event in result.trace],
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            session_id=result.session_id,
            was_cached=result.was_cached,
        )

    @app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
    def get_session(session_id: str) -> SessionHistoryResponse:
        """Retrieve the history of a session."""
        if not _active_store:
            raise HTTPException(status_code=404, detail="Session store not configured")

        history = _active_store.history(session_id)
        if not history:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionHistoryResponse(session_id=session_id, turns=history)

    return app


def _ensure_agent() -> Agent:
    """Get the active agent, building it lazily if needed."""
    global _active_agent
    if _active_agent is None:
        _active_agent = build_agent()
    return _active_agent


# Module-level ASGI app for `uvicorn tripmate.adapters.api:app` (Dockerfile CMD and
# the README quickstart). Building it here is side-effect free: create_app() only
# registers routes, the agent itself is constructed lazily on first request.
app = create_app()
