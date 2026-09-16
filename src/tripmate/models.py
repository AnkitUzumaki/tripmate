"""Shared domain models. Every boundary in the system speaks these types."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

CITATION_RE = re.compile(r"\[([A-Za-z\s\-']+)/([A-Za-z\s&'\-]+)\]")

ToolStatus = Literal["ok", "no_data", "error"]
WeatherSource = Literal["forecast", "climate_normal", "mock_fallback"]


class Chunk(BaseModel):
    """One retrievable section of a destination guide."""

    id: str
    text: str
    city: str
    section: str
    source_file: str
    score: float = 0.0

    @property
    def ref(self) -> str:
        return f"{self.city}/{self.section}"


class WeatherReport(BaseModel):
    city: str
    period: str
    source: WeatherSource
    conditions: str
    temp_range_c: list[float]
    precip_days: int


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Tools always return this. They never raise into the agent loop."""

    tool_name: str
    status: ToolStatus
    data: dict[str, Any] | None = None
    reason: str | None = None

    @classmethod
    def ok(cls, tool_name: str, data: dict[str, Any]) -> ToolResult:
        return cls(tool_name=tool_name, status="ok", data=data)

    @classmethod
    def no_data(cls, tool_name: str, reason: str, **extra: Any) -> ToolResult:
        return cls(
            tool_name=tool_name, status="no_data", reason=reason,
            data=dict(extra) if extra else None,
        )

    @classmethod
    def error(cls, tool_name: str, reason: str) -> ToolResult:
        return cls(tool_name=tool_name, status="error", reason=reason)


class Citation(BaseModel):
    city: str
    section: str

    @property
    def ref(self) -> str:
        return f"{self.city}/{self.section}"

    @classmethod
    def parse(cls, text: str) -> Citation | None:
        """Parse a single '[city/SECTION]' reference. Returns None when malformed."""
        match = CITATION_RE.search(text)
        if match is None:
            return None
        return cls(city=match.group(1).strip().lower(),
                   section=match.group(2).strip().upper())


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class TraceEvent(BaseModel):
    seq: int
    event_type: str
    timestamp: float
    duration_ms: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    session_id: str = ""
    was_cached: bool = False
