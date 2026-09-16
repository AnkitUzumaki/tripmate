"""Structured, append-only, replayable reasoning trace."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tripmate.models import TraceEvent


class EventType:
    QUERY_RECEIVED = "QUERY_RECEIVED"
    CACHE_HIT = "CACHE_HIT"
    LLM_CALL = "LLM_CALL"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    TOOL_ERROR = "TOOL_ERROR"
    FALLBACK_USED = "FALLBACK_USED"
    CITATION_REJECTED = "CITATION_REJECTED"
    ANSWER_SYNTHESIZED = "ANSWER_SYNTHESIZED"


@dataclass(frozen=True)
class TraceTotals:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    llm_calls: int
    tool_calls: int


class Tracer:
    """Collects trace events for one session and writes them as JSONL."""

    def __init__(self, session_id: str, trace_dir: str | None = None) -> None:
        self.session_id = session_id
        self._trace_dir = trace_dir
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def record(
        self, event_type: str, duration_ms: float | None = None, **payload: Any
    ) -> TraceEvent:
        event = TraceEvent(
            seq=len(self._events) + 1,
            event_type=event_type,
            timestamp=time.time(),
            duration_ms=duration_ms,
            payload=payload,
        )
        self._events.append(event)
        return event

    def totals(self) -> TraceTotals:
        return TraceTotals(
            prompt_tokens=sum(e.payload.get("prompt_tokens", 0) for e in self._events),
            completion_tokens=sum(
                e.payload.get("completion_tokens", 0) for e in self._events
            ),
            cost_usd=sum(e.payload.get("cost_usd", 0.0) for e in self._events),
            latency_ms=sum(e.duration_ms or 0.0 for e in self._events),
            llm_calls=sum(1 for e in self._events if e.event_type == EventType.LLM_CALL),
            tool_calls=sum(1 for e in self._events if e.event_type == EventType.TOOL_CALL),
        )

    def flush(self) -> Path | None:
        """Append every event to `{trace_dir}/{session_id}.jsonl`."""
        if not self._trace_dir:
            return None

        directory = Path(self._trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.session_id}.jsonl"

        with path.open("a", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(json.dumps(event.model_dump(), default=str) + "\n")
        return path


def load_trace(path: str | Path) -> list[TraceEvent]:
    """Read a JSONL trace back into events, for offline replay and auditing."""
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    return [TraceEvent(**json.loads(line)) for line in lines if line]
