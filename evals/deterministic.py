"""Layer 1: deterministic scoring. No LLM judging — exact assertions only."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tripmate.core.trace import EventType
from tripmate.models import AgentResponse

REFUSAL_PHRASES = (
    "can't", "cannot", "unable to", "not able to", "outside what i",
    "i don't have the ability", "i'm not able",
)


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    expected_tools: list[str]
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    must_cite: list[str] = field(default_factory=list)
    must_refuse: bool = False
    session_id: str | None = None


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    tool_selection_correct: bool
    citations_valid: bool
    refusal_correct: bool
    mentions_ok: bool
    latency_ms: float
    cost_usd: float
    tools_called: list[str]
    answer: str

    @property
    def passed(self) -> bool:
        return (
            self.tool_selection_correct
            and self.citations_valid
            and self.refusal_correct
            and self.mentions_ok
        )


@dataclass(frozen=True)
class DeterministicReport:
    rows: list[CaseScore]

    @property
    def accuracy(self) -> float:
        return self._ratio([row.tool_selection_correct for row in self.rows])

    @property
    def citation_rate(self) -> float:
        return self._ratio([row.citations_valid for row in self.rows])

    @property
    def refusal_accuracy(self) -> float:
        return self._ratio([row.refusal_correct for row in self.rows])

    @property
    def pass_rate(self) -> float:
        return self._ratio([row.passed for row in self.rows])

    @property
    def p50_latency_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_latency_ms(self) -> float:
        return self._percentile(95)

    @property
    def total_cost_usd(self) -> float:
        return sum(row.cost_usd for row in self.rows)

    def _ratio(self, flags: list[bool]) -> float:
        return sum(flags) / len(flags) if flags else 0.0

    def _percentile(self, percentile: int) -> float:
        values = sorted(row.latency_ms for row in self.rows)
        if not values:
            return 0.0
        if percentile == 50:
            return statistics.median(values)
        index = min(int(len(values) * percentile / 100), len(values) - 1)
        return values[index]


def load_dataset(path: str | Path) -> list[EvalCase]:
    raw: list[dict[str, Any]] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=entry["id"],
            query=entry["query"],
            expected_tools=list(entry.get("expected_tools", [])),
            must_mention=list(entry.get("must_mention", [])),
            must_not_mention=list(entry.get("must_not_mention", [])),
            must_cite=list(entry.get("must_cite", [])),
            must_refuse=bool(entry.get("must_refuse", False)),
            session_id=entry.get("session_id"),
        )
        for entry in raw
    ]


def tools_called(response: AgentResponse) -> list[str]:
    return [
        event.payload["tool"]
        for event in response.trace
        if event.event_type == EventType.TOOL_CALL
    ]


def score_case(case: EvalCase, response: AgentResponse) -> CaseScore:
    called = tools_called(response)
    answer_lower = (response.answer or "").lower()
    cited = {citation.ref for citation in response.citations}

    mentions_ok = all(
        phrase.lower() in answer_lower for phrase in case.must_mention
    ) and not any(phrase.lower() in answer_lower for phrase in case.must_not_mention)

    refusal_correct = True
    if case.must_refuse:
        refusal_correct = any(phrase in answer_lower for phrase in REFUSAL_PHRASES)

    return CaseScore(
        case_id=case.id,
        tool_selection_correct=set(called) == set(case.expected_tools),
        citations_valid=all(ref in cited for ref in case.must_cite),
        refusal_correct=refusal_correct,
        mentions_ok=mentions_ok,
        latency_ms=response.latency_ms,
        cost_usd=response.cost_usd,
        tools_called=called,
        answer=response.answer,
    )


def run_deterministic(agent: Any, cases: list[EvalCase]) -> DeterministicReport:
    rows = [
        score_case(case, agent.chat(case.query, session_id=case.session_id))
        for case in cases
    ]
    return DeterministicReport(rows=rows)
