"""Layer 2: RAGAS metrics over the same dataset.

Retrieved guide chunks are pulled out of the trace, so RAGAS scores the contexts
the agent actually used rather than a re-run of retrieval.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from evals.deterministic import EvalCase
from tripmate.core.trace import EventType
from tripmate.models import AgentResponse

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")


def _contexts_from(response: AgentResponse) -> list[str]:
    """Guide text the agent retrieved this turn, taken from the trace."""
    contexts: list[str] = []
    for event in response.trace:
        if event.event_type != EventType.TOOL_RESULT:
            continue
        if event.payload.get("tool") != "search_destination_guide":
            continue
        contexts.extend(event.payload.get("texts", []))
    return contexts


def collect_samples(agent: Any, cases: list[EvalCase]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for case in cases:
        if "search_destination_guide" not in case.expected_tools:
            continue
        response = agent.chat(case.query, session_id=case.session_id)
        contexts = _contexts_from(response)
        if not contexts:
            continue
        samples.append({
            "question": case.query,
            "answer": response.answer,
            "contexts": contexts,
        })
    return samples


def run_ragas(agent: Any, cases: list[EvalCase]) -> dict[str, float]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    samples = collect_samples(agent, cases)
    if not samples:
        return {}

    result = evaluate(
        Dataset.from_list(samples),
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return {name: float(result[name]) for name in METRIC_NAMES if name in result}


def print_ragas_report(console: Console, scores: dict[str, float]) -> None:
    if not scores:
        console.print("[yellow]RAGAS: no RAG samples collected[/yellow]")
        return
    table = Table(title="RAGAS metrics", show_header=False)
    for name, value in scores.items():
        table.add_row(name.replace("_", " "), f"{value:.3f}")
    console.print(table)
