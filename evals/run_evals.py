"""Eval entry point. Runs the deterministic layer by default; --ragas and
--simulation add the LLM-judged layers."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from evals.deterministic import DeterministicReport, load_dataset, run_deterministic
from tripmate.bootstrap import build_agent
from tripmate.config import get_settings

DATASET_PATH = Path(__file__).parent / "dataset.yaml"


def _print_report(console: Console, report: DeterministicReport) -> None:
    table = Table(title="deterministic evaluation")
    table.add_column("case", overflow="fold")
    table.add_column("tools", justify="center", width=6)
    table.add_column("cite", justify="center", width=5)
    table.add_column("refuse", justify="center", width=7)
    table.add_column("text", justify="center", width=5)
    table.add_column("ms", justify="right", width=7)
    table.add_column("called", overflow="fold")

    def mark(flag: bool) -> str:
        return "[green]PASS[/]" if flag else "[red]FAIL[/]"

    for row in report.rows:
        table.add_row(
            row.case_id, mark(row.tool_selection_correct), mark(row.citations_valid),
            mark(row.refusal_correct), mark(row.mentions_ok),
            f"{row.latency_ms:.0f}", ", ".join(row.tools_called) or "-",
        )
    console.print(table)

    summary = Table(title="summary", show_header=False)
    summary.add_row("tool-selection accuracy", f"{report.accuracy:.1%}")
    summary.add_row("citation validity", f"{report.citation_rate:.1%}")
    summary.add_row("refusal accuracy", f"{report.refusal_accuracy:.1%}")
    summary.add_row("overall pass rate", f"{report.pass_rate:.1%}")
    summary.add_row("p50 latency", f"{report.p50_latency_ms:.0f} ms")
    summary.add_row("p95 latency", f"{report.p95_latency_ms:.0f} ms")
    summary.add_row("total cost", f"${report.total_cost_usd:.4f}")
    console.print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TripMate eval suite.")
    parser.add_argument("--ragas", action="store_true", help="add RAGAS RAG metrics")
    parser.add_argument("--simulation", action="store_true",
                        help="add simulated-user conversations")
    parser.add_argument("--limit", type=int, default=0,
                        help="run only the first N cases")
    args = parser.parse_args()

    console = Console()
    # Cache off: replaying a cached answer skips tool calls entirely, which makes
    # tool-selection unmeasurable and scores cache hits as routing failures.
    settings = get_settings().model_copy(update={"semantic_cache_enabled": False})
    agent = build_agent(settings)
    cases = load_dataset(DATASET_PATH)
    if args.limit:
        cases = cases[: args.limit]

    console.print(f"[dim]running {len(cases)} deterministic case(s)...[/dim]")
    _print_report(console, run_deterministic(agent, cases))

    if args.ragas:
        from evals.ragas_eval import run_ragas, print_ragas_report
        print_ragas_report(console, run_ragas(agent, cases))

    if args.simulation:
        from evals.simulation import run_simulations, print_simulation_report
        print_simulation_report(console, run_simulations(agent))


if __name__ == "__main__":
    main()
