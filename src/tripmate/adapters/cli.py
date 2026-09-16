"""Multi-turn CLI with live reasoning-trace rendering."""

from __future__ import annotations

import uuid

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tripmate.bootstrap import build_agent
from tripmate.config import ConfigError, get_settings
from tripmate.core.trace import EventType
from tripmate.models import AgentResponse, TraceEvent

BANNER = (
    "TripMate — ask about Tokyo, Reykjavik, Bangkok or Barcelona.\n"
    "Commands: /trace  /cost  /reset  /quit"
)

EVENT_STYLES = {
    EventType.QUERY_RECEIVED: "dim",
    EventType.CACHE_HIT: "bold green",
    EventType.LLM_CALL: "cyan",
    EventType.TOOL_CALL: "yellow",
    EventType.TOOL_RESULT: "green",
    EventType.TOOL_ERROR: "bold red",
    EventType.FALLBACK_USED: "bold magenta",
    EventType.CITATION_REJECTED: "bold red",
    EventType.ANSWER_SYNTHESIZED: "dim",
}


def render_trace(console: Console, events: list[TraceEvent]) -> None:
    """Render the reasoning trace as a formatted table."""
    table = Table(title="reasoning trace", show_lines=False, title_style="dim")
    table.add_column("#", justify="right", width=3)
    table.add_column("event", width=20)
    table.add_column("ms", justify="right", width=7)
    table.add_column("detail", overflow="fold")

    for event in events:
        duration = f"{event.duration_ms:.0f}" if event.duration_ms else ""
        detail = ", ".join(
            f"{key}={value}" for key, value in event.payload.items() if value not in (None, [], {})
        )
        table.add_row(
            str(event.seq),
            f"[{EVENT_STYLES.get(event.event_type, 'white')}]{event.event_type}[/]",
            duration,
            detail[:160],
        )
    console.print(table)


def _render_answer(console: Console, response: AgentResponse) -> None:
    """Render the final answer with metadata."""
    console.print(Panel(response.answer, title="TripMate", border_style="blue"))
    refs = ", ".join(c.ref for c in response.citations) or "none"
    cached = " (cached)" if response.was_cached else ""
    console.print(
        f"[dim]sources: {refs} | {response.prompt_tokens}+"
        f"{response.completion_tokens} tokens | ${response.cost_usd:.5f} | "
        f"{response.latency_ms:.0f} ms{cached}[/dim]\n"
    )


def main() -> None:
    """Run the multi-turn CLI."""
    console = Console()
    try:
        settings = get_settings()
        agent = build_agent(settings)
    except ConfigError as exc:
        console.print(f"[bold red]Configuration error:[/] {exc}")
        return

    console.print(Panel(BANNER, border_style="blue"))
    console.print(f"[dim]model: {settings.llm_model}[/dim]\n")

    session_id = uuid.uuid4().hex[:12]
    last: AgentResponse | None = None
    total_cost = 0.0

    while True:
        try:
            query = console.input("[bold]you >[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return

        if query in {"/quit", "/exit"}:
            console.print("bye")
            return
        if query == "/reset":
            session_id = uuid.uuid4().hex[:12]
            console.print("[dim]new session[/dim]\n")
            continue
        if query == "/cost":
            console.print(f"[dim]session total: ${total_cost:.5f}[/dim]\n")
            continue
        if query == "/trace":
            if last is None:
                console.print("[dim]no turn yet[/dim]\n")
            else:
                render_trace(console, last.trace)
            continue

        with console.status("thinking..."):
            last = agent.chat(query, session_id=session_id)

        total_cost += last.cost_usd
        render_trace(console, last.trace)
        _render_answer(console, last)


if __name__ == "__main__":
    main()
