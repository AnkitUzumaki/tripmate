"""Layer 3: simulated multi-turn traveler conversations.

Scripted turns stand in for a real user across a whole conversation, scoring goal
completion and whether context carried between turns. Single-turn evals cannot
catch a follow-up like "and what about Bangkok?" losing the thread.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class Conversation:
    id: str
    turns: list[str]
    goal_phrases: list[str] = field(default_factory=list)
    context_phrases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SimulationResult:
    conversation_id: str
    goal_completion: float
    context_retained: bool
    answers: list[str]

    @property
    def passed(self) -> bool:
        return self.goal_completion >= 1.0 and self.context_retained


CONVERSATIONS: list[Conversation] = [
    Conversation(
        id="tokyo_winter_trip",
        turns=[
            "I'm planning a trip to Tokyo in December.",
            "What should I pack?",
            "And is it safe there?",
        ],
        goal_phrases=["layer", "safe"],
        context_phrases=["tokyo"],
    ),
    Conversation(
        id="switch_destination",
        turns=[
            "What's the weather like in Bangkok in July?",
            "And what about Barcelona instead?",
        ],
        goal_phrases=["barcelona"],
        context_phrases=["barcelona"],
    ),
    Conversation(
        id="scope_then_recover",
        turns=[
            "Can you book me a flight to Reykjavik?",
            "Okay, then just tell me what to pack for January.",
        ],
        goal_phrases=["pack"],
        context_phrases=["reykjavik"],
    ),
]


def score_conversation(
    conversation: Conversation, answers: list[str]
) -> SimulationResult:
    combined = " ".join(answers).lower()

    if conversation.goal_phrases:
        met = sum(1 for phrase in conversation.goal_phrases
                  if phrase.lower() in combined)
        goal_completion = met / len(conversation.goal_phrases)
    else:
        goal_completion = 1.0

    if conversation.context_phrases and len(answers) > 1:
        later = " ".join(answers[1:]).lower()
        context_retained = all(
            phrase.lower() in later for phrase in conversation.context_phrases
        )
    else:
        context_retained = True

    return SimulationResult(
        conversation_id=conversation.id,
        goal_completion=goal_completion,
        context_retained=context_retained,
        answers=answers,
    )


def run_simulations(
    agent: Any, conversations: list[Conversation] | None = None
) -> list[SimulationResult]:
    results: list[SimulationResult] = []
    for conversation in conversations or CONVERSATIONS:
        session_id = uuid.uuid4().hex[:12]
        answers = [
            agent.chat(turn, session_id=session_id).answer
            for turn in conversation.turns
        ]
        results.append(score_conversation(conversation, answers))
    return results


def print_simulation_report(
    console: Console, results: list[SimulationResult]
) -> None:
    table = Table(title="simulated conversations")
    table.add_column("conversation")
    table.add_column("goal", justify="right", width=7)
    table.add_column("context", justify="center", width=9)
    table.add_column("result", justify="center", width=7)

    for result in results:
        table.add_row(
            result.conversation_id,
            f"{result.goal_completion:.0%}",
            "[green]kept[/]" if result.context_retained else "[red]lost[/]",
            "[green]PASS[/]" if result.passed else "[red]FAIL[/]",
        )
    console.print(table)
