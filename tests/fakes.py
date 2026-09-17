"""Test doubles.

`FakeChatModel` replays scripted AIMessages through LangChain's own fake, so the graph
runs its real nodes, real edges and real ToolNode with only the model substituted. That
is what keeps the suite deterministic: no API key, no network, no cost, no flake.
"""

from __future__ import annotations

from typing import Any, Iterator

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage


class ScriptExhausted(RuntimeError):
    """Raised when the graph asks for more model responses than the script holds.

    Load-bearing in the cache tests: exhausting the script is how they prove the cache
    prevented an LLM call, rather than merely asserting a flag.
    """


class FakeChatModel(GenericFakeChatModel):
    """Replays a list of AIMessages in order and records every request.

    Subclasses LangChain's fake so `bind_tools` and the runnable protocol work exactly
    as the real model's do.
    """

    calls: list[list[BaseMessage]] = []

    def __init__(self, script: list[AIMessage], **kwargs: Any) -> None:
        super().__init__(messages=iter(script), **kwargs)
        object.__setattr__(self, "_script_len", len(script))
        object.__setattr__(self, "_served", 0)
        object.__setattr__(self, "calls", [])

    def bind_tools(self, tools, **kwargs: Any):
        """No-op: responses are scripted, so tool schemas change nothing here.

        The real model uses this to learn what it may call; the fake already knows what
        it will answer. Returning self keeps the graph's wiring identical either way.
        """
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        served = getattr(self, "_served", 0)
        if served >= getattr(self, "_script_len", 0):
            raise ScriptExhausted(
                f"script exhausted after {self._script_len} response(s); "
                f"the graph made an unexpected extra model call"
            )
        object.__setattr__(self, "_served", served + 1)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def ai(
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> AIMessage:
    """Build a scripted assistant message.

    `tool_calls` entries are LangChain's shape: {"name", "args", "id"}. Token counts
    populate `usage_metadata`, which is where the agent node reads them from, so tests
    asserting on cost and token accounting stay meaningful.
    """
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata={
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )
