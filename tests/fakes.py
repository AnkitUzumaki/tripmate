"""Test doubles. FakeLLMClient makes agent tests deterministic and free."""

from __future__ import annotations

from typing import Any

from tripmate.llm.client import LLMError
from tripmate.models import LLMResponse


class FakeLLMClient:
    """Replays a scripted list of LLMResponse objects in order."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        self.calls.append({"messages": list(messages), "tools": tools})
        if self._index >= len(self._script):
            raise LLMError(
                f"script exhausted after {len(self._script)} response(s); "
                f"the agent made an unexpected extra call"
            )
        response = self._script[self._index]
        self._index += 1
        return response
