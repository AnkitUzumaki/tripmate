"""Provider-agnostic LLM client.

LiteLLM normalises tool-calling across OpenAI, Anthropic, Gemini, Groq, Ollama and
others, so the agent loop is identical whichever model the user configures.
"""

from __future__ import annotations

import json
from typing import Any

import litellm

from tripmate.config import Settings, get_settings
from tripmate.models import LLMResponse, ToolCall


class LLMError(Exception):
    """Raised when the provider call fails after retries."""


class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        timeout_s: int,
        base_url: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._base_url = base_url
        self._max_retries = max_retries

    @staticmethod
    def _parse(raw: Any) -> LLMResponse:
        message = raw.choices[0].message
        usage = getattr(raw, "usage", None)

        tool_calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "timeout": self._timeout_s,
            "num_retries": self._max_retries,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self._base_url:
            kwargs["api_base"] = self._base_url

        try:
            raw = litellm.completion(**kwargs)
        except Exception as exc:
            raise LLMError(f"{self._model} call failed: {type(exc).__name__}: {exc}") from exc

        parsed = self._parse(raw)
        try:
            cost = float(litellm.completion_cost(completion_response=raw))
        except Exception:
            cost = 0.0  # unpriced or local model; not an error
        return parsed.model_copy(update={"cost_usd": cost})


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    settings.export_provider_key()
    return LLMClient(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_s=settings.llm_timeout_s,
        base_url=settings.llm_base_url,
    )
