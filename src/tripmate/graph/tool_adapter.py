"""Bridge the tool registry to LangChain tool objects.

`ToolNode` expects LangChain tools; the registry derives its schemas from Python type
hints. Rather than rewrite the tools against LangChain's decorator — which would throw
away the schema derivation and both tool modules' tests — this adapts one to the other.

The registry stays the single source of truth for what a tool is and what it accepts.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from tripmate.tools.registry import ToolRegistry, ToolSpec


def _make_runner(spec: ToolSpec) -> Callable[..., str]:
    """Wrap a registry dispatch so ToolNode receives a string, never an exception.

    Dispatch already converts every failure into a ToolResult, so the invariant that no
    exception reaches the model is preserved unchanged by this layer.
    """

    def run(**kwargs: Any) -> str:
        result = spec.fn(**kwargs)
        return json.dumps(result.model_dump(), default=str)

    run.__name__ = spec.name
    run.__doc__ = spec.description
    return run


def to_langchain_tool(spec: ToolSpec) -> StructuredTool:
    """One registry ToolSpec as a LangChain StructuredTool.

    `args_schema` reuses the pydantic model the @tool decorator already built from the
    function's type hints, so the schema the LLM sees is identical to the one the raw
    loop sent.
    """
    return StructuredTool.from_function(
        func=_make_runner(spec),
        name=spec.name,
        description=spec.description,
        args_schema=spec.arg_model,
    )


def to_langchain_tools(registry: ToolRegistry) -> list[StructuredTool]:
    """Every registered tool, in registration order."""
    return [to_langchain_tool(spec) for spec in registry.specs()]
