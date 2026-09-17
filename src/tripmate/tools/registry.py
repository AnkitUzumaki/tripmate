"""Tool registry. Schemas are derived from type hints, never hand-written."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Annotated, Any, Callable, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, ValidationError, create_model

from tripmate.models import ToolResult

TOOL_SPEC_ATTR = "_tool_spec"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., ToolResult]
    arg_model: type[BaseModel]
    schema: dict[str, Any]


def _describe(hint: Any) -> tuple[Any, str]:
    """Split `Annotated[T, "description"]` into (T, description)."""
    if get_origin(hint) is not Annotated:
        return hint, ""
    actual, *metadata = get_args(hint)
    description = next((m for m in metadata if isinstance(m, str)), "")
    return actual, description


def _build_arg_model(fn: Callable[..., Any]) -> type[BaseModel]:
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        annotation, description = _describe(hints.get(name, str))
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, Field(default, description=description))
    return create_model(f"{fn.__name__}_Args", **fields)


def _build_schema(fn: Callable[..., Any], arg_model: type[BaseModel]) -> dict[str, Any]:
    parameters = arg_model.model_json_schema()
    parameters.pop("title", None)
    for prop in parameters.get("properties", {}).values():
        prop.pop("title", None)
    description = (fn.__doc__ or "").strip().split("\n")[0]
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": parameters,
        },
    }


def tool(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    """Mark a function as an agent tool and attach its derived schema."""
    arg_model = _build_arg_model(fn)
    setattr(
        fn,
        TOOL_SPEC_ATTR,
        ToolSpec(
            name=fn.__name__,
            description=(fn.__doc__ or "").strip().split("\n")[0],
            fn=fn,
            arg_model=arg_model,
            schema=_build_schema(fn, arg_model),
        ),
    )
    return fn


class ToolRegistry:
    """Holds tool specs and dispatches calls with validated arguments."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, fn: Callable[..., ToolResult]) -> None:
        spec = getattr(fn, TOOL_SPEC_ATTR, None)
        if spec is None:
            raise ValueError(f"{fn.__name__} is not decorated with @tool")
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._specs)

    def specs(self) -> list[ToolSpec]:
        """Every registered spec, in registration order.

        Public because adapters to other frameworks need the arg_model and callable,
        not just the JSON schema `schemas()` returns.
        """
        return list(self._specs.values())

    def schemas(self) -> list[dict[str, Any]]:
        """Every tool schema, in registration order.

        Scale hook: past ~10 tools this is where semantic tool routing plugs in —
        embed the descriptions and return only the top-K relevant to the query.
        """
        return [spec.schema for spec in self._specs.values()]

    def dispatch(self, name: str, raw_args: dict[str, Any]) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult.error(name, f"unknown tool {name!r}")

        if not isinstance(raw_args, dict):
            return ToolResult.error(
                name, f"invalid arguments: expected an object, got {type(raw_args).__name__}"
            )

        try:
            args = spec.arg_model(**raw_args)
        except ValidationError as exc:
            return ToolResult.error(name, f"invalid arguments: {exc.errors()}")

        try:
            return spec.fn(**args.model_dump())
        except Exception as exc:  # tools must never raise into the loop
            return ToolResult.error(name, f"{type(exc).__name__}: {exc}")
