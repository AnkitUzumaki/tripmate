from typing import Annotated

import pytest

from tripmate.models import ToolResult
from tripmate.tools.registry import ToolRegistry, tool


@tool
def sample_add(
    first: Annotated[int, "The first number"],
    second: Annotated[int, "The second number"] = 10,
) -> ToolResult:
    """Add two numbers together."""
    return ToolResult.ok("sample_add", {"sum": first + second})


@tool
def sample_boom(city: Annotated[str, "A city"]) -> ToolResult:
    """Always explodes."""
    raise RuntimeError("kaboom")


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(sample_add)
    registry.register(sample_boom)
    return registry


def test_schema_uses_docstring_summary_as_description():
    schema = _registry().schemas()[0]
    assert schema["function"]["description"] == "Add two numbers together."


def test_schema_derives_parameter_types_from_type_hints():
    params = _registry().schemas()[0]["function"]["parameters"]
    assert params["properties"]["first"]["type"] == "integer"


def test_schema_carries_annotated_description():
    params = _registry().schemas()[0]["function"]["parameters"]
    assert params["properties"]["first"]["description"] == "The first number"


def test_schema_marks_only_defaultless_parameters_required():
    params = _registry().schemas()[0]["function"]["parameters"]
    assert params["required"] == ["first"]


def test_dispatch_executes_the_tool_and_returns_its_result():
    result = _registry().dispatch("sample_add", {"first": 1, "second": 2})
    assert result.data == {"sum": 3}


def test_dispatch_applies_declared_defaults():
    result = _registry().dispatch("sample_add", {"first": 5})
    assert result.data == {"sum": 15}


def test_dispatch_coerces_valid_string_input_to_declared_type():
    result = _registry().dispatch("sample_add", {"first": "7", "second": "3"})
    assert result.data == {"sum": 10}


def test_dispatch_returns_error_result_for_invalid_arguments():
    result = _registry().dispatch("sample_add", {"first": "not a number"})
    assert result.status == "error"
    assert "invalid arguments" in result.reason


def test_dispatch_returns_error_result_for_unknown_tool():
    result = _registry().dispatch("no_such_tool", {})
    assert result.status == "error"
    assert "unknown tool" in result.reason


def test_dispatch_converts_tool_exceptions_into_error_results():
    result = _registry().dispatch("sample_boom", {"city": "tokyo"})
    assert result.status == "error"
    assert "kaboom" in result.reason


def test_names_lists_every_registered_tool():
    assert _registry().names() == ["sample_add", "sample_boom"]


def test_dispatch_returns_error_result_for_non_mapping_arguments():
    registry = _registry()

    for bad_args in ([1, 2], "a string", None, 42):
        result = registry.dispatch("sample_add", bad_args)
        assert result.status == "error"
        assert "expected an object" in result.reason


def test_register_rejects_a_function_without_the_tool_decorator():
    def undecorated(x: int) -> ToolResult:
        """Not a tool."""
        return ToolResult.ok("undecorated", {})

    with pytest.raises(ValueError, match="not decorated"):
        ToolRegistry().register(undecorated)


def test_schema_handles_a_bare_type_hint_without_annotated():
    @tool
    def bare_hint_tool(city: str) -> ToolResult:
        """Takes a bare hint."""
        return ToolResult.ok("bare_hint_tool", {})

    registry = ToolRegistry()
    registry.register(bare_hint_tool)
    params = registry.schemas()[0]["function"]["parameters"]

    assert params["properties"]["city"]["type"] == "string"
    assert params["properties"]["city"].get("description", "") == ""
