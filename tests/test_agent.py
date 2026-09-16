# tests/test_agent.py
import pytest

from tripmate.core.agent import (
    Agent, InvalidQuery, extract_citations, validate_citations, validate_query,
)
from tripmate.core.trace import EventType
from tripmate.models import Citation, LLMResponse, ToolCall, ToolResult
from tripmate.tools.registry import ToolRegistry, tool

from tests.fakes import FakeLLMClient
from typing import Annotated


@tool
def fake_guide(
    query: Annotated[str, "topic"],
    city: Annotated[str | None, "city"] = None,
) -> ToolResult:
    """Search the guide."""
    return ToolResult.ok("fake_guide", {"chunks": [
        {"ref": "tokyo/PACKING TIPS", "city": "tokyo", "section": "PACKING TIPS",
         "text": "Layered clothing works well.", "score": 0.8}
    ]})


@tool
def fake_weather(
    city: Annotated[str, "city"],
    date_or_month: Annotated[str, "month"],
) -> ToolResult:
    """Get weather."""
    return ToolResult.ok("fake_weather", {
        "city": "Tokyo", "period": "December", "source": "climate_normal",
        "conditions": "cold, mostly dry", "temp_range_c": [3.0, 12.0],
        "precip_days": 4,
    })


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(fake_guide)
    registry.register(fake_weather)
    return registry


def _agent(script: list[LLMResponse]) -> tuple[Agent, FakeLLMClient]:
    fake = FakeLLMClient(script)
    return Agent(llm=fake, registry=_registry()), fake


# --- input validation ---

def test_validate_query_strips_surrounding_whitespace():
    assert validate_query("  hello  ", max_chars=100) == "hello"


def test_validate_query_rejects_empty_input():
    with pytest.raises(InvalidQuery, match="empty"):
        validate_query("   ", max_chars=100)


def test_validate_query_rejects_overlong_input():
    with pytest.raises(InvalidQuery, match="too long"):
        validate_query("x" * 101, max_chars=100)


# --- citations ---

def test_extract_citations_finds_every_reference():
    refs = extract_citations("See [tokyo/PACKING TIPS] and [tokyo/SAFETY & HEALTH].")
    assert [c.ref for c in refs] == ["tokyo/PACKING TIPS", "tokyo/SAFETY & HEALTH"]


def test_validate_citations_keeps_references_that_were_retrieved():
    answer = "Pack layers [tokyo/PACKING TIPS]."
    cleaned, kept, rejected = validate_citations(
        answer, extract_citations(answer), {"tokyo/PACKING TIPS"}
    )
    assert kept[0].ref == "tokyo/PACKING TIPS"
    assert rejected == []
    assert cleaned == answer


def test_validate_citations_strips_references_never_retrieved():
    answer = "Pack layers [osaka/PACKING TIPS]."
    cleaned, kept, rejected = validate_citations(
        answer, extract_citations(answer), {"tokyo/PACKING TIPS"}
    )
    assert kept == []
    assert rejected == ["osaka/PACKING TIPS"]
    assert "[osaka/PACKING TIPS]" not in cleaned


# --- the loop ---

def test_no_tool_query_returns_the_model_answer_directly():
    agent, _ = _agent([LLMResponse(content="I help with travel questions.")])
    response = agent.chat("what can you do?")

    assert response.answer == "I help with travel questions."


def test_no_tool_query_makes_exactly_one_llm_call():
    agent, fake = _agent([LLMResponse(content="hello")])
    agent.chat("hi")

    assert len(fake.calls) == 1


def test_single_tool_query_calls_that_tool_then_answers():
    agent, _ = _agent([
        LLMResponse(tool_calls=[ToolCall(id="1", name="fake_guide",
                                         arguments={"query": "visa", "city": "tokyo"})]),
        LLMResponse(content="No visa needed [tokyo/PACKING TIPS]."),
    ])
    response = agent.chat("do I need a visa for Tokyo?")

    tools_called = [e.payload["tool"] for e in response.trace
                    if e.event_type == EventType.TOOL_CALL]
    assert tools_called == ["fake_guide"]


def test_multi_tool_query_calls_both_tools_in_one_turn():
    agent, _ = _agent([
        LLMResponse(tool_calls=[
            ToolCall(id="1", name="fake_guide",
                     arguments={"query": "packing", "city": "tokyo"}),
            ToolCall(id="2", name="fake_weather",
                     arguments={"city": "Tokyo", "date_or_month": "December"}),
        ]),
        LLMResponse(content="Pack warm layers [tokyo/PACKING TIPS]."),
    ])
    response = agent.chat("what should I pack for Tokyo in December?")

    tools_called = {e.payload["tool"] for e in response.trace
                    if e.event_type == EventType.TOOL_CALL}
    assert tools_called == {"fake_guide", "fake_weather"}


def test_tool_results_are_appended_as_tool_role_messages():
    agent, fake = _agent([
        LLMResponse(tool_calls=[ToolCall(id="1", name="fake_weather",
                                         arguments={"city": "Tokyo",
                                                    "date_or_month": "December"})]),
        LLMResponse(content="Cold."),
    ])
    agent.chat("weather in Tokyo in December?")

    roles = [m["role"] for m in fake.calls[1]["messages"]]
    assert "tool" in roles


def test_valid_citations_are_returned_on_the_response():
    agent, _ = _agent([
        LLMResponse(tool_calls=[ToolCall(id="1", name="fake_guide",
                                         arguments={"query": "packing"})]),
        LLMResponse(content="Layers [tokyo/PACKING TIPS]."),
    ])
    response = agent.chat("packing for Tokyo?")

    assert [c.ref for c in response.citations] == ["tokyo/PACKING TIPS"]


def test_hallucinated_citations_are_stripped_and_traced():
    agent, _ = _agent([
        LLMResponse(tool_calls=[ToolCall(id="1", name="fake_guide",
                                         arguments={"query": "packing"})]),
        LLMResponse(content="Layers [paris/PACKING TIPS]."),
    ])
    response = agent.chat("packing for Paris?")

    assert response.citations == []
    assert "[paris/PACKING TIPS]" not in response.answer
    assert any(e.event_type == EventType.CITATION_REJECTED for e in response.trace)


def test_iteration_ceiling_forces_a_final_answer():
    looping = [
        LLMResponse(tool_calls=[ToolCall(id=str(i), name="fake_weather",
                                         arguments={"city": "Tokyo",
                                                    "date_or_month": "December"})])
        for i in range(10)
    ]
    fake = FakeLLMClient(looping)
    agent = Agent(llm=fake, registry=_registry())

    response = agent.chat("weather?")

    assert response.answer  # non-empty, synthesised from what was gathered
    assert len(fake.calls) <= 6  # max_tool_iterations (5) plus the forced synthesis


def test_empty_input_is_rejected_before_any_llm_call():
    fake = FakeLLMClient([])
    agent = Agent(llm=fake, registry=_registry())

    response = agent.chat("   ")

    assert "empty" in response.answer.lower()
    assert fake.calls == []


def test_trace_records_tokens_and_cost_from_llm_calls():
    agent, _ = _agent([
        LLMResponse(content="hi", prompt_tokens=100, completion_tokens=20,
                    cost_usd=0.001)
    ])
    response = agent.chat("hello")

    assert response.prompt_tokens == 100
    assert response.cost_usd == pytest.approx(0.001)
