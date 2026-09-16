"""Test suite for tool selection: the agent picks the right tool(s) per query.

This suite validates the core requirement: the agent dynamically selects single-tool,
multi-tool, or no-tool paths based on the query. Tests use real tools with a scripted
model, ensuring tool wiring is genuinely exercised while results remain deterministic.
"""

from __future__ import annotations

from tripmate.models import LLMResponse, ToolCall


# --- no tool ---

def test_greeting_uses_no_tools(build_test_agent, tools_called):
    agent, _ = build_test_agent([LLMResponse(content="Hello! Ask me about Tokyo.")])
    assert tools_called(agent.chat("hello there")) == []


def test_capability_question_uses_no_tools(build_test_agent, tools_called):
    agent, _ = build_test_agent([
        LLMResponse(content="I cover Tokyo, Reykjavik, Bangkok and Barcelona.")
    ])
    assert tools_called(agent.chat("what can you help with?")) == []


# --- single tool: RAG ---

def test_visa_question_uses_only_the_guide(build_test_agent, tools_called):
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="search_destination_guide",
            arguments={"query": "visa and entry requirements", "city": "tokyo"})]),
        LLMResponse(content="Many nationalities enter visa-free "
                            "[tokyo/VISA & ENTRY]."),
    ])
    assert tools_called(agent.chat("do I need a visa for Japan?")) == [
        "search_destination_guide"
    ]


def test_guide_tool_result_reaches_the_second_llm_call(build_test_agent):
    agent, fake = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="search_destination_guide",
            arguments={"query": "local customs", "city": "tokyo"})]),
        LLMResponse(content="Bowing is common [tokyo/LOCAL CUSTOMS]."),
    ])
    agent.chat("what are the customs in Tokyo?")

    tool_messages = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert "Bowing" in tool_messages[0]["content"]


# --- single tool: weather ---

def test_weather_question_uses_only_the_weather_tool(build_test_agent, tools_called):
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="get_weather_forecast",
            arguments={"city": "Reykjavik", "date_or_month": "January"})]),
        LLMResponse(content="Cold, around -3 to 3 C."),
    ])
    assert tools_called(agent.chat("how cold is Reykjavik in January?")) == [
        "get_weather_forecast"
    ]


# --- multi tool ---

def test_packing_question_uses_both_tools(build_test_agent, tools_called):
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[
            ToolCall(id="1", name="search_destination_guide",
                     arguments={"query": "packing tips", "city": "tokyo"}),
            ToolCall(id="2", name="get_weather_forecast",
                     arguments={"city": "Tokyo", "date_or_month": "December"}),
        ]),
        LLMResponse(content="Pack warm layers [tokyo/PACKING TIPS]."),
    ])
    assert set(tools_called(agent.chat("what should I pack for Tokyo in December?"))) == {
        "search_destination_guide", "get_weather_forecast"
    }


def test_both_tool_results_are_present_before_synthesis(build_test_agent):
    agent, fake = build_test_agent([
        LLMResponse(tool_calls=[
            ToolCall(id="1", name="search_destination_guide",
                     arguments={"query": "packing tips", "city": "bangkok"}),
            ToolCall(id="2", name="get_weather_forecast",
                     arguments={"city": "Bangkok", "date_or_month": "July"}),
        ]),
        LLMResponse(content="Light, breathable clothing [bangkok/PACKING TIPS]."),
    ])
    agent.chat("what should I pack for Bangkok in July?")

    tool_messages = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2


def test_sequential_tool_calls_across_two_iterations_are_supported(
    build_test_agent, tools_called
):
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="search_destination_guide",
            arguments={"query": "best time to visit", "city": "barcelona"})]),
        LLMResponse(tool_calls=[ToolCall(
            id="2", name="get_weather_forecast",
            arguments={"city": "Barcelona", "date_or_month": "May"})]),
        LLMResponse(content="May is mild [barcelona/BEST TIME TO VISIT]."),
    ])
    response = agent.chat("when should I visit Barcelona and how warm is it then?")

    assert tools_called(response) == [
        "search_destination_guide", "get_weather_forecast"
    ]
