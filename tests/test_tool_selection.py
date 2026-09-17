"""Test suite for tool selection: the agent picks the right tool(s) per query.

This suite validates the core requirement: the agent dynamically selects single-tool,
multi-tool, or no-tool paths based on the query. Tests use real tools with a scripted
model, ensuring tool wiring is genuinely exercised while results remain deterministic.
"""

from __future__ import annotations

from tests.fakes import FakeChatModel, ai
from langchain_core.messages import ToolMessage



# --- no tool ---

def test_greeting_uses_no_tools(build_test_agent, tools_called):
    agent, _ = build_test_agent([ai("Hello! Ask me about Tokyo.")])
    assert tools_called(agent.chat("hello there")) == []


def test_capability_question_uses_no_tools(build_test_agent, tools_called):
    agent, _ = build_test_agent([
        ai("I cover Tokyo, Reykjavik, Bangkok and Barcelona.")
    ])
    assert tools_called(agent.chat("what can you help with?")) == []


# --- single tool: RAG ---

def test_visa_question_uses_only_the_guide(build_test_agent, tools_called):
    agent, _ = build_test_agent([
        ai(tool_calls=[{"name": "search_destination_guide", "args": {"query": "visa and entry requirements", "city": "tokyo"}, "id": "1"}]),
        ai("Many nationalities enter visa-free "
                            "[tokyo/VISA & ENTRY]."),
    ])
    assert tools_called(agent.chat("do I need a visa for Japan?")) == [
        "search_destination_guide"
    ]


def test_guide_tool_result_reaches_the_second_llm_call(build_test_agent):
    agent, fake = build_test_agent([
        ai(tool_calls=[{"name": "search_destination_guide", "args": {"query": "local customs", "city": "tokyo"}, "id": "1"}]),
        ai("Bowing is common [tokyo/LOCAL CUSTOMS]."),
    ])
    agent.chat("what are the customs in Tokyo?")

    tool_messages = [m for m in fake.calls[1] if isinstance(m, ToolMessage)]
    assert "Bowing" in tool_messages[0].content


# --- single tool: weather ---

def test_weather_question_uses_only_the_weather_tool(build_test_agent, tools_called):
    agent, _ = build_test_agent([
        ai(tool_calls=[{"name": "get_weather_forecast", "args": {"city": "Reykjavik", "date_or_month": "January"}, "id": "1"}]),
        ai("Cold, around -3 to 3 C."),
    ])
    assert tools_called(agent.chat("how cold is Reykjavik in January?")) == [
        "get_weather_forecast"
    ]


# --- multi tool ---

def test_packing_question_uses_both_tools(build_test_agent, tools_called):
    agent, _ = build_test_agent([
        ai(tool_calls=[
            {"name": "search_destination_guide", "args": {"query": "packing tips", "city": "tokyo"}, "id": "1"},
            {"name": "get_weather_forecast", "args": {"city": "Tokyo", "date_or_month": "December"}, "id": "2"},
        ]),
        ai("Pack warm layers [tokyo/PACKING TIPS]."),
    ])
    assert set(tools_called(agent.chat("what should I pack for Tokyo in December?"))) == {
        "search_destination_guide", "get_weather_forecast"
    }


def test_both_tool_results_are_present_before_synthesis(build_test_agent):
    agent, fake = build_test_agent([
        ai(tool_calls=[
            {"name": "search_destination_guide", "args": {"query": "packing tips", "city": "bangkok"}, "id": "1"},
            {"name": "get_weather_forecast", "args": {"city": "Bangkok", "date_or_month": "July"}, "id": "2"},
        ]),
        ai("Light, breathable clothing [bangkok/PACKING TIPS]."),
    ])
    agent.chat("what should I pack for Bangkok in July?")

    tool_messages = [m for m in fake.calls[1] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2


def test_sequential_tool_calls_across_two_iterations_are_supported(
    build_test_agent, tools_called
):
    agent, _ = build_test_agent([
        ai(tool_calls=[{"name": "search_destination_guide", "args": {"query": "best time to visit", "city": "barcelona"}, "id": "1"}]),
        ai(tool_calls=[{"name": "get_weather_forecast", "args": {"city": "Barcelona", "date_or_month": "May"}, "id": "2"}]),
        ai("May is mild [barcelona/BEST TIME TO VISIT]."),
    ])
    response = agent.chat("when should I visit Barcelona and how warm is it then?")

    assert tools_called(response) == [
        "search_destination_guide", "get_weather_forecast"
    ]
