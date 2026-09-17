"""One test per error scenario named in the assessment brief.

Brief: unknown destination, missing weather data, ambiguous query, tool failure or
timeout, out-of-scope request, malformed or empty input.
"""

import httpx
import pytest

from tests.fakes import ai
import respx

from tripmate.core.trace import EventType
from tripmate.tools.weather import GEOCODE_URL


# --- 1. Unknown / unsupported destination ---

def test_unknown_destination_returns_no_data_and_names_supported_cities(build_test_agent):
    agent, _ = build_test_agent([
        ai(tool_calls=[{"name": "search_destination_guide", "args": {"query": "visa requirements", "city": "paris"}, "id": "1"}]),
        ai("I don't have a guide for Paris. I cover Tokyo, "
                            "Reykjavik, Bangkok and Barcelona."),
    ])
    response = agent.chat("what are the visa rules for Paris?")

    statuses = [e.payload["status"] for e in response.trace
                if e.event_type == EventType.TOOL_RESULT]
    assert statuses == ["no_data"]
    assert "Paris" in response.answer


def test_unknown_destination_answer_carries_no_citations(build_test_agent):
    agent, _ = build_test_agent([
        ai(tool_calls=[{"name": "search_destination_guide", "args": {"query": "visa", "city": "paris"}, "id": "1"}]),
        ai("No guide for Paris."),
    ])
    assert agent.chat("visa rules for Paris?").citations == []


# --- 2. Missing / incomplete weather data ---

@respx.mock
def test_unresolvable_city_returns_no_data_from_the_weather_tool(build_test_agent):
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json={"results": []}))

    agent, _ = build_test_agent([
        ai(tool_calls=[{"name": "get_weather_forecast", "args": {"city": "Atlantis", "date_or_month": "July"}, "id": "1"}]),
        ai("I couldn't find weather data for Atlantis."),
    ])
    response = agent.chat("weather in Atlantis in July?")

    statuses = [e.payload["status"] for e in response.trace
                if e.event_type == EventType.TOOL_RESULT]
    assert statuses == ["no_data"]


def test_unparseable_timeframe_returns_no_data(build_test_agent):
    agent, _ = build_test_agent([
        ai(tool_calls=[{"name": "get_weather_forecast", "args": {"city": "Tokyo", "date_or_month": "whenever"}, "id": "1"}]),
        ai("Could you tell me which month you mean?"),
    ])
    response = agent.chat("weather in Tokyo whenever?")

    statuses = [e.payload["status"] for e in response.trace
                if e.event_type == EventType.TOOL_RESULT]
    assert statuses == ["no_data"]


# --- 3. Ambiguous query ---

def test_ambiguous_query_asks_for_clarification_without_calling_tools(build_test_agent):
    agent, _ = build_test_agent([
        ai("Which destination did you have in mind — Tokyo, "
                            "Reykjavik, Bangkok or Barcelona?")
    ])
    response = agent.chat("what should I pack?")

    assert [e for e in response.trace if e.event_type == EventType.TOOL_CALL] == []
    assert "?" in response.answer


# --- 4. Tool failure or timeout ---

@respx.mock
def test_weather_api_timeout_falls_back_and_records_the_fallback(build_test_agent):
    respx.get(GEOCODE_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    agent, _ = build_test_agent([
        ai(tool_calls=[{"name": "get_weather_forecast", "args": {"city": "Tokyo", "date_or_month": "December"}, "id": "1"}]),
        ai("The live weather service was unavailable; typical "
                            "December conditions in Tokyo are cold and dry."),
    ])
    response = agent.chat("weather in Tokyo in December?")

    assert any(e.event_type == EventType.FALLBACK_USED for e in response.trace)


@respx.mock
def test_weather_failure_without_offline_data_surfaces_a_tool_error(build_test_agent):
    respx.get(GEOCODE_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    agent, _ = build_test_agent([
        ai(tool_calls=[{"name": "get_weather_forecast", "args": {"city": "Lisbon", "date_or_month": "December"}, "id": "1"}]),
        ai("I couldn't reach the weather service for Lisbon."),
    ])
    response = agent.chat("weather in Lisbon in December?")

    assert any(e.event_type == EventType.TOOL_ERROR for e in response.trace)


def test_unknown_tool_name_from_the_model_becomes_a_tool_error(build_test_agent):
    agent, _ = build_test_agent([
        ai(tool_calls=[{"name": "book_flight", "args": {"to": "Tokyo"}, "id": "1"}]),
        ai("I can't book flights."),
    ])
    response = agent.chat("book me a flight")

    assert any(e.event_type == EventType.TOOL_ERROR for e in response.trace)


# --- 5. Out-of-scope request ---

def test_booking_request_is_refused_without_calling_tools(build_test_agent):
    agent, _ = build_test_agent([
        ai("I can't book flights. I can tell you about visas, "
                            "weather, packing, customs and safety for four cities.")
    ])
    response = agent.chat("can you book my flight to Barcelona?")

    assert [e for e in response.trace if e.event_type == EventType.TOOL_CALL] == []
    assert "can't" in response.answer.lower()


def test_unrelated_topic_is_declined_without_calling_tools(build_test_agent):
    agent, _ = build_test_agent([
        ai("That's outside what I cover. I help with travel "
                            "questions for four destinations.")
    ])
    response = agent.chat("write me a python script to sort a list")

    assert [e for e in response.trace if e.event_type == EventType.TOOL_CALL] == []


# --- 6. Malformed or empty input ---

def test_empty_input_is_rejected_before_any_llm_call(build_test_agent):
    registry_agent, fake = build_test_agent([])
    response = registry_agent.chat("")

    assert fake.calls == []
    assert "empty" in response.answer.lower()


def test_whitespace_only_input_is_rejected_before_any_llm_call(build_test_agent):
    agent, fake = build_test_agent([])
    agent.chat("\n\t   ")

    assert fake.calls == []


def test_overlong_input_is_rejected_before_any_llm_call(build_test_agent):
    agent, fake = build_test_agent([])
    response = agent.chat("x" * 5000)

    assert fake.calls == []
    assert "too long" in response.answer.lower()
