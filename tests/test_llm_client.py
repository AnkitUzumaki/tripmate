import pytest

from tripmate.llm.client import LLMClient, LLMError
from tripmate.models import LLMResponse, ToolCall

from tests.fakes import FakeLLMClient


class _Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FnCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _RawToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _FnCall(name, arguments)


class _Choice:
    def __init__(self, message):
        self.message = message


class _Usage:
    prompt_tokens = 120
    completion_tokens = 40


class _Raw:
    def __init__(self, message):
        self.choices = [_Choice(message)]
        self.usage = _Usage()


def test_parses_plain_text_response():
    raw = _Raw(_Message(content="Hello there"))
    parsed = LLMClient._parse(raw)
    assert parsed.content == "Hello there"
    assert parsed.tool_calls == []


def test_parses_token_usage():
    parsed = LLMClient._parse(_Raw(_Message(content="hi")))
    assert parsed.prompt_tokens == 120
    assert parsed.completion_tokens == 40


def test_parses_tool_calls_with_json_arguments():
    raw = _Raw(_Message(tool_calls=[
        _RawToolCall("call_1", "get_weather_forecast",
                     '{"city": "Tokyo", "date_or_month": "December"}')
    ]))
    parsed = LLMClient._parse(raw)
    assert parsed.tool_calls[0].name == "get_weather_forecast"
    assert parsed.tool_calls[0].arguments["city"] == "Tokyo"


def test_malformed_tool_arguments_become_an_empty_dict():
    raw = _Raw(_Message(tool_calls=[_RawToolCall("c1", "t", "{not json")]))
    assert LLMClient._parse(raw).tool_calls[0].arguments == {}


def test_fake_client_replays_scripted_responses_in_order():
    fake = FakeLLMClient([
        LLMResponse(tool_calls=[ToolCall(id="1", name="t", arguments={})]),
        LLMResponse(content="done"),
    ])

    assert fake.complete([{"role": "user", "content": "q"}]).tool_calls[0].name == "t"
    assert fake.complete([{"role": "user", "content": "q"}]).content == "done"


def test_fake_client_records_every_request():
    fake = FakeLLMClient([LLMResponse(content="ok")])
    fake.complete([{"role": "user", "content": "q"}], tools=[{"x": 1}])

    assert fake.calls[0]["tools"] == [{"x": 1}]


def test_fake_client_raises_when_the_script_is_exhausted():
    fake = FakeLLMClient([LLMResponse(content="ok")])
    fake.complete([])

    with pytest.raises(LLMError, match="script exhausted"):
        fake.complete([])
