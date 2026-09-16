import json

from tripmate.core.trace import EventType, Tracer, load_trace


def test_events_are_numbered_from_one():
    tracer = Tracer(session_id="s1")
    tracer.record(EventType.QUERY_RECEIVED, query="hi")
    tracer.record(EventType.LLM_CALL)

    assert [event.seq for event in tracer.events] == [1, 2]


def test_payload_keywords_are_stored_on_the_event():
    tracer = Tracer(session_id="s1")
    event = tracer.record(EventType.TOOL_CALL, tool="get_weather_forecast")

    assert event.payload["tool"] == "get_weather_forecast"


def test_totals_sum_tokens_and_cost_across_llm_calls():
    tracer = Tracer(session_id="s1")
    tracer.record(EventType.LLM_CALL, prompt_tokens=100, completion_tokens=20,
                  cost_usd=0.001)
    tracer.record(EventType.LLM_CALL, prompt_tokens=50, completion_tokens=10,
                  cost_usd=0.0005)

    totals = tracer.totals()
    assert totals.prompt_tokens == 150
    assert totals.completion_tokens == 30
    assert round(totals.cost_usd, 5) == 0.0015


def test_totals_count_llm_and_tool_calls_separately():
    tracer = Tracer(session_id="s1")
    tracer.record(EventType.LLM_CALL)
    tracer.record(EventType.TOOL_CALL, tool="a")
    tracer.record(EventType.TOOL_CALL, tool="b")

    totals = tracer.totals()
    assert totals.llm_calls == 1
    assert totals.tool_calls == 2


def test_totals_latency_sums_recorded_durations():
    tracer = Tracer(session_id="s1")
    tracer.record(EventType.LLM_CALL, duration_ms=120.0)
    tracer.record(EventType.TOOL_CALL, duration_ms=80.0)

    assert tracer.totals().latency_ms == 200.0


def test_flush_writes_one_json_object_per_line(tmp_path):
    tracer = Tracer(session_id="s1", trace_dir=str(tmp_path))
    tracer.record(EventType.QUERY_RECEIVED, query="hi")
    tracer.record(EventType.ANSWER_SYNTHESIZED, answer="hello")

    path = tracer.flush()

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == EventType.QUERY_RECEIVED


def test_flush_appends_rather_than_overwriting(tmp_path):
    first = Tracer(session_id="s1", trace_dir=str(tmp_path))
    first.record(EventType.QUERY_RECEIVED, query="one")
    first.flush()

    second = Tracer(session_id="s1", trace_dir=str(tmp_path))
    second.record(EventType.QUERY_RECEIVED, query="two")
    path = second.flush()

    assert len(path.read_text(encoding="utf-8").strip().split("\n")) == 2


def test_flush_is_a_noop_without_a_trace_directory():
    assert Tracer(session_id="s1").flush() is None


def test_load_trace_round_trips_written_events(tmp_path):
    tracer = Tracer(session_id="s1", trace_dir=str(tmp_path))
    tracer.record(EventType.TOOL_RESULT, tool="x", status="ok")
    path = tracer.flush()

    replayed = load_trace(path)
    assert replayed[0].payload["status"] == "ok"
