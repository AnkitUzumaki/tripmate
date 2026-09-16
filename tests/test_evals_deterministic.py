from pathlib import Path

from evals.deterministic import EvalCase, load_dataset, score_case
from tripmate.core.trace import EventType
from tripmate.models import AgentResponse, Citation, TraceEvent


def _response(answer: str, tools: list[str], citations: list[str] | None = None):
    trace = [
        TraceEvent(seq=index + 1, event_type=EventType.TOOL_CALL, timestamp=0.0,
                   payload={"tool": tool})
        for index, tool in enumerate(tools)
    ]
    return AgentResponse(
        answer=answer,
        citations=[Citation(city=c.split("/")[0], section=c.split("/")[1])
                   for c in (citations or [])],
        trace=trace,
    )


def test_dataset_loads_every_case():
    cases = load_dataset(Path("evals/dataset.yaml"))
    assert len(cases) >= 30


def test_dataset_case_ids_are_unique():
    cases = load_dataset(Path("evals/dataset.yaml"))
    assert len({case.id for case in cases}) == len(cases)


def test_correct_tool_selection_scores_true():
    case = EvalCase(id="c", query="q", expected_tools=["get_weather_forecast"])
    score = score_case(case, _response("Cold.", ["get_weather_forecast"]))
    assert score.tool_selection_correct


def test_tool_selection_ignores_call_order():
    case = EvalCase(id="c", query="q",
                    expected_tools=["get_weather_forecast", "search_destination_guide"])
    score = score_case(case, _response(
        "x", ["search_destination_guide", "get_weather_forecast"]))
    assert score.tool_selection_correct


def test_missing_expected_tool_scores_false():
    case = EvalCase(id="c", query="q",
                    expected_tools=["search_destination_guide", "get_weather_forecast"])
    score = score_case(case, _response("x", ["search_destination_guide"]))
    assert not score.tool_selection_correct


def test_required_mention_is_case_insensitive():
    case = EvalCase(id="c", query="q", expected_tools=[], must_mention=["LAYER"])
    assert score_case(case, _response("Pack layers.", [])).mentions_ok


def test_forbidden_mention_fails_the_case():
    case = EvalCase(id="c", query="q", expected_tools=[],
                    must_not_mention=["booked"])
    assert not score_case(case, _response("I have booked your flight.", [])).mentions_ok


def test_refusal_is_detected_from_refusal_phrases():
    case = EvalCase(id="c", query="q", expected_tools=[], must_refuse=True)
    assert score_case(case, _response("I can't book flights.", [])).refusal_correct


def test_missing_refusal_scores_false():
    case = EvalCase(id="c", query="q", expected_tools=[], must_refuse=True)
    assert not score_case(case, _response("Sure, done!", [])).refusal_correct


def test_required_citation_present_scores_true():
    case = EvalCase(id="c", query="q", expected_tools=[],
                    must_cite=["tokyo/PACKING TIPS"])
    score = score_case(case, _response("Layers.", [], ["tokyo/PACKING TIPS"]))
    assert score.citations_valid


def test_required_citation_absent_scores_false():
    case = EvalCase(id="c", query="q", expected_tools=[],
                    must_cite=["tokyo/PACKING TIPS"])
    assert not score_case(case, _response("Layers.", [], [])).citations_valid


def test_case_passes_only_when_every_check_passes():
    case = EvalCase(id="c", query="q", expected_tools=["get_weather_forecast"],
                    must_mention=["cold"])
    assert score_case(case, _response("It is cold.", ["get_weather_forecast"])).passed
    assert not score_case(case, _response("It is warm.", ["get_weather_forecast"])).passed
