from tripmate.models import Citation, Chunk, ToolResult, WeatherReport


def test_chunk_defaults_score_to_zero():
    chunk = Chunk(id="a1", text="t", city="tokyo", section="VISA & ENTRY",
                  source_file="tokyo.txt")
    assert chunk.score == 0.0


def test_citation_ref_is_city_slash_uppercase_section():
    assert Citation(city="tokyo", section="PACKING TIPS").ref == "tokyo/PACKING TIPS"


def test_citation_parse_accepts_bracketed_reference():
    citation = Citation.parse("[tokyo/PACKING TIPS]")
    assert citation == Citation(city="tokyo", section="PACKING TIPS")


def test_citation_parse_lowercases_city_and_uppercases_section():
    citation = Citation.parse("[Tokyo/packing tips]")
    assert citation.ref == "tokyo/PACKING TIPS"


def test_citation_parse_returns_none_for_malformed_text():
    assert Citation.parse("not a citation") is None


def test_tool_result_ok_carries_data_and_no_reason():
    result = ToolResult.ok("get_weather_forecast", {"temp": 5})
    assert result.status == "ok"
    assert result.data == {"temp": 5}
    assert result.reason is None


def test_tool_result_no_data_carries_reason_and_extra_fields():
    result = ToolResult.no_data(
        "search_destination_guide", "no guide for paris",
        available_cities=["tokyo"],
    )
    assert result.status == "no_data"
    assert result.reason == "no guide for paris"
    assert result.data == {"available_cities": ["tokyo"]}


def test_tool_result_error_has_error_status():
    result = ToolResult.error("get_weather_forecast", "timeout")
    assert result.status == "error"
    assert result.reason == "timeout"


def test_weather_report_serialises_to_dict():
    report = WeatherReport(
        city="Tokyo", period="December", source="climate_normal",
        conditions="cold, dry", temp_range_c=[3.0, 12.0], precip_days=4,
    )
    assert report.model_dump()["source"] == "climate_normal"
