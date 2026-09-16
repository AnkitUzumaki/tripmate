from datetime import date

import httpx
import pytest
import respx

from tripmate.tools.weather import (
    ARCHIVE_URL,
    FORECAST_URL,
    GEOCODE_URL,
    clear_cache,
    describe,
    get_weather_forecast,
    resolve_period,
)

GEOCODE_OK = {"results": [{"latitude": 35.68, "longitude": 139.75, "name": "Tokyo"}]}


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


def test_resolve_period_parses_a_month_name():
    period = resolve_period("December")
    assert period.kind == "month"
    assert period.month == 12


def test_resolve_period_parses_an_abbreviated_month():
    assert resolve_period("dec").month == 12


def test_resolve_period_parses_an_iso_date_near_today():
    today = date(2026, 9, 16)
    period = resolve_period("2026-09-20", today=today)
    assert period.kind == "date"


def test_resolve_period_downgrades_a_far_future_date_to_its_month():
    today = date(2026, 9, 16)
    period = resolve_period("2026-12-25", today=today)
    assert period.kind == "month"
    assert period.month == 12


def test_resolve_period_rejects_unparseable_text():
    with pytest.raises(ValueError):
        resolve_period("sometime soonish")


def test_describe_reports_cold_for_low_temperatures():
    assert "cold" in describe(-3.0, 2.0, 2)


def test_describe_reports_hot_for_high_temperatures():
    assert "hot" in describe(30.0, 36.0, 1)


def test_describe_mentions_rain_when_precipitation_days_are_high():
    assert "rain" in describe(20.0, 26.0, 18)


def test_describe_does_not_call_a_single_rainy_day_frequent_rain():
    assert "frequent rain" not in describe(18.0, 24.0, 1, total_days=1)


@respx.mock
def test_month_query_uses_the_archive_endpoint_and_labels_climate_normal():
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_OK))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json={
            "daily": {
                "time": ["2021-12-01", "2021-12-02", "2022-12-01"],
                "temperature_2m_max": [12.0, 11.0, 13.0],
                "temperature_2m_min": [3.0, 2.0, 4.0],
                "precipitation_sum": [0.0, 5.0, 0.0],
            }
        })
    )

    result = get_weather_forecast("Tokyo", "December")

    assert result.status == "ok"
    assert result.data["source"] == "climate_normal"
    assert result.data["period"] == "December"


@respx.mock
def test_near_date_query_uses_the_forecast_endpoint():
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_OK))
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(200, json={
            "daily": {
                "time": ["2026-09-18"],
                "temperature_2m_max": [26.0],
                "temperature_2m_min": [19.0],
                "precipitation_sum": [0.0],
            }
        })
    )

    result = get_weather_forecast("Tokyo", date.today().isoformat())

    assert result.data["source"] == "forecast"


@respx.mock
def test_unknown_city_returns_no_data():
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json={"results": []}))

    result = get_weather_forecast("Atlantis", "July")

    assert result.status == "no_data"
    assert "Atlantis" in result.reason


@respx.mock
def test_api_timeout_falls_back_to_the_mock_table_for_covered_cities():
    respx.get(GEOCODE_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    result = get_weather_forecast("Tokyo", "December")

    assert result.status == "ok"
    assert result.data["source"] == "mock_fallback"


@respx.mock
def test_api_failure_for_an_uncovered_city_returns_error():
    respx.get(GEOCODE_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    result = get_weather_forecast("Lisbon", "December")

    assert result.status == "error"


@respx.mock
def test_malformed_period_returns_no_data_without_calling_the_api():
    route = respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_OK))

    result = get_weather_forecast("Tokyo", "whenever")

    assert result.status == "no_data"
    assert not route.called


@respx.mock
def test_repeated_month_query_is_served_from_cache():
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_OK))
    archive = respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json={
            "daily": {
                "time": ["2021-12-01"],
                "temperature_2m_max": [12.0],
                "temperature_2m_min": [3.0],
                "precipitation_sum": [0.0],
            }
        })
    )

    get_weather_forecast("Tokyo", "December")
    get_weather_forecast("Tokyo", "December")

    assert archive.call_count == 1
