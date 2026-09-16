"""Weather tool over Open-Meteo, with a climate-normal path and a mock fallback.

A forecast API reaches about 16 days. "What should I pack for Tokyo in December?"
is not a forecast question — it is a climate-normal question, answered from
historical reanalysis averaged over the last N years. The `source` field on every
response states which path produced it, so the agent never misrepresents what it knows.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime
from statistics import mean
from typing import Annotated, Literal

import diskcache
import httpx

from tripmate.config import get_settings
from tripmate.models import ToolResult, WeatherReport
from tripmate.tools.registry import tool

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_HORIZON_DAYS = 16
ARCHIVE_LAG_DAYS = 7
DAILY_FIELDS = "temperature_2m_max,temperature_2m_min,precipitation_sum"
RAINY_DAY_MM = 1.0
FORECAST_TTL_S = 3600

COLD_MAX_C = 10.0
HOT_MIN_C = 28.0
WET_RATIO = 0.4
DRY_RATIO = 0.15
# ponytail: a 1-day sample can't establish a "frequent rain" pattern — flooring
# the ratio's denominator stops a single rainy forecast day from reading as a
# rainy season. Raise this if precip classification needs to react to spans
# shorter than 3 days.
MIN_RATIO_SPAN_DAYS = 3

TOOL_NAME = "get_weather_forecast"

MONTH_NUMBERS: dict[str, int] = {
    name.lower(): index
    for index, name in enumerate(calendar.month_name)
    if name
} | {
    name.lower(): index
    for index, name in enumerate(calendar.month_abbr)
    if name
}

# Seasonal fallback used only when Open-Meteo is unreachable.
# (temp_min_c, temp_max_c, precip_days, conditions)
MOCK_CLIMATE: dict[str, dict[str, tuple[float, float, int, str]]] = {
    "tokyo": {
        "winter": (2.0, 10.0, 5, "cold, dry, mostly clear"),
        "spring": (10.0, 20.0, 10, "mild, occasional showers"),
        "summer": (23.0, 31.0, 14, "hot, humid, frequent rain"),
        "autumn": (14.0, 23.0, 11, "mild, some rain"),
    },
    "reykjavik": {
        "winter": (-3.0, 3.0, 15, "cold, windy, occasional snow"),
        "spring": (1.0, 8.0, 12, "chilly, changeable, windy"),
        "summer": (8.0, 14.0, 11, "cool, breezy, long daylight"),
        "autumn": (2.0, 8.0, 15, "cold, wet, windy"),
    },
    "bangkok": {
        "winter": (21.0, 32.0, 2, "hot, dry, sunny"),
        "spring": (26.0, 35.0, 6, "very hot, humid"),
        "summer": (25.0, 33.0, 17, "hot, humid, heavy monsoon rain"),
        "autumn": (24.0, 32.0, 18, "hot, humid, frequent heavy rain"),
    },
    "barcelona": {
        "winter": (5.0, 14.0, 5, "mild, some rain"),
        "spring": (10.0, 20.0, 6, "mild, pleasant"),
        "summer": (20.0, 29.0, 3, "hot, dry, sunny"),
        "autumn": (13.0, 23.0, 8, "mild, occasional heavy rain"),
    },
}

SEASONS: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

_cache: diskcache.Cache | None = None


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(get_settings().weather_cache_dir)
    return _cache


def clear_cache() -> None:
    _get_cache().clear()


@dataclass(frozen=True)
class Period:
    kind: Literal["date", "month"]
    month: int
    label: str
    day: date | None = None


def resolve_period(text: str, today: date | None = None) -> Period:
    """Classify a date-or-month string.

    An ISO date inside the forecast horizon stays a date; anything else becomes a
    month, because only a climate normal can answer it.
    """
    today = today or date.today()
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty date_or_month")

    try:
        parsed = datetime.strptime(cleaned, "%Y-%m-%d").date()
    except ValueError:
        month = MONTH_NUMBERS.get(cleaned.lower())
        if month is None:
            raise ValueError(f"cannot interpret {text!r} as a date or month")
        return Period(kind="month", month=month, label=calendar.month_name[month])

    delta_days = (parsed - today).days
    if 0 <= delta_days <= FORECAST_HORIZON_DAYS:
        return Period(kind="date", month=parsed.month, label=parsed.isoformat(),
                      day=parsed)
    return Period(kind="month", month=parsed.month,
                  label=calendar.month_name[parsed.month])


def describe(temp_min: float, temp_max: float, precip_days: int,
             total_days: int = 30) -> str:
    """Turn numbers into a short human phrase."""
    if temp_max <= COLD_MAX_C:
        temperature = "cold"
    elif temp_min >= HOT_MIN_C:
        temperature = "hot"
    elif temp_max >= 24.0:
        temperature = "warm"
    else:
        temperature = "mild"

    ratio = precip_days / max(total_days, MIN_RATIO_SPAN_DAYS)
    if ratio >= WET_RATIO:
        precipitation = "frequent rain"
    elif ratio <= DRY_RATIO:
        precipitation = "mostly dry"
    else:
        precipitation = "occasional rain"

    return f"{temperature}, {precipitation}"


def _geocode(city: str, timeout: float) -> tuple[float, float, str] | None:
    response = httpx.get(
        GEOCODE_URL, params={"name": city, "count": 1}, timeout=timeout
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None
    first = results[0]
    return float(first["latitude"]), float(first["longitude"]), str(first["name"])


def _summarise(daily: dict, month: int | None) -> tuple[float, float, int, int]:
    """Average max/min temperature and count rainy days, optionally for one month."""
    times = daily["time"]
    maxes, mins, precip = [], [], []
    for index, stamp in enumerate(times):
        if month is not None and int(stamp.split("-")[1]) != month:
            continue
        maxes.append(daily["temperature_2m_max"][index])
        mins.append(daily["temperature_2m_min"][index])
        precip.append(daily["precipitation_sum"][index] or 0.0)

    if not maxes:
        raise ValueError("no daily records for the requested period")

    rainy = sum(1 for value in precip if value >= RAINY_DAY_MM)
    days = len(maxes)
    return mean(mins), mean(maxes), rainy, days


def _fetch_forecast(lat: float, lon: float, day: date, timeout: float) -> dict:
    response = httpx.get(
        FORECAST_URL,
        params={
            "latitude": lat, "longitude": lon, "daily": DAILY_FIELDS,
            "start_date": day.isoformat(), "end_date": day.isoformat(),
            "timezone": "auto",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["daily"]


def _fetch_archive(lat: float, lon: float, years: int, timeout: float) -> dict:
    end = date.today().replace(day=1)
    start = end.replace(year=end.year - years)
    response = httpx.get(
        ARCHIVE_URL,
        params={
            "latitude": lat, "longitude": lon, "daily": DAILY_FIELDS,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "timezone": "auto",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["daily"]


def _mock_report(city: str, period: Period) -> WeatherReport | None:
    seasons = MOCK_CLIMATE.get(city.strip().lower())
    if seasons is None:
        return None
    low, high, rainy, conditions = seasons[SEASONS[period.month]]
    return WeatherReport(
        city=city.title(), period=period.label, source="mock_fallback",
        conditions=conditions, temp_range_c=[low, high], precip_days=rainy,
    )


@tool
def get_weather_forecast(
    city: Annotated[str, "City name, e.g. 'Tokyo'."],
    date_or_month: Annotated[
        str,
        "An ISO date like '2026-12-25' for near-term forecasts, or a month name "
        "like 'December' for typical seasonal conditions.",
    ],
) -> ToolResult:
    """Get the expected weather for a city on a specific date or during a given month."""
    settings = get_settings()

    try:
        period = resolve_period(date_or_month)
    except ValueError as exc:
        return ToolResult.no_data(TOOL_NAME, str(exc))

    cache_key = f"{period.kind}:{city.strip().lower()}:{period.label}"
    cached = _get_cache().get(cache_key)
    if cached is not None:
        return ToolResult.ok(TOOL_NAME, cached)

    try:
        located = _geocode(city, settings.weather_timeout_s)
        if located is None:
            return ToolResult.no_data(
                TOOL_NAME, f"could not find a city named {city!r}"
            )
        latitude, longitude, resolved_name = located

        if period.kind == "date" and period.day is not None:
            daily = _fetch_forecast(latitude, longitude, period.day,
                                    settings.weather_timeout_s)
            low, high, rainy, days = _summarise(daily, month=None)
            source = "forecast"
            precip_days = rainy
        else:
            daily = _fetch_archive(latitude, longitude,
                                   settings.weather_climate_years,
                                   settings.weather_timeout_s)
            low, high, rainy, days = _summarise(daily, month=period.month)
            source = "climate_normal"
            precip_days = round(rainy * 30 / days) if days else 0

        report = WeatherReport(
            city=resolved_name, period=period.label, source=source,
            conditions=describe(low, high, rainy, total_days=days),
            temp_range_c=[round(low, 1), round(high, 1)], precip_days=precip_days,
        )

    except (httpx.HTTPError, KeyError, ValueError) as exc:
        fallback = _mock_report(city, period)
        if fallback is None:
            return ToolResult.error(
                TOOL_NAME,
                f"weather lookup failed for {city!r} and no offline data is "
                f"available: {type(exc).__name__}: {exc}",
            )
        payload = fallback.model_dump()
        return ToolResult.ok(TOOL_NAME, payload)

    payload = report.model_dump()
    if source == "climate_normal":
        _get_cache().set(cache_key, payload)  # normals never change
    else:
        _get_cache().set(cache_key, payload, expire=FORECAST_TTL_S)
    return ToolResult.ok(TOOL_NAME, payload)
