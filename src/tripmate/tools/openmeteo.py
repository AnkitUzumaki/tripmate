"""Open-Meteo HTTP transport: geocoding, forecast and archive (climate) fetches.

Pure transport layer — no tool logic, no caching, no fallback. `weather.py` owns
the routing, caching and mock-fallback decisions and calls into this module.
"""

from __future__ import annotations

from datetime import date, timedelta
from statistics import mean

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_FIELDS = "temperature_2m_max,temperature_2m_min,precipitation_sum"
# ERA5 reanalysis publishes with roughly a 5-day lag; archive queries must stay
# clear of that window or Open-Meteo returns null temperatures for those days.
ARCHIVE_LAG_DAYS = 7
RAINY_DAY_MM = 1.0


def geocode(city: str, timeout: float) -> tuple[float, float, str] | None:
    response = httpx.get(
        GEOCODE_URL, params={"name": city, "count": 1}, timeout=timeout
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None
    first = results[0]
    return float(first["latitude"]), float(first["longitude"]), str(first["name"])


def summarise(daily: dict, month: int | None) -> tuple[float, float, int, int]:
    """Average max/min temperature and count rainy days, optionally for one month.

    Skips any day with a null max or min temperature (the ERA5 publication-lag
    gap) rather than letting `mean()` raise on a `None` in the list.
    """
    times = daily["time"]
    maxes, mins, precip = [], [], []
    for index, stamp in enumerate(times):
        if month is not None and int(stamp.split("-")[1]) != month:
            continue
        day_max = daily["temperature_2m_max"][index]
        day_min = daily["temperature_2m_min"][index]
        if day_max is None or day_min is None:
            continue
        maxes.append(day_max)
        mins.append(day_min)
        precip.append(daily["precipitation_sum"][index] or 0.0)

    if not maxes:
        raise ValueError("no daily records for the requested period")

    rainy = sum(1 for value in precip if value >= RAINY_DAY_MM)
    days = len(maxes)
    return mean(mins), mean(maxes), rainy, days


def fetch_forecast(lat: float, lon: float, day: date, timeout: float) -> dict:
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


def fetch_archive(lat: float, lon: float, years: int, timeout: float) -> dict:
    end = (date.today() - timedelta(days=ARCHIVE_LAG_DAYS)).replace(day=1)
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
