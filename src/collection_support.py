"""Network diagnostics, rate-aware retries and historical weather fallback."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from typing import TypeVar

import numpy as np
import pandas as pd
import requests
from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError

from src.utils import DATA, LOG, read_frame, save_frame

T = TypeVar("T")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FIELDS = {
    "temperature_2m": "AirTemp",
    "relative_humidity_2m": "Humidity",
    "wind_speed_10m": "WindSpeed",
    "wind_direction_10m": "WindDirection",
    "precipitation": "Precipitation",
    "surface_pressure": "Pressure",
}


def pause(seconds: float) -> None:
    """Sleep in interruptible chunks; do not reset FastF1's request counters."""
    while seconds > 0:
        chunk = min(seconds, 30)
        time.sleep(chunk)
        seconds -= chunk


def rate_wait(exc: Exception, default: float) -> float | None:
    """Recognize FastF1 limits and HTTP 429, including Retry-After dates."""
    if isinstance(exc, RateLimitExceededError):
        return max(default, 3601)  # FastF1 uses rolling one-hour request budgets.
    response = getattr(exc, "response", None)
    if response is not None and response.status_code == 429:
        value = response.headers.get("Retry-After", "")
        try:
            return max(float(value), 1)
        except ValueError:
            try:
                return max(
                    (pd.Timestamp(parsedate_to_datetime(value)) - pd.Timestamp.now(tz="UTC")).total_seconds(),
                    1,
                )
            except (ValueError, TypeError):
                return default
    return None


def network_retry(
    fn: Callable[[], T], attempts: int = 3, backoff: float = 2, rate_limit_wait: float = 3601
) -> T:
    """Retry transient errors exponentially; pause separately on rate limiting."""
    failures = rate_hits = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            delay = rate_wait(exc, rate_limit_wait)
            if delay is not None:
                rate_hits += 1
                if rate_hits > attempts:
                    raise
                LOG.warning("Rate limit hit; pausing %.0fs, then resuming the same request", delay)
                pause(delay)
                continue
            failures += 1
            if failures >= attempts:
                raise
            delay = backoff * 2 ** (failures - 1)
            LOG.warning("Request failed (%s); retry %d/%d in %.1fs", exc, failures + 1, attempts, delay)
            pause(delay)


class RateLimitCapture(logging.Handler):
    """Recover limits swallowed by FastF1's soft-failure decorators."""

    error: Exception | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info and isinstance(record.exc_info[1], Exception):
            exc = record.exc_info[1]
            if rate_wait(exc, 3601) is not None:
                self.error = exc
        if "RateLimitExceededError" in record.getMessage():
            self.error = RateLimitExceededError(record.getMessage())


def safe_frame(session: object, attr: str) -> pd.DataFrame:
    """Missing FastF1 properties raise instead of returning an empty frame."""
    try:
        return pd.DataFrame(getattr(session, attr)).copy()
    except (AttributeError, RuntimeError, ValueError, DataNotLoadedError):
        return pd.DataFrame()


def archive_weather(
    latitude: float,
    longitude: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    attempts: int = 3,
    backoff: float = 2,
) -> pd.DataFrame:
    """Fetch hourly reanalysis for the UTC session window, cached by location/day.

    Archive data is retrospective, not a race-day forecast. A conservative seven-day
    publication delay is recorded so it can only contribute to subsequent history.
    """
    start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    if not np.isfinite([latitude, longitude]).all() or not (
        -90 <= latitude <= 90 and -180 <= longitude <= 180
    ):
        raise ValueError("Circuit coordinates unavailable or invalid")
    path = (
        DATA / "weather_archive" / f"{latitude:.5f}_{longitude:.5f}_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"
    )
    weather = read_frame(path)
    if weather.empty:

        def fetch() -> pd.DataFrame:
            response = requests.get(
                ARCHIVE_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": end.strftime("%Y-%m-%d"),
                    "hourly": ",".join(WEATHER_FIELDS),
                    "timezone": "UTC",
                    "wind_speed_unit": "ms",
                    "temperature_unit": "celsius",
                    "precipitation_unit": "mm",
                },
                timeout=(10, 45),
            )
            response.raise_for_status()
            frame = pd.DataFrame(response.json().get("hourly", {})).rename(columns=WEATHER_FIELDS)
            if frame.empty or "time" not in frame:
                raise ValueError("Open-Meteo returned no hourly observations")
            frame["observed_at"] = pd.to_datetime(frame.pop("time"), utc=True)
            return frame

        weather = network_retry(fetch, attempts=attempts, backoff=backoff)
        save_frame(weather, path)
    weather = weather[(weather.observed_at >= start.ceil("h")) & (weather.observed_at < end)].copy()
    if weather.empty or not weather[list(WEATHER_FIELDS.values())].notna().any().any():
        return pd.DataFrame()
    weather["Time"] = weather.observed_at - start
    weather["TrackTemp"] = np.nan  # Reanalysis has no track-surface sensor.
    weather["Rainfall"] = weather.Precipitation.gt(0).where(weather.Precipitation.notna())
    weather["weather_source"] = "open-meteo-archive"
    weather["weather_available_at"] = weather.observed_at + pd.Timedelta(days=7)
    return weather


def check_network() -> bool:
    """Probe real endpoints; distinguish DNS/proxy failures from HTTP refusals."""
    probes = {
        "FastF1 schedules": (
            "https://raw.githubusercontent.com/theOehrly/f1schedule/master/schedule_2024.json",
            None,
        ),
        "FastF1 live timing": (
            "https://livetiming.formula1.com/static/2024/2024-03-02_Bahrain_Grand_Prix/2024-03-02_Race/WeatherData.jsonStream",
            None,
        ),
        "FastF1 timing mirror": (
            "https://livetiming-mirror.fastf1.dev/static/2024/2024-03-02_Bahrain_Grand_Prix/2024-03-02_Race/WeatherData.jsonStream",
            None,
        ),
        "FastF1/Jolpica results": ("https://api.jolpi.ca/ergast/f1/2024/1/results.json?limit=1", None),
        "Open-Meteo archive": (
            ARCHIVE_URL,
            {
                "latitude": 26.0325,
                "longitude": 50.5106,
                "start_date": "2024-03-02",
                "end_date": "2024-03-02",
                "hourly": "temperature_2m",
            },
        ),
    }
    success = True
    for name, (url, params) in probes.items():
        try:
            response = requests.get(url, params=params, timeout=(5, 10))
            response.raise_for_status()
            print(f"OK: {name} (HTTP {response.status_code})")
        except requests.HTTPError as exc:
            success = False
            print(
                f"UNAVAILABLE: {name}: HTTP {exc.response.status_code}; server/authentication/rate-limit or network-policy refusal, not proof of a local DNS block."
            )
        except requests.RequestException as exc:
            success = False
            print(
                f"BLOCKED/UNREACHABLE: {name}: {exc}. Check DNS, proxy, firewall or environment network permissions."
            )
    return success
