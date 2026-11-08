"""Race-hour archive matching and coverage for explicitly retrospective models.

Race-window observations are never a substitute for a pre-race issued forecast.
The fixed two-hour window is chosen without using race duration or outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.collection_support import archive_weather, pause
from src.utils import DATA, LOG, read_frame, save_frame, setup

WEATHER_FEATURES = [
    "weather_AirTemp",
    "weather_Humidity",
    "weather_WindSpeed",
    "weather_Rainfall",
    "weather_is_wet",
]
WEATHER_COLUMNS = WEATHER_FEATURES + ["weather_TrackTemp", "weather_Precipitation"]


def race_start(row: pd.Series) -> pd.Timestamp:
    """Resolve race start in UTC from provider date/time, then timetable if absent."""
    time = row.get("raceTime")
    date = pd.to_datetime(row.get("raceDate", row.get("race_date")), utc=True)
    if pd.notna(time) and pd.notna(date):
        return pd.to_datetime(f"{date:%Y-%m-%d}T{time}", utc=True)
    schedule = read_frame(DATA / "raw" / f"{int(row.year)}_schedule.parquet")
    if not schedule.empty:
        event = schedule[schedule.RoundNumber == int(row["round"])]
        if not event.empty:
            event = event.iloc[0]
            for i in range(1, 6):
                if event.get(f"Session{i}") == "Race":
                    result = pd.to_datetime(event.get(f"Session{i}DateUtc"), utc=True)
                    if pd.notna(result):
                        return result
    raise ValueError("Race start time unavailable; refusing to guess a matching hour")


def summarize_weather(weather: pd.DataFrame) -> dict:
    """Aggregate a fixed hourly window; missing precipitation never means dry."""
    result = dict.fromkeys(WEATHER_COLUMNS, np.nan)
    for source, target in [
        ("AirTemp", "weather_AirTemp"),
        ("Humidity", "weather_Humidity"),
        ("WindSpeed", "weather_WindSpeed"),
    ]:
        if source in weather:
            result[target] = pd.to_numeric(weather[source], errors="coerce").mean()
    if "Precipitation" in weather:
        rain = pd.to_numeric(weather.Precipitation, errors="coerce").sum(min_count=1)
        result["weather_Rainfall"] = rain
        result["weather_Precipitation"] = rain
        result["weather_is_wet"] = float(rain > 0) if pd.notna(rain) else np.nan
    return result


def collect_race_weather(download: bool = False, delay: float = 1) -> pd.DataFrame:
    """Match each race by coordinates and UTC hour; optionally fetch missing days."""
    setup()
    races = read_frame(DATA / "race.parquet")
    if races.empty:
        raise ValueError("Collect race results before weather matching")
    rows = []
    for (_, _), event in races.groupby(["year", "round"], sort=True):
        entry = event.iloc[0]
        row = {
            "year": int(entry.year),
            "round": int(entry["round"]),
            "weather_scope": "observed-race-window",
            "weather_source": "open-meteo-archive",
            "weather_error": "",
        }
        row.update(dict.fromkeys(WEATHER_COLUMNS, np.nan))
        try:
            start = race_start(entry)
            anchor = start.floor("h")
            end = anchor + pd.Timedelta(hours=2)
            latitude, longitude = float(entry.lat), float(entry["long"])
            row.update(
                weather_latitude=latitude,
                weather_longitude=longitude,
                weather_race_start=start,
                weather_window_start=anchor,
                weather_window_end=end,
            )
            cache = (
                DATA
                / "weather_archive"
                / f"{latitude:.5f}_{longitude:.5f}_{anchor:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"
            )
            if not cache.exists() and not download:
                raise FileNotFoundError(f"Archive day not cached: {cache.name}; run --download")
            if not cache.exists():
                pause(delay)
            weather = archive_weather(latitude, longitude, anchor, end)
            row.update(summarize_weather(weather))
            row["weather_hour_count"] = len(weather)
            row["weather_available_at"] = weather.weather_available_at.max() if not weather.empty else pd.NaT
            if len(weather) < 2:
                # Do not silently accept a partial two-hour window as complete.
                row.update(dict.fromkeys(WEATHER_COLUMNS, np.nan))
                row["weather_error"] = "Incomplete two-hour race window"
        except Exception as exc:
            row["weather_error"] = str(exc)
            LOG.warning("Weather %s/%s: %s", row["year"], row["round"], exc)
        rows.append(row)
        # Archive requests checkpoint individually; publish the event table only after validation.
    frame = pd.DataFrame(rows)
    joined = races[["year", "round", "driver"]].merge(frame, on=["year", "round"], validate="many_to_one")
    coverage = {col: float(joined[col].notna().mean()) for col in WEATHER_COLUMNS}
    report = {
        "rows": len(joined),
        "races": len(frame),
        "coverage": coverage,
        "threshold": 0.95,
        "supported_fields_pass": all(coverage[c] >= 0.95 for c in WEATHER_FEATURES),
        "unsupported": {
            "weather_TrackTemp": "Open-Meteo has no track-surface temperature; excluded from models"
        },
        "window": "[floor(race start UTC hour), +2 hours); mean air/humidity/wind, summed precipitation in mm",
        "scope": "Retrospective observed weather. Not available before this race; future inference requires forecast data.",
    }
    (DATA / "weather_coverage.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["supported_fields_pass"]:
        raise ValueError("Weather coverage below 95% for supported fields; see data/weather_coverage.json")
    save_frame(frame, DATA / "race_weather.parquet")
    return frame


def attach_weather(features: pd.DataFrame) -> pd.DataFrame:
    """Attach observations for analysis; estimator allowlists control their use."""
    weather = read_frame(DATA / "race_weather.parquet")
    result = features.drop(columns=[c for c in features if c.startswith("weather_")], errors="ignore")
    if not weather.empty:
        result = result.merge(weather, on=["year", "round"], how="left", validate="many_to_one")
    for col in WEATHER_COLUMNS:
        if col not in result:
            result[col] = np.nan
    return result


def load_race_weather_samples(year: int, rnd: int) -> pd.DataFrame:
    """Read the matched race-hour bins for the UI without making network requests."""
    events = read_frame(DATA / "race_weather.parquet")
    if events.empty:
        return pd.DataFrame()
    match = events[(events.year == year) & (events["round"] == rnd)]
    if match.empty:
        return pd.DataFrame()
    row = match.iloc[0]
    start, end = row.weather_window_start, row.weather_window_end
    path = (
        DATA
        / "weather_archive"
        / f"{row.weather_latitude:.5f}_{row.weather_longitude:.5f}_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"
    )
    weather = read_frame(path)
    if weather.empty:
        return weather
    weather = weather[(weather.observed_at >= start) & (weather.observed_at < end)].copy()
    weather["Time"] = weather.observed_at - row.weather_race_start
    weather["TrackTemp"] = np.nan
    weather["Rainfall"] = weather.Precipitation.gt(0).where(weather.Precipitation.notna())
    weather["weather_source"] = "open-meteo-archive"
    return weather


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch uncached archive days")
    parser.add_argument("--delay", type=float, default=1)
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be nonnegative")
    collect_race_weather(args.download, args.delay)
