"""Resumable FastF1/Jolpica results and optional timing-session collection.

Run from the project root: python src/data_collection.py --mode full.
Raw race timing is stored for exploration and lagged history, never current-race features.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from types import SimpleNamespace
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fastf1
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast

from src.utils import DATA, LOG, read_frame, save_frame, seconds, setup
from src.collection_support import (
    RateLimitCapture,
    archive_weather,
    check_network,
    network_retry as retry,
    pause,
    safe_frame,
)

RENAME = {
    "season": "year",
    "round": "round",
    "driverId": "driver",
    "constructorId": "team",
    "driverCode": "code",
    "circuitId": "circuit",
    "raceName": "race",
    "position": "finish",
    "grid": "grid",
    "dateOfBirth": "birth_date",
    "fastestLapRank": "fastest_lap_rank",
}


def fetch_results(year: int, kind: str) -> pd.DataFrame:
    """Fetch every page; season responses can split an event across pages."""
    api = Ergast(result_type="pandas", auto_cast=True, limit=100)
    method = {
        "race": api.get_race_results,
        "quali": api.get_qualifying_results,
        "sprint": api.get_sprint_results,
    }[kind]
    response = retry(lambda: method(season=year))
    frames = []
    while True:
        for (_, meta), content in zip(response.description.iterrows(), response.content, strict=True):
            frame = pd.DataFrame(content).copy()
            for key, value in meta.items():
                frame[key] = value
            frames.append(frame)
        if response.is_complete or not response.content:
            break
        try:
            # is_complete stays false on the final page when its offset is nonzero.
            if (
                int(response._response_headers.get("offset", 0))
                + int(response._response_headers.get("limit", 100))
                >= response.total_results
            ):
                break
            response = retry(response.get_next_result_page)
        except ValueError:
            break
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True).rename(columns=RENAME)
    result = result.drop_duplicates(["year", "round", "driver"])
    result["race_date"] = pd.to_datetime(result["raceDate"], utc=True)
    # A result may only be used as an outcome after its event has taken place.
    if kind == "race":
        result = result[result.race_date < pd.Timestamp.now(tz="UTC")]
    if kind == "quali":
        result = result.rename(columns={"finish": "quali_position"})
        for col in ("Q1", "Q2", "Q3"):
            result[col.lower()] = seconds(result[col]) if col in result else np.nan
    return result


def collect_core(start: int, end: int, refresh: bool = False) -> None:
    """Collect race, qualifying, sprint results and schedules by season."""
    setup()
    errors = []
    for year in range(start, end + 1):
        for kind in ("race", "quali", "sprint"):
            path = DATA / "raw" / f"{year}_{kind}.parquet"
            if path.exists() and not refresh and year < datetime.now().year:
                continue
            try:
                frame = fetch_results(year, kind)
                if not frame.empty:
                    save_frame(frame, path)
                LOG.info("%s %s: %s rows", year, kind, len(frame))
            except Exception as exc:
                errors.append({"year": year, "kind": kind, "error": str(exc)})
                LOG.exception("Unable to collect %s %s", year, kind)
        path = DATA / "raw" / f"{year}_schedule.parquet"
        if not path.exists() or refresh or year == datetime.now().year:
            try:
                schedule = retry(lambda year=year: fastf1.get_event_schedule(year, include_testing=False))
                save_frame(schedule, path)
            except Exception as exc:
                errors.append({"year": year, "kind": "schedule", "error": str(exc)})
    (DATA / "collection_errors.json").write_text(json.dumps(errors, indent=2))
    consolidate()


def consolidate() -> None:
    """Combine successful season checkpoints, preserving provider metadata."""
    for kind in ("race", "quali", "sprint"):
        frames = [pd.read_parquet(p) for p in sorted((DATA / "raw").glob(f"*_{kind}.parquet"))]
        if frames:
            save_frame(
                pd.concat(frames, ignore_index=True).drop_duplicates(["year", "round", "driver"]),
                DATA / f"{kind}.parquet",
            )


def summarize_session(session: object, year: int, rnd: int, name: str) -> pd.DataFrame:
    """Summarize timing; long runs are a proxy, as fuel loads are unobserved."""
    laps = safe_frame(session, "laps")
    results = safe_frame(session, "results")
    weather = safe_frame(session, "weather_data")
    if results.empty:
        return pd.DataFrame()
    rows = []
    for _, driver in results.iterrows():
        dl = (
            laps[laps.DriverNumber.astype(str) == str(driver.DriverNumber)].copy() if not laps.empty else laps
        )
        row = {
            "year": year,
            "round": rnd,
            "session": name,
            "driver": driver.get("DriverId", ""),
            "code": driver.get("Abbreviation", ""),
            "team_name": driver.get("TeamName", ""),
            "team_color": driver.get("TeamColor", ""),
            "session_date": pd.Timestamp(session.date).tz_localize("UTC")
            if pd.Timestamp(session.date).tzinfo is None
            else pd.Timestamp(session.date),
        }
        row.update({name: np.nan for name in ("best_lap", "long_run", "sector1", "sector2", "sector3")})
        row["has_practice_data"] = False
        if not dl.empty:
            valid = dl[dl.IsAccurate.fillna(False) & dl.PitInTime.isna() & dl.PitOutTime.isna()].copy()
            valid["lap_seconds"] = seconds(valid.LapTime)
            valid = valid[valid.lap_seconds > 0]
            # Filter conspicuously slow laps; >=5 clean laps per stint defines a long run.
            if not valid.empty:
                valid = valid[valid.lap_seconds < valid.lap_seconds.min() * 1.12]
                long = valid.groupby("Stint").filter(lambda group: len(group) >= 5)
                row["best_lap"] = valid.lap_seconds.min()
                row["has_practice_data"] = name.startswith("Practice") and pd.notna(row["best_lap"])
                row["long_run"] = long.lap_seconds.median()
                for sector in (1, 2, 3):
                    row[f"sector{sector}"] = seconds(valid[f"Sector{sector}Time"]).median()
            row["compounds"] = ",".join(sorted(dl.Compound.dropna().unique()))
            row["tyre_age"] = dl.TyreLife.mean()
            row["stint_length"] = dl.groupby("Stint").size().mean()
            row["pit_count"] = int(dl.PitInTime.notna().sum())
            # Pair pit-in with the following lap's pit-out; this includes pit-lane transit.
            row["pit_duration"] = (
                (dl.PitOutTime.shift(-1) - dl.PitInTime)
                .dt.total_seconds()
                .where(lambda x: x.between(0, 300))
                .mean()
            )
            for code, label in (("4", "sc"), ("6", "vsc"), ("5", "red")):
                active = dl.TrackStatus.astype(str).str.contains(code)
                row[f"{label}_periods"] = int((active & ~active.shift(fill_value=False)).sum())
        row["has_weather_data"] = not weather.empty
        row["weather_source"] = (
            weather.weather_source.iloc[0] if "weather_source" in weather and len(weather) else "missing"
        )
        row["weather_available_at"] = (
            weather.weather_available_at.max() if "weather_available_at" in weather else pd.NaT
        )
        if not weather.empty:
            for col in (
                "AirTemp",
                "TrackTemp",
                "Humidity",
                "Pressure",
                "WindSpeed",
                "Rainfall",
                "Precipitation",
            ):
                row[col] = pd.to_numeric(weather.get(col, pd.Series(dtype=float)), errors="coerce").mean()
            rad = np.deg2rad(weather.get("WindDirection", pd.Series(dtype=float)))
            row["WindDirection"] = float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)
            wet = weather.get("Rainfall", pd.Series(dtype=float)).dropna()
            row["conditions"] = (
                "unknown" if wet.empty else "wet" if wet.all() else "mixed" if wet.any() else "dry"
            )
        rows.append(row)
    return pd.DataFrame(rows)


LOG_FIELDS = [
    "timestamp",
    "year",
    "round",
    "session",
    "status",
    "has_laps",
    "has_weather",
    "weather_source",
    "detail",
]


class MissingSessionError(ValueError):
    """The provider returned no usable lap observations."""


def log_session(
    year: int,
    rnd: int,
    name: str,
    status: str,
    has_laps: bool = False,
    has_weather: bool = False,
    weather_source: str = "missing",
    detail: str = "",
) -> None:
    """Append and flush each outcome, including partial or failed downloads."""
    path = DATA / "collection_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(
            dict(
                timestamp=pd.Timestamp.now(tz="UTC").isoformat(),
                year=year,
                round=rnd,
                session=name,
                status=status,
                has_laps=has_laps,
                has_weather=has_weather,
                weather_source=weather_source,
                detail=detail,
            )
        )
        handle.flush()
        os.fsync(handle.fileno())


def latest_collection_status() -> dict:
    """Last outcome wins; a recovered failure is not retried again."""
    path = DATA / "collection_log.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path).drop_duplicates(["year", "round", "session"], keep="last")
    return {(int(r.year), int(r["round"]), r.session): r.status for _, r in frame.iterrows()}


def event_metadata(year: int, rnd: int, event: pd.Series) -> pd.DataFrame:
    """Use results/qualifying identities solely to retain historical driver rows."""
    for kind in ("quali", "race"):
        frame = read_frame(DATA / f"{kind}.parquet")
        if not frame.empty:
            rows = frame[(frame.year == year) & (frame["round"] == rnd)]
            if not rows.empty:
                return rows
    # Future event: coordinates may be reused, but never invent an entry list.
    return pd.DataFrame()


def collect_sessions(
    year: int,
    rounds: list[int] | None = None,
    delay: float = 2,
    attempts: int = 3,
    backoff: float = 2,
    retry_failed: bool = False,
    weather_only: bool = False,
) -> None:
    """Resume individual sessions, retaining partial data and persistent outcomes."""
    setup()
    schedule = read_frame(DATA / "raw" / f"{year}_schedule.parquet")
    if schedule.empty:
        schedule = retry(lambda: fastf1.get_event_schedule(year, include_testing=False), attempts, backoff)
        save_frame(schedule, DATA / "raw" / f"{year}_schedule.parquet")
    latest = latest_collection_status()
    for _, event in schedule.iterrows():
        rnd = int(event.RoundNumber)
        if rounds and rnd not in rounds:
            continue
        meta = event_metadata(year, rnd, event)
        for i in range(1, 6):
            name = event.get(f"Session{i}")
            if name not in ("Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"):
                continue
            key = (year, rnd, name)
            path = DATA / "sessions" / f"{year}_{rnd}_{name.replace(' ', '_')}"
            if retry_failed and latest.get(key) not in ("failed", "missing"):
                continue
            if latest.get(key) == "success" and (path / "summary.parquet").exists():
                continue
            date = pd.to_datetime(event.get(f"Session{i}DateUtc"), utc=True)
            if pd.isna(date) or date + pd.Timedelta(hours=4) > pd.Timestamp.now(tz="UTC"):
                log_session(year, rnd, name, "missing", detail="Session not yet completed or time unknown")
                continue
            pause(delay)
            session = None
            load_error = None
            try:
                if weather_only:
                    raise MissingSessionError(
                        "FastF1 lap load skipped (--weather-only); cached laps retained"
                    )
                session = fastf1.get_session(year, rnd, name)

                def load(session=session):
                    capture = RateLimitCapture()
                    logger = logging.getLogger("fastf1")
                    logger.addHandler(capture)
                    try:
                        session.load(laps=True, weather=True, telemetry=False, messages=False)
                    finally:
                        logger.removeHandler(capture)
                    if capture.error is not None:
                        raise capture.error
                    if safe_frame(session, "laps").empty:
                        raise MissingSessionError(
                            "FastF1 returned no laps (missing feed, cancelled session or access failure)"
                        )

                retry(load, attempts=attempts, backoff=backoff)
            except Exception as exc:
                load_error = exc
                LOG.warning("%s %s %s: %s", year, rnd, name, exc)
            laps = safe_frame(session, "laps")
            results = safe_frame(session, "results")
            weather = safe_frame(session, "weather_data")
            # A retry must never replace previously downloaded observations with NaNs.
            if laps.empty:
                laps = read_frame(path / "laps.parquet")
            if results.empty:
                results = read_frame(path / "results.parquet")
            if weather.empty:
                weather = read_frame(path / "weather_data.parquet")
            if results.empty and not meta.empty:
                results = pd.DataFrame(
                    {
                        "DriverId": meta.driver,
                        "Abbreviation": meta.get("code", ""),
                        "DriverNumber": meta.get("number", ""),
                        "TeamName": meta.get("constructorName", meta.team),
                    }
                )
            weather_error = ""
            end = date + pd.Timedelta(hours=3 if name == "Race" else 2)
            if (
                weather.empty
                or not weather.reindex(columns=["AirTemp", "Humidity", "WindSpeed"]).notna().any().any()
            ):
                weather = pd.DataFrame()
                try:
                    coords = meta
                    if coords.empty:
                        history = read_frame(DATA / "race.parquet")
                        coords = history[history.race == event.EventName] if not history.empty else history
                    if coords.empty:
                        raise ValueError("No circuit coordinates available")
                    weather = archive_weather(
                        float(coords.iloc[-1].lat),
                        float(coords.iloc[-1]["long"]),
                        date,
                        end,
                        attempts=attempts,
                        backoff=backoff,
                    )
                except Exception as exc:
                    weather_error = str(exc)
                    LOG.warning("Weather fallback failed for %s %s %s: %s", year, rnd, name, exc)
            elif "weather_source" not in weather:
                weather["weather_source"] = "fastf1"
                weather["weather_available_at"] = end
            source = weather.weather_source.iloc[0] if not weather.empty else "missing"
            try:
                summary = summarize_session(
                    SimpleNamespace(laps=laps, results=results, weather_data=weather, date=date),
                    year,
                    rnd,
                    name,
                )
                for attr, value in (("laps", laps), ("results", results), ("weather_data", weather)):
                    if not value.empty:
                        save_frame(value, path / f"{attr}.parquet")
                if not summary.empty:
                    save_frame(summary, path / "summary.parquet")
                status = (
                    "success" if not laps.empty and not weather.empty and not summary.empty else "missing"
                )
                if (
                    load_error is not None and not isinstance(load_error, MissingSessionError)
                ) or weather_error:
                    status = "failed"
                log_session(
                    year,
                    rnd,
                    name,
                    status,
                    not laps.empty,
                    not weather.empty,
                    source,
                    f"{load_error or ''}; {weather_error}".strip("; "),
                )
            except Exception as exc:
                log_session(year, rnd, name, "failed", not laps.empty, not weather.empty, source, str(exc))
                LOG.exception("Could not persist session")


def load_summaries() -> pd.DataFrame:
    """Load available timing checkpoints."""
    frames = [pd.read_parquet(p) for p in (DATA / "sessions").glob("*/summary.parquet")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--mode", choices=["core", "full", "sessions"], default="full")
    parser.add_argument("--rounds", type=int, nargs="+")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry only latest missing/failed CSV entries"
    )
    parser.add_argument("--session-delay", type=float, default=2, help="Seconds between sessions")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=2)
    parser.add_argument("--check-network", action="store_true")
    parser.add_argument(
        "--weather-only",
        action="store_true",
        help="Recover archive weather without requesting blocked timing feeds",
    )
    args = parser.parse_args()
    if args.check_network:
        sys.exit(0 if check_network() else 1)
    if args.session_delay < 0 or args.backoff < 0 or args.attempts < 1:
        parser.error("delay/backoff must be nonnegative and attempts must be positive")
    if args.mode != "sessions" and not args.retry_failed:
        collect_core(args.start, args.end, args.refresh)
    if args.mode != "core":
        for year in range(args.start, args.end + 1):
            collect_sessions(
                year,
                args.rounds,
                args.session_delay,
                args.attempts,
                args.backoff,
                args.retry_failed,
                args.weather_only,
            )


if __name__ == "__main__":
    main()
