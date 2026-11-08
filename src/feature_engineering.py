"""Build explicitly allowlisted, race-time-safe features from prior events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_collection import load_summaries
from src.metadata import circuit_type, power_unit
from src.utils import DATA, read_frame, save_frame
from src.weather_features import WEATHER_FEATURES, attach_weather

CATEGORICAL = ["driver", "team", "circuit", "country", "era", "circuit_type", "power_unit"]
FORM_FIELDS = ["finish", "points", "grid", "gained", "dnf"]
NUMERIC = [
    "year",
    "round",
    "experience",
    "age",
    "driver_points",
    "team_points",
    "championship_position",
    "team_championship_position",
    "circuit_avg_finish",
    "circuit_wins",
    "circuit_podiums",
    "team_circuit_avg_finish",
    "pole_conversion",
    "overtaking_proxy",
    "team_reliability",
    "team_trend",
    "teammate_form_gap",
    "team_mechanical_rate",
]
NUMERIC += [f"form_{field}_{window}" for window in (3, 5) for field in FORM_FIELDS]
FP_FEATURES = [
    "fp_best_gap",
    "fp_long_run_rank",
    "fp_sector1",
    "fp_sector2",
    "fp_sector3",
]
PRACTICE_WEATHER = [
    "practice_weather_" + c for c in ("AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall")
]
QUA_FEATURES = ["grid", "quali_position", "quali_gap", "teammate_quali_gap"]
STRATEGY_FEATURES = [
    "history_" + c
    for c in (
        "pit_count",
        "pit_duration",
        "stint_length",
        "tyre_age",
        "sc_periods",
        "vsc_periods",
        "red_periods",
    )
]


AVAILABILITY = ["has_practice_data", "has_weather_data", "has_strategy_data"]
HISTORY_WEATHER = ["history_" + x for x in ("AirTemp", "Humidity", "WindSpeed", "Precipitation")]


def feature_columns(phase: str, variant: str = "enriched") -> list[str]:
    """Exclude outcomes; allow current-race weather only in the hindsight variant."""
    if variant not in ("results-only", "enriched", "observed-weather"):
        raise ValueError(f"Unknown variant: {variant}")
    base = CATEGORICAL + NUMERIC + (QUA_FEATURES if phase == "post" else [])
    timing = FP_FEATURES + PRACTICE_WEATHER + STRATEGY_FEATURES + HISTORY_WEATHER + AVAILABILITY
    return (
        base
        + (timing if variant != "results-only" else [])
        + (WEATHER_FEATURES if variant == "observed-weather" else [])
    )


def outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add labels and historical-only outcome fields."""
    frame = frame.copy()
    status = frame.status.fillna("").astype(str)
    frame["dnf"] = (~(status.eq("Finished") | status.str.match(r"\+\d+ Laps?"))).astype(int)
    frame["gained"] = frame.grid.where(frame.grid > 0) - frame.finish
    frame["winner"] = (frame.finish == 1).astype(int)
    frame["podium"] = (frame.finish <= 3).astype(int)
    frame["top10"] = (frame.finish <= 10).astype(int)
    frame["mechanical"] = status.str.contains(
        "Engine|Gearbox|Transmission|Hydraulics|Electrical|Suspension|Brakes|Power Unit|Turbo|Fuel|Oil|Water|Overheating|Mechanical",
        case=False,
    ).astype(int)
    return frame


def add_qualifying(entries: pd.DataFrame, qualifying: pd.DataFrame) -> pd.DataFrame:
    """Use qualifying position as a grid proxy; never infer it from race results."""
    entries = entries.drop(columns=QUA_FEATURES, errors="ignore").copy()
    if qualifying.empty:
        for col in QUA_FEATURES:
            entries[col] = np.nan
        return entries
    q = qualifying.copy()
    # Compare Q1 to Q1: taking min(Q1,Q2,Q3) creates session evolution bias.
    q["quali_gap"] = q.q1 - q.q1.min()
    q["teammate_quali_gap"] = q.q1 - q.groupby("team").q1.transform("min")
    q["grid"] = q.quali_position
    return entries.merge(q[["driver"] + QUA_FEATURES], on="driver", how="left", validate="one_to_one")


def build_event(
    entries: pd.DataFrame,
    history: pd.DataFrame,
    qualifying: pd.DataFrame,
    summaries: pd.DataFrame | None = None,
    sprints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a race atomically; no driver sees any outcome from this race.

    History is filtered inside this function even if callers supply the full dataset.
    Official historical classification remains available to create labels elsewhere.
    """
    entries = entries.sort_values("driver").reset_index(drop=True)
    year, rnd = int(entries.year.iloc[0]), int(entries["round"].iloc[0])
    prior = history[(history.year < year) | ((history.year == year) & (history["round"] < rnd))].copy()
    if not prior.empty and "dnf" not in prior:
        prior = outcomes(prior)
    prior = prior.sort_values(["year", "round"])
    current = prior[prior.year == year]
    circuit = entries.circuit.iloc[0]
    track = prior[prior.circuit == circuit]
    dp = current.groupby("driver").points.sum()
    tp = current.groupby("team").points.sum()
    if sprints is not None and not sprints.empty:
        sp = sprints[(sprints.year == year) & (sprints["round"] < rnd)]
        dp = dp.add(sp.groupby("driver").points.sum(), fill_value=0)
        tp = tp.add(sp.groupby("team").points.sum(), fill_value=0)
    # Rank only current entrants; ties share a rank (official countback unavailable here).
    driver_scores = entries.driver.map(dp).fillna(0)
    team_scores = entries.team.drop_duplicates().map(tp).fillna(0)
    dranks = dict(zip(entries.driver, driver_scores.rank(method="min", ascending=False), strict=True))
    tranks = dict(
        zip(entries.team.drop_duplicates(), team_scores.rank(method="min", ascending=False), strict=True)
    )
    rows = []
    for _, entry in entries.iterrows():
        row = entry.to_dict()
        dh = prior[prior.driver == entry.driver]
        th = prior[prior.team == entry.team]
        ch = track[track.driver == entry.driver]
        tch = track[track.team == entry.team]
        row.update(
            experience=len(dh),
            driver_points=float(dp.get(entry.driver, 0)),
            team_points=float(tp.get(entry.team, 0)),
            championship_position=dranks[entry.driver],
            team_championship_position=tranks[entry.team],
            circuit_avg_finish=ch.finish.mean(),
            circuit_wins=int((ch.finish == 1).sum()),
            circuit_podiums=int((ch.finish <= 3).sum()),
            team_circuit_avg_finish=tch.finish.mean(),
            pole_conversion=track.loc[track.grid == 1, "winner"].mean() if not track.empty else np.nan,
            overtaking_proxy=track.gained.abs().mean() if not track.empty else np.nan,
            team_reliability=1 - th.tail(10).dnf.mean() if not th.empty else np.nan,
            team_mechanical_rate=th.tail(10).mechanical.mean() if "mechanical" in th else np.nan,
            team_trend=th.tail(6).finish.mean() - th.tail(10).finish.mean(),
            circuit_type=circuit_type(str(circuit)),
            power_unit=power_unit(str(entry.team), year),
            era="2026" if year >= 2026 else "2022" if year >= 2022 else "2018",
        )
        birth = pd.to_datetime(entry.get("birth_date"), utc=True, errors="coerce")
        row["age"] = (
            (pd.to_datetime(entry.race_date, utc=True) - birth).days / 365.25 if pd.notna(birth) else np.nan
        )
        for window in (3, 5):
            for field in FORM_FIELDS:
                row[f"form_{field}_{window}"] = dh.tail(window)[field].mean() if field in dh else np.nan
        rows.append(row)
    features = pd.DataFrame(rows)
    features["teammate_form_gap"] = features.form_finish_5 - features.groupby("team").form_finish_5.transform(
        "mean"
    )
    features = add_qualifying(features, qualifying)
    for col in FP_FEATURES + PRACTICE_WEATHER + STRATEGY_FEATURES + HISTORY_WEATHER:
        features[col] = np.nan
    for col in AVAILABILITY:
        features[col] = 0
    if summaries is not None and not summaries.empty:
        summaries = summaries.copy()
        if not {"year", "round", "session", "driver", "session_date"}.issubset(summaries):
            summaries = pd.DataFrame(columns=["year", "round", "session", "driver", "session_date"])
        for col in [
            "best_lap",
            "long_run",
            "sector1",
            "sector2",
            "sector3",
            "AirTemp",
            "TrackTemp",
            "Humidity",
            "Pressure",
            "WindSpeed",
            "Rainfall",
            "Precipitation",
        ]:
            if col not in summaries:
                summaries[col] = np.nan
        summaries["session"] = summaries.session.fillna("").astype(str)
        observed_end = pd.to_datetime(summaries.session_date, utc=True) + pd.Timedelta(hours=2)
        if "weather_available_at" not in summaries:
            summaries["weather_available_at"] = observed_end
        summaries["weather_available_at"] = pd.to_datetime(summaries.weather_available_at, utc=True).fillna(
            observed_end
        )
        historical_sessions = summaries[
            (summaries.session == "Race")
            & ((summaries.year < year) | ((summaries.year == year) & (summaries["round"] < rnd)))
        ]
        for name in STRATEGY_FEATURES:
            original = name.removeprefix("history_")
            if original in historical_sessions:
                agg = (
                    historical_sessions.sort_values(["year", "round"])
                    .groupby("driver")
                    .tail(5)
                    .groupby("driver")[original]
                    .mean()
                )
                features[name] = features.driver.map(agg)
        features["has_strategy_data"] = features[STRATEGY_FEATURES].notna().any(axis=1).astype(int)
        prior_weather = historical_sessions[
            historical_sessions.weather_available_at < pd.to_datetime(entries.race_date.iloc[0], utc=True)
        ]
        for col in HISTORY_WEATHER:
            original = col.removeprefix("history_")
            agg = (
                prior_weather.sort_values(["year", "round"])
                .groupby("driver")
                .tail(5)
                .groupby("driver")[original]
                .mean()
            )
            features[col] = features.driver.map(agg)
        fp = summaries[
            (summaries.year == year)
            & (summaries["round"] == rnd)
            & summaries.session.str.startswith("Practice")
        ].copy()
        cutoff = pd.to_datetime(entries.race_date.iloc[0], utc=True)
        found_cutoff = False
        schedule = read_frame(DATA / "raw" / f"{year}_schedule.parquet")
        if not schedule.empty:
            event = schedule[schedule.RoundNumber == rnd]
            if not event.empty:
                event = event.iloc[0]
                for i in range(1, 6):
                    if event.get(f"Session{i}") == "Qualifying":
                        cutoff = pd.to_datetime(event.get(f"Session{i}DateUtc"), utc=True)
                        if pd.notna(cutoff):
                            found_cutoff = True
                            fp = fp[
                                pd.to_datetime(fp.session_date, utc=True) + pd.Timedelta(hours=2) <= cutoff
                            ]
        if not found_cutoff:
            # A missing timetable cannot prove that FP2/3 preceded qualifying.
            fp = fp[fp.session == "Practice 1"]
        if not fp.empty:
            # Archive data published after the cutoff cannot masquerade as a forecast.
            weather_late = fp.weather_available_at > cutoff
            fp.loc[
                weather_late, ["AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall"]
            ] = np.nan
            # Relative pace within session, rather than mixing absolute track conditions.
            fp["fp_best_gap"] = fp.best_lap - fp.groupby("session").best_lap.transform("min")
            fp["fp_long_run_rank"] = fp.groupby("session").long_run.rank()
            for name in FP_FEATURES[2:]:
                fp[name] = fp.get(name.removeprefix("fp_"), np.nan)
            for name in PRACTICE_WEATHER:
                fp[name] = fp.get(name.removeprefix("practice_weather_"), np.nan)
            agg = fp.groupby("driver")[FP_FEATURES + PRACTICE_WEATHER].mean()
            for col in FP_FEATURES + PRACTICE_WEATHER:
                features[col] = features.driver.map(agg[col])
    # Published weather of earlier races is eligible for the deployable enriched model.
    archive = read_frame(DATA / "race_weather.parquet")
    if not archive.empty and "weather_available_at" in archive:
        known = archive[
            pd.to_datetime(archive.weather_available_at, utc=True)
            < pd.to_datetime(entries.race_date.iloc[0], utc=True)
        ]
        known = prior[["year", "round", "driver"]].merge(
            known, on=["year", "round"], how="inner", validate="many_to_one"
        )
        if not known.empty:
            tail = known.sort_values(["year", "round"]).groupby("driver").tail(5)
            for col in HISTORY_WEATHER:
                source = "weather_" + col.removeprefix("history_")
                if source in tail:
                    features[col] = features.driver.map(tail.groupby("driver")[source].mean()).combine_first(
                        features[col]
                    )
    features["has_practice_data"] = (
        features[["fp_best_gap", "fp_long_run_rank"]].notna().any(axis=1).astype(int)
    )
    features["has_weather_data"] = features[PRACTICE_WEATHER].notna().any(axis=1).astype(int)
    for col in feature_columns("post"):
        if col not in features:
            features[col] = np.nan
    return attach_weather(features)


def build_dataset() -> pd.DataFrame:
    """Build historical features with event-level chronology and separate labels."""
    races = read_frame(DATA / "race.parquet")
    if races.empty:
        raise ValueError("No race data. Run data_collection.py --mode core first.")
    races = outcomes(races)
    races["actual_grid"] = races.grid
    quali = read_frame(DATA / "quali.parquet")
    sprints = read_frame(DATA / "sprint.parquet")
    summaries = load_summaries()
    frames = []
    for (year, rnd), event in races.groupby(["year", "round"], sort=True):
        q = quali[(quali.year == year) & (quali["round"] == rnd)] if not quali.empty else quali
        built = build_event(event, races, q, summaries, sprints)
        built["race_id"] = f"{year}_{rnd:02d}"
        frames.append(built)
    result = pd.concat(frames, ignore_index=True)
    save_frame(result, DATA / "features.parquet")
    return result
