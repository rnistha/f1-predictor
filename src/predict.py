"""Pre-race prediction with calibration-cutoff enforcement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.data_collection import load_summaries
from src.feature_engineering import build_event, outcomes
from src.train_model import predict_bundle
from src.utils import DATA, MODELS, read_frame, setup


def race_entries(year: int, rnd: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve entrants from qualifying/practice; historical results supply identity only."""
    races = read_frame(DATA / "race.parquet")
    quali = read_frame(DATA / "quali.parquet")
    q = quali[(quali.year == year) & (quali["round"] == rnd)] if not quali.empty else quali
    past = races[(races.year == year) & (races["round"] == rnd)] if not races.empty else races
    identity = [
        "year",
        "round",
        "driver",
        "team",
        "code",
        "givenName",
        "familyName",
        "birth_date",
        "circuit",
        "country",
        "race",
        "race_date",
    ]
    source = q if not q.empty else past
    if not source.empty:
        return source[[c for c in identity if c in source]].copy(), q
    summaries = load_summaries()
    fp = (
        summaries[
            (summaries.year == year)
            & (summaries["round"] == rnd)
            & summaries.session.str.startswith("Practice")
        ]
        if not summaries.empty
        else summaries
    )
    if fp.empty:
        raise ValueError(
            "No confirmed entrants for this event. Download practice/qualifying data first; future grids are not guessed."
        )
    schedule = read_frame(DATA / "raw" / f"{year}_schedule.parquet")
    event = schedule[schedule.RoundNumber == rnd].iloc[0]
    entries = (
        fp.sort_values("session_date")
        .drop_duplicates("driver", keep="last")[["year", "round", "driver", "code", "team_name"]]
        .copy()
    )
    # Map provider team names via the last known driver identity only if available.
    latest = races.sort_values(["year", "round"]).drop_duplicates("driver", keep="last").set_index("driver")
    entries["team"] = entries.team_name
    for col in ("birth_date",):
        entries[col] = entries.driver.map(latest[col]) if col in latest else np.nan
    historical_track = races[races.race == event.EventName]
    entries["circuit"] = historical_track.circuit.iloc[-1] if not historical_track.empty else event.Location
    entries["country"] = event.Country
    entries["race"] = event.EventName
    entries["race_date"] = pd.to_datetime(event.EventDate, utc=True)
    return entries, q


def predict_race(
    year: int, round: int, model: str = "best", phase: str = "auto", variant: str = "results-only"
) -> pd.DataFrame:
    """Return driver probabilities and finishing order using only pre-race features.

    Refuses historical races at/before calibration cutoff. Choose a held-out race,
    or train a separate earlier-cutoff artifact for honest historical backtesting.
    """
    setup()
    if variant == "observed-weather":
        races = read_frame(DATA / "race.parquet")
        if races.empty or races[(races.year == year) & (races["round"] == round)].empty:
            raise ValueError(
                "Observed-weather is retrospective and requires a completed race. Future races need a timestamped forecast, which is not implemented."
            )
    entries, q = race_entries(year, round)
    selected_phase = ("post" if not q.empty else "pre") if phase == "auto" else phase
    if selected_phase == "post" and q.empty:
        selected_phase = "pre"
    if variant not in ("results-only", "enriched", "observed-weather"):
        raise ValueError("variant must be results-only, enriched or observed-weather")
    path = MODELS / f"{variant}_{selected_phase}_{model}.joblib"
    if not path.exists():
        raise ValueError("Trained model missing. Run python src/train_model.py first.")
    bundle = joblib.load(path)
    if entries.race_date.min() <= pd.Timestamp(bundle["trained_through"]):
        raise ValueError(
            f"This model uses calibration data through {bundle['trained_through']}. Select {bundle['test_year']} or later for an honest prediction."
        )
    history = outcomes(read_frame(DATA / "race.parquet"))
    features = build_event(
        entries,
        history,
        q if selected_phase == "post" else pd.DataFrame(),
        load_summaries(),
        read_frame(DATA / "sprint.parquet"),
    )
    if variant == "observed-weather":
        from src.weather_features import WEATHER_FEATURES

        if features[WEATHER_FEATURES].isna().any().any():
            raise ValueError("Observed weather is incomplete for this race; backfill archive weather first.")
    features["race_id"] = f"{year}_{round:02d}"
    result = predict_bundle(bundle, features)
    result["phase"] = selected_phase
    result["model_variant"] = bundle["variant"]
    result["prediction_scope"] = (
        "retrospective observed weather" if variant == "observed-weather" else "pre-race inputs"
    )
    result["grid_source"] = "qualifying proxy" if selected_phase == "post" else "unavailable"
    result = result.rename(columns={"winner_probability": "win_probability"})
    result.attrs["model"] = model
    result.attrs["trained_through"] = bundle["trained_through"]
    return result.sort_values("predicted_position").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--round", type=int)
    parser.add_argument("--model", choices=["best", "sklearn", "xgboost", "lightgbm"], default="best")
    parser.add_argument("--phase", choices=["auto", "pre", "post"], default="auto")
    parser.add_argument(
        "--variant", choices=["results-only", "enriched", "observed-weather"], default="results-only"
    )
    args = parser.parse_args()
    races = read_frame(DATA / "race.parquet")
    rnd = args.round or int(races[races.year == args.year]["round"].max())
    pred = predict_race(args.year, rnd, args.model, args.phase, args.variant)
    cols = [
        "predicted_position",
        "driver",
        "team",
        "grid",
        "win_probability",
        "podium_probability",
        "top10_probability",
        "dnf_probability",
    ]
    print(pred[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    pred.to_csv(DATA / f"prediction_{args.year}_{rnd}_{pred.phase.iloc[0]}_{args.variant}.csv", index=False)
