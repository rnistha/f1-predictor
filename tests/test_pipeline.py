"""Behavioral safeguards against leakage and malformed race probabilities."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.feature_engineering import build_event, feature_columns, outcomes
from src.train_model import evaluate, predict_bundle


@pytest.fixture(autouse=True)
def isolated_schedule(monkeypatch):
    """Synthetic fixtures must not pick up real schedules from disk."""
    monkeypatch.setattr("src.feature_engineering.read_frame", lambda path: pd.DataFrame())


def sample() -> pd.DataFrame:
    rows = []
    for rnd in range(1, 5):
        for i, driver in enumerate(["a", "b", "c"]):
            rows.append(
                dict(
                    year=2024,
                    round=rnd,
                    driver=driver,
                    team="one" if i < 2 else "two",
                    circuit="track",
                    country="country",
                    race_date=pd.Timestamp(f"2024-0{rnd}-01", tz="UTC"),
                    birth_date="1995-01-01",
                    finish=i + 1,
                    grid=i + 1,
                    points=25 - i * 7,
                    status="Finished",
                )
            )
    return outcomes(pd.DataFrame(rows))


def test_current_and_future_outcomes_do_not_change_features():
    history = sample()
    entries = history[history["round"] == 3].copy()
    a = build_event(entries, history, pd.DataFrame())
    poisoned = history.copy()
    poisoned.loc[poisoned["round"] >= 3, ["finish", "points", "grid", "dnf", "winner", "gained"]] = 999
    entries.loc[:, ["finish", "points", "grid"]] = -999
    b = build_event(entries, poisoned, pd.DataFrame())
    assert_frame_equal(a[feature_columns("post")], b[feature_columns("post")])


def test_history_uses_only_prior_races_and_handles_rookies():
    history = sample()
    entries = history[history["round"] == 3].copy()
    entries.loc[entries.driver == "a", "driver"] = "rookie"
    result = build_event(entries, history, pd.DataFrame())
    assert result.loc[result.driver == "rookie", "experience"].iloc[0] == 0
    assert result.loc[result.driver == "b", "experience"].iloc[0] == 2
    assert result.loc[result.driver == "b", "driver_points"].iloc[0] == 36
    assert result.grid.isna().all()
    assert "grid" not in feature_columns("pre")


class Prep:
    def transform(self, frame):
        return frame.to_numpy()


class Classifier:
    def predict_proba(self, x):
        p = np.linspace(0.2, 0.8, len(x))
        return np.column_stack([1 - p, p])


class Regressor:
    def predict(self, x):
        return np.ones(len(x))


def test_probability_constraints_and_unique_ranks():
    bundle = {
        "columns": ["value"],
        "targets": {
            target: {"preprocessor": Prep(), "model": Classifier(), "calibrator": Classifier()}
            for target in ("winner", "podium", "top10", "dnf")
        },
    }
    bundle["targets"]["finish"] = {"preprocessor": Prep(), "model": Regressor()}
    data = pd.DataFrame({"race_id": ["a"] * 3 + ["b"] * 3, "value": range(6)})
    p = predict_bundle(bundle, data)
    np.testing.assert_allclose(p.groupby("race_id").winner_probability.sum(), 1)
    assert (p.winner_probability <= p.podium_probability).all()
    assert (p.podium_probability <= p.top10_probability).all()
    assert p.groupby("race_id").predicted_position.nunique().eq(3).all()


def test_evaluation_on_perfect_prediction():
    data = sample().query("round == 1").copy()
    data["race_id"] = "test"
    data["finish_score"] = data.finish
    data["predicted_position"] = data.finish
    data["championship_position"] = data.finish
    for target in ("winner", "podium", "top10", "dnf"):
        data[f"{target}_probability"] = data[target] * 0.98 + 0.01
    metrics = evaluate(data)
    assert metrics["winner_accuracy"] == 1
    assert metrics["spearman"] == 1
    assert metrics["rank_MAE"] == 0


def test_optional_timing_and_variant_allowlist():
    from src.feature_engineering import FP_FEATURES, STRATEGY_FEATURES, AVAILABILITY

    history = sample()
    entries = history[history["round"] == 3]
    partial = pd.DataFrame(
        {
            "year": [2024],
            "round": [3],
            "session": ["Practice 1"],
            "driver": ["a"],
            "session_date": [pd.Timestamp("2024-02-29T10:00Z")],
        }
    )
    features = build_event(entries, history, pd.DataFrame(), partial)
    assert len(features) == len(entries)
    assert features[AVAILABILITY].eq(0).all().all()
    assert features[FP_FEATURES].isna().all().all()
    assert not set(FP_FEATURES + STRATEGY_FEATURES + AVAILABILITY) & set(
        feature_columns("post", "results-only")
    )


def test_archive_current_race_and_late_publication_do_not_leak():
    history = sample()
    entries = history[history["round"] == 3]
    summaries = pd.DataFrame(
        [
            dict(
                year=2024,
                round=3,
                session="Practice 1",
                driver="a",
                session_date=pd.Timestamp("2024-02-29T10:00Z"),
                AirTemp=99.0,
                weather_available_at=pd.Timestamp("2024-03-07T10:00Z"),
            ),
            dict(
                year=2024,
                round=3,
                session="Race",
                driver="a",
                session_date=pd.Timestamp("2024-03-01T10:00Z"),
                AirTemp=999.0,
                pit_count=999,
                weather_available_at=pd.Timestamp("2024-03-08T10:00Z"),
            ),
            dict(
                year=2024,
                round=2,
                session="Race",
                driver="a",
                session_date=pd.Timestamp("2024-02-01T10:00Z"),
                AirTemp=25.0,
                weather_available_at=pd.Timestamp("2024-02-08T10:00Z"),
            ),
        ]
    )
    features = build_event(entries, history, pd.DataFrame(), summaries)
    assert features.practice_weather_AirTemp.isna().all()
    assert features.history_pit_count.isna().all()
    assert features.loc[features.driver == "a", "history_AirTemp"].iloc[0] == 25
