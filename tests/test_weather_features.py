"""Validate race-hour matching, wet flags and separation of hindsight inputs."""

import numpy as np
import pandas as pd
import pytest

from src import weather_features as weather
from src.feature_engineering import feature_columns


def test_race_time_is_utc_and_date_is_not_substituted():
    row = pd.Series({"year": 2024, "round": 1, "raceDate": "2024-03-02", "raceTime": "15:10:00"})
    assert weather.race_start(row) == pd.Timestamp("2024-03-02T15:10Z")


def test_precipitation_and_track_temperature():
    result = weather.summarize_weather(
        pd.DataFrame(
            {
                "AirTemp": [20.0, 22.0],
                "Humidity": [50.0, 60.0],
                "WindSpeed": [2.0, 4.0],
                "Precipitation": [0.0, 0.2],
            }
        )
    )
    assert result["weather_AirTemp"] == 21
    assert result["weather_Rainfall"] == 0.2
    assert result["weather_is_wet"] == 1
    assert np.isnan(result["weather_TrackTemp"])
    missing = weather.summarize_weather(pd.DataFrame({"Precipitation": [np.nan]}))
    assert np.isnan(missing["weather_is_wet"])


def test_observed_weather_is_excluded_from_prerace_models():
    for phase in ("pre", "post"):
        for variant in ("results-only", "enriched"):
            assert not set(weather.WEATHER_FEATURES) & set(feature_columns(phase, variant))
        assert set(weather.WEATHER_FEATURES) <= set(feature_columns(phase, "observed-weather"))
        assert "weather_TrackTemp" not in feature_columns(phase, "observed-weather")
        assert "fp_AirTemp" not in feature_columns(phase, "observed-weather")


def test_cached_matching_uses_coordinates_and_start_hour(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "DATA", tmp_path)
    monkeypatch.setattr(weather, "setup", lambda: None)
    rows = pd.DataFrame(
        [
            dict(
                year=2024,
                round=1,
                driver="a",
                raceDate="2024-03-02",
                raceTime="15:10:00",
                lat=26.0,
                long=50.0,
            ),
            dict(
                year=2024,
                round=1,
                driver="b",
                raceDate="2024-03-02",
                raceTime="15:10:00",
                lat=26.0,
                long=50.0,
            ),
        ]
    )
    rows.to_parquet(tmp_path / "race.parquet")
    (tmp_path / "weather_archive").mkdir()
    (tmp_path / "weather_archive/26.00000_50.00000_2024-03-02_2024-03-02.parquet").touch()
    calls = []

    def archived(lat, lon, start, end):
        calls.append((lat, lon, start, end))
        return pd.DataFrame(
            {
                "AirTemp": [20.0, 22.0],
                "Humidity": [50.0, 60.0],
                "WindSpeed": [2.0, 4.0],
                "Precipitation": [0.0, 0.2],
                "weather_available_at": pd.to_datetime(["2024-03-09T15:00Z", "2024-03-09T16:00Z"]),
            }
        )

    monkeypatch.setattr(weather, "archive_weather", archived)
    result = weather.collect_race_weather()
    assert calls == [(26.0, 50.0, pd.Timestamp("2024-03-02T15:00Z"), pd.Timestamp("2024-03-02T17:00Z"))]
    assert len(result) == 1
    attached = weather.attach_weather(rows)
    assert len(attached) == 2 and attached.weather_AirTemp.eq(21).all()
    # A missing day must fail coverage instead of silently matching a different date.
    rows["raceDate"] = "2024-03-03"
    rows.to_parquet(tmp_path / "race.parquet")
    with pytest.raises(ValueError, match="coverage below"):
        weather.collect_race_weather()


def test_observed_variant_refuses_unknown_future_race(monkeypatch):
    from src import predict

    monkeypatch.setattr(predict, "setup", lambda: None)
    monkeypatch.setattr(predict, "read_frame", lambda path: pd.DataFrame({"year": [2026], "round": [19]}))
    with pytest.raises(ValueError, match="retrospective"):
        predict.predict_race(2026, 20, variant="observed-weather")
