"""Failure recovery, network fallback and data provenance contracts."""

from types import SimpleNamespace
import logging

import numpy as np
import pandas as pd
import requests
from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError

from src import collection_support as support
from src import data_collection as collect


def test_exponential_backoff_and_rate_pause(monkeypatch):
    sleeps = []
    monkeypatch.setattr(support, "pause", sleeps.append)
    values = iter([OSError("temporary"), OSError("temporary"), 42])

    def action():
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    assert support.network_retry(action, attempts=3, backoff=2) == 42
    assert sleeps == [2, 4]
    sleeps.clear()
    values = iter([RateLimitExceededError("500 calls/h"), 42])
    assert support.network_retry(action) == 42
    assert sleeps == [3601]


def test_http_retry_after_and_swallowed_rate_limit():
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "17"
    error = requests.HTTPError(response=response)
    assert support.rate_wait(error, 3601) == 17
    capture = support.RateLimitCapture()
    record = logging.LogRecord(
        "fastf1.core", logging.ERROR, "", 0, "soft failure", (), (type(error), error, None)
    )
    capture.emit(record)
    assert capture.error is error


def test_safe_missing_property():
    class Session:
        @property
        def laps(self):
            raise DataNotLoadedError("not loaded")

    assert support.safe_frame(Session(), "laps").empty


def test_archive_window_units_cache_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(support, "DATA", tmp_path)
    calls = []

    def get(url, params, timeout):
        calls.append(params)
        payload = {"hourly": {"time": ["2024-03-02T14:00", "2024-03-02T15:00", "2024-03-02T16:00"]}}
        for key in support.WEATHER_FIELDS:
            payload["hourly"][key] = [1.0, 2.0, 3.0]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    monkeypatch.setattr(support.requests, "get", get)
    start, end = pd.Timestamp("2024-03-02T14:30Z"), pd.Timestamp("2024-03-02T16:00Z")
    data = support.archive_weather(26, 50, start, end)
    assert len(data) == 1 and data.AirTemp.iloc[0] == 2
    assert data.TrackTemp.isna().all()
    assert data.weather_source.iloc[0] == "open-meteo-archive"
    assert data.weather_available_at.iloc[0] > end
    assert calls[0]["timezone"] == "UTC" and calls[0]["wind_speed_unit"] == "ms"
    support.archive_weather(26, 50, start, end)
    assert len(calls) == 1


def test_missing_laps_retains_driver_rows():
    session = SimpleNamespace(
        laps=pd.DataFrame(),
        weather_data=pd.DataFrame(),
        results=pd.DataFrame({"DriverId": ["a", "b"], "DriverNumber": ["1", "2"]}),
        date=pd.Timestamp("2024-03-01T10:00Z"),
    )
    result = collect.summarize_session(session, 2024, 1, "Practice 1")
    assert list(result.driver) == ["a", "b"]
    assert result.best_lap.isna().all()
    assert not result.has_practice_data.any()


def test_collector_logs_partial_and_retry_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "DATA", tmp_path)
    monkeypatch.setattr(collect, "setup", lambda: None)
    monkeypatch.setattr(collect, "pause", lambda _: None)
    (tmp_path / "raw").mkdir()
    pd.DataFrame(
        [
            {
                "RoundNumber": 1,
                "EventName": "Race",
                "Session1": "Practice 1",
                "Session1DateUtc": pd.Timestamp("2024-03-01T10:00Z"),
            }
        ]
    ).to_parquet(tmp_path / "raw/2024_schedule.parquet")
    pd.DataFrame(
        [dict(year=2024, round=1, driver="a", team="team", number=1, lat=26.0, long=50.0)]
    ).to_parquet(tmp_path / "quali.parquet")
    calls = []
    session = SimpleNamespace(
        laps=pd.DataFrame(),
        results=pd.DataFrame(),
        weather_data=pd.DataFrame(),
        load=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(collect.fastf1, "get_session", lambda *args: session)
    weather = pd.DataFrame(
        dict(
            AirTemp=[20.0],
            Humidity=[50.0],
            WindSpeed=[2.0],
            WindDirection=[90.0],
            Rainfall=[False],
            weather_source=["open-meteo-archive"],
            weather_available_at=[pd.Timestamp("2024-03-08T12:00Z")],
        )
    )
    monkeypatch.setattr(collect, "archive_weather", lambda *args, **kwargs: weather)
    collect.collect_sessions(2024, attempts=1, delay=0)
    log = pd.read_csv(tmp_path / "collection_log.csv")
    assert log.status.iloc[-1] == "missing"
    assert log.has_weather.iloc[-1]
    assert calls == [dict(laps=True, weather=True, telemetry=False, messages=False)]
    summary = pd.read_parquet(tmp_path / "sessions/2024_1_Practice_1/summary.parquet")
    assert len(summary) == 1 and np.isnan(summary.best_lap.iloc[0])
    collect.collect_sessions(2024, attempts=1, delay=0, retry_failed=True)
    assert len(calls) == 2  # Partial checkpoint must not suppress retries.
    collect.log_session(2024, 1, "Practice 1", "success", True, True, "fastf1")
    collect.collect_sessions(2024, attempts=1, delay=0, retry_failed=True)
    assert len(calls) == 2


def test_network_diagnostics_explain_block(monkeypatch, capsys):
    def blocked(*args, **kwargs):
        raise requests.ConnectionError("DNS blocked")

    monkeypatch.setattr(support.requests, "get", blocked)
    assert not support.check_network()
    assert "BLOCKED/UNREACHABLE" in capsys.readouterr().out


def test_retry_preserves_cached_archive_provenance(tmp_path, monkeypatch):
    """A saved archive frame must not be relabeled as native pre-race weather."""
    monkeypatch.setattr(collect, "DATA", tmp_path)
    monkeypatch.setattr(collect, "setup", lambda: None)
    monkeypatch.setattr(collect, "pause", lambda _: None)
    (tmp_path / "raw").mkdir()
    pd.DataFrame(
        [
            {
                "RoundNumber": 1,
                "EventName": "Race",
                "Session1": "Practice 1",
                "Session1DateUtc": pd.Timestamp("2024-03-01T10:00Z"),
            }
        ]
    ).to_parquet(tmp_path / "raw/2024_schedule.parquet")
    pd.DataFrame(
        [dict(year=2024, round=1, driver="a", team="team", number=1, lat=26.0, long=50.0)]
    ).to_parquet(tmp_path / "quali.parquet")
    folder = tmp_path / "sessions/2024_1_Practice_1"
    folder.mkdir(parents=True)
    weather = pd.DataFrame(
        {
            "AirTemp": [20.0],
            "Humidity": [50.0],
            "WindSpeed": [2.0],
            "weather_source": ["open-meteo-archive"],
            "weather_available_at": [pd.Timestamp("2024-03-08T12:00Z")],
        }
    )
    weather.to_parquet(folder / "weather_data.parquet")
    collect.collect_sessions(2024, attempts=1, delay=0, weather_only=True)
    saved = pd.read_parquet(folder / "summary.parquet")
    assert saved.weather_source.iloc[0] == "open-meteo-archive"
    assert saved.weather_available_at.iloc[0] == weather.weather_available_at.iloc[0]
