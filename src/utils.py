"""Shared paths, persistence, logging and retry helpers."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import fastf1
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
T = TypeVar("T")
LOG = logging.getLogger("f1_predictor")


def setup() -> None:
    """Create local directories and enable the persistent FastF1 cache."""
    for path in (DATA, DATA / "raw", DATA / "sessions", MODELS, ROOT / "cache"):
        path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(ROOT / "cache"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def retry(fn: Callable[[], T], attempts: int = 3) -> T:
    """Retry transient failures with bounded exponential backoff."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            LOG.warning("Request failed; retrying", exc_info=True)
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace parquet so interrupted downloads remain resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    pd.DataFrame(frame).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_frame(path: Path) -> pd.DataFrame:
    """Read an optional dataset without inventing missing observations."""
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def seconds(values: pd.Series) -> pd.Series:
    """Convert provider timedeltas to floating-point seconds."""
    return pd.to_timedelta(values, errors="coerce").dt.total_seconds()
