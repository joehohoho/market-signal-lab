"""Shared test fixtures for Market Signal Lab.

Provides reusable OHLCV DataFrames and temporary storage directories
used across all test modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_ohlcv_df() -> pd.DataFrame:
    """A 100-row OHLCV DataFrame with realistic daily price data.

    The series starts at $100 and follows a random walk with moderate
    volatility.  Volume oscillates around 1 000 000 shares.
    """
    rng = np.random.default_rng(42)
    n = 100

    # Build a realistic close series via cumulative log-returns.
    log_returns = rng.normal(loc=0.0005, scale=0.015, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    # Derive OHLC from close with small intraday ranges.
    high = close * (1.0 + rng.uniform(0.001, 0.02, size=n))
    low = close * (1.0 - rng.uniform(0.001, 0.02, size=n))
    open_ = low + rng.uniform(0.3, 0.7, size=n) * (high - low)
    volume = rng.integers(500_000, 2_000_000, size=n).astype(float)

    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture()
def small_ohlcv_df() -> pd.DataFrame:
    """A 20-row OHLCV DataFrame for simple / edge-case tests.

    Prices follow a gentle uptrend from $50 to ~$60 with deterministic
    values so that hand-calculated expected results are straightforward.
    """
    n = 20
    close = np.linspace(50.0, 60.0, n)
    high = close + 1.0
    low = close - 1.0
    open_ = close - 0.5
    volume = np.full(n, 1_000_000.0)

    timestamps = pd.date_range("2024-06-01", periods=n, freq="D")

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture()
def tmp_storage_dir(tmp_path: Path) -> Path:
    """Return a temporary directory suitable for parquet / duckdb files.

    The directory is unique per test invocation and is automatically
    cleaned up by pytest.
    """
    storage = tmp_path / "storage"
    storage.mkdir()
    return storage
