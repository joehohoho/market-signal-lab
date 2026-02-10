"""Tests for technical indicator functions in indicators.core.

Each test uses small, hand-crafted DataFrames so that expected values
can be verified by inspection or simple arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.core import (
    atr,
    bollinger_bands,
    donchian,
    ema,
    macd,
    rolling_volume_mean,
    rsi,
    sma,
    true_range,
    volatility,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _close_series(values: list[float]) -> pd.Series:
    """Build a simple float64 Series from a list of values."""
    return pd.Series(values, dtype="float64")


def _ohlcv_columns(n: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (high, low, close) series of length *n* with a gentle uptrend."""
    close = pd.Series(np.linspace(100.0, 100.0 + n, n), dtype="float64")
    high = close + 1.0
    low = close - 1.0
    return high, low, close


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------

class TestSMA:
    def test_sma_known_values(self) -> None:
        """SMA of [1,2,3,4,5] with period 3 should yield [NaN, NaN, 2, 3, 4]."""
        s = _close_series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, period=3)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_sma_period_one(self) -> None:
        """SMA with period=1 should equal the original series."""
        s = _close_series([10.0, 20.0, 30.0])
        result = sma(s, period=1)
        pd.testing.assert_series_equal(result, s, check_names=False)

    def test_sma_all_same(self) -> None:
        """SMA of a constant series should equal that constant."""
        s = _close_series([5.0] * 10)
        result = sma(s, period=4)
        for v in result.dropna():
            assert v == pytest.approx(5.0)

    def test_sma_length(self) -> None:
        """Output length should match input length."""
        s = _close_series(list(range(20)))
        result = sma(s, period=5)
        assert len(result) == len(s)

    def test_sma_nan_count(self) -> None:
        """First (period - 1) values should be NaN."""
        s = _close_series(list(range(15)))
        period = 7
        result = sma(s, period=period)
        assert result.iloc[: period - 1].isna().all()
        assert result.iloc[period - 1:].notna().all()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_ema_known_values(self) -> None:
        """EMA should converge towards recent values faster than SMA."""
        s = _close_series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ema(s, period=3)

        # First (period-1) values are NaN.
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # Third value: EMA with adjust=False seeds at value[0]=1.0,
        # alpha=2/(3+1)=0.5, so EMA[1]=0.5*2+0.5*1=1.5, EMA[2]=0.5*3+0.5*1.5=2.25.
        assert result.iloc[2] == pytest.approx(2.25, abs=0.01)
        # Subsequent values should be higher due to uptrend weighting.
        assert result.iloc[-1] > sma(s, period=3).iloc[-1]

    def test_ema_all_same(self) -> None:
        """EMA of a constant series should equal that constant."""
        s = _close_series([42.0] * 15)
        result = ema(s, period=5)
        for v in result.dropna():
            assert v == pytest.approx(42.0)

    def test_ema_length(self) -> None:
        """Output length should match input length."""
        s = _close_series(list(range(20)))
        result = ema(s, period=5)
        assert len(result) == len(s)

    def test_ema_nan_count(self) -> None:
        """First (period - 1) values should be NaN."""
        s = _close_series(list(range(12)))
        period = 4
        result = ema(s, period=period)
        assert result.iloc[: period - 1].isna().all()
        assert result.iloc[period - 1:].notna().all()


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_bounds(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """RSI values should be in [0, 100] for all non-NaN entries."""
        result = rsi(sample_ohlcv_df["close"], period=14)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_rsi_all_gains(self) -> None:
        """A monotonically increasing series should have RSI = 100."""
        s = _close_series(list(range(1, 30)))
        result = rsi(s, period=14)
        valid = result.dropna()
        # All gains, no losses -> RSI should be 100.
        for v in valid:
            assert v == pytest.approx(100.0)

    def test_rsi_all_losses(self) -> None:
        """A monotonically decreasing series should have RSI = 0."""
        s = _close_series(list(range(30, 0, -1)))
        result = rsi(s, period=14)
        valid = result.dropna()
        for v in valid:
            assert v == pytest.approx(0.0, abs=0.01)

    def test_rsi_nan_count(self) -> None:
        """First `period` values should be NaN."""
        s = _close_series(list(range(1, 25)))
        period = 14
        result = rsi(s, period=period)
        assert result.iloc[:period].isna().all()
        assert result.iloc[period:].notna().all()

    def test_rsi_known_midpoint(self) -> None:
        """Alternating up/down moves should produce RSI around 50."""
        values = [100.0]
        for i in range(1, 40):
            values.append(values[-1] + (1.0 if i % 2 == 1 else -1.0))
        s = _close_series(values)
        result = rsi(s, period=14)
        valid = result.dropna()
        # Should be roughly around 50 for equal gains/losses.
        assert 30.0 < valid.iloc[-1] < 70.0


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_macd_keys(self) -> None:
        """MACD should return a dict with 'macd', 'signal', 'histogram' keys."""
        s = _close_series(list(range(1, 50)))
        result = macd(s)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"macd", "signal", "histogram"}

    def test_macd_series_length(self) -> None:
        """All returned Series should have the same length as input."""
        s = _close_series(list(range(1, 50)))
        result = macd(s)
        for key in ("macd", "signal", "histogram"):
            assert len(result[key]) == len(s)

    def test_macd_histogram_is_difference(self) -> None:
        """Histogram should equal MACD line minus signal line."""
        s = _close_series(list(range(1, 60)))
        result = macd(s)
        expected = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(
            result["histogram"], expected, check_names=False
        )

    def test_macd_constant_series(self) -> None:
        """MACD of a constant series should be zero (after warm-up)."""
        s = _close_series([100.0] * 50)
        result = macd(s)
        valid = result["macd"].dropna()
        for v in valid:
            assert v == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# True Range & ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_true_range_positive(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """True range should be non-negative."""
        tr = true_range(
            sample_ohlcv_df["high"],
            sample_ohlcv_df["low"],
            sample_ohlcv_df["close"],
        )
        valid = tr.dropna()
        assert (valid >= 0.0).all()

    def test_atr_positive(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """ATR should always be positive (for non-constant data)."""
        result = atr(
            sample_ohlcv_df["high"],
            sample_ohlcv_df["low"],
            sample_ohlcv_df["close"],
            period=14,
        )
        valid = result.dropna()
        assert (valid > 0.0).all()

    def test_atr_nan_count(self) -> None:
        """First `period - 1` values should be NaN.

        true_range produces NaN at index 0 (no prev close), so ATR
        (ewm with min_periods=period) first becomes valid at index
        period - 1 (0-based), leaving period - 1 leading NaN values.
        """
        high, low, close = _ohlcv_columns(30)
        period = 14
        result = atr(high, low, close, period=period)
        assert result.iloc[:period - 1].isna().all()

    def test_atr_known_constant_range(self) -> None:
        """When high - low is constant and close = previous close, ATR = that range."""
        n = 30
        close = pd.Series([100.0] * n, dtype="float64")
        high = close + 2.0
        low = close - 2.0
        result = atr(high, low, close, period=5)
        valid = result.dropna()
        # True range = 4.0 every bar (high - low = 4, no gap), so ATR ~ 4.0.
        for v in valid:
            assert v == pytest.approx(4.0, abs=0.01)


# ---------------------------------------------------------------------------
# Donchian Channel
# ---------------------------------------------------------------------------

class TestDonchian:
    def test_donchian_upper_gte_lower(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Upper channel should always be >= lower channel."""
        result = donchian(
            sample_ohlcv_df["high"], sample_ohlcv_df["low"], period=20
        )
        mask = result["upper"].notna() & result["lower"].notna()
        assert (result["upper"][mask] >= result["lower"][mask]).all()

    def test_donchian_mid_is_average(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Mid should be the arithmetic mean of upper and lower."""
        result = donchian(
            sample_ohlcv_df["high"], sample_ohlcv_df["low"], period=20
        )
        expected_mid = (result["upper"] + result["lower"]) / 2.0
        pd.testing.assert_series_equal(
            result["mid"], expected_mid, check_names=False
        )

    def test_donchian_keys(self) -> None:
        """Should return 'upper', 'lower', 'mid' keys."""
        high, low, _ = _ohlcv_columns(30)
        result = donchian(high, low, period=5)
        assert set(result.keys()) == {"upper", "lower", "mid"}

    def test_donchian_known_values(self) -> None:
        """For a simple rising series, upper should be the rolling max of highs."""
        high = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype="float64")
        low = pd.Series([8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype="float64")
        result = donchian(high, low, period=3)
        # At index 2 (first non-NaN), upper = max(10,11,12) = 12
        assert result["upper"].iloc[2] == pytest.approx(12.0)
        # lower = min(8,9,10) = 8
        assert result["lower"].iloc[2] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

class TestVolatility:
    def test_volatility_non_negative(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Volatility (std dev) should be non-negative."""
        result = volatility(sample_ohlcv_df["close"], period=20)
        valid = result.dropna()
        assert (valid >= 0.0).all()

    def test_volatility_constant_series(self) -> None:
        """Volatility of a constant series should be zero."""
        s = _close_series([100.0] * 30)
        result = volatility(s, period=10)
        valid = result.dropna()
        for v in valid:
            assert v == pytest.approx(0.0, abs=1e-12)

    def test_volatility_nan_count(self) -> None:
        """First `period` values should be NaN."""
        s = _close_series(list(range(1, 30)))
        period = 10
        result = volatility(s, period=period)
        assert result.iloc[:period].isna().all()


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_bollinger_keys(self) -> None:
        """Should return 'upper', 'mid', 'lower' keys."""
        s = _close_series(list(range(1, 30)))
        result = bollinger_bands(s, period=10)
        assert set(result.keys()) == {"upper", "mid", "lower"}

    def test_bollinger_ordering(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Upper >= mid >= lower for all non-NaN rows."""
        result = bollinger_bands(sample_ohlcv_df["close"], period=20)
        mask = (
            result["upper"].notna()
            & result["mid"].notna()
            & result["lower"].notna()
        )
        assert (result["upper"][mask] >= result["mid"][mask]).all()
        assert (result["mid"][mask] >= result["lower"][mask]).all()

    def test_bollinger_mid_is_sma(self) -> None:
        """The mid band should equal the SMA."""
        s = _close_series(list(range(1, 30)))
        period = 10
        result = bollinger_bands(s, period=period)
        expected = sma(s, period=period)
        pd.testing.assert_series_equal(
            result["mid"], expected, check_names=False
        )


# ---------------------------------------------------------------------------
# Rolling Volume Mean
# ---------------------------------------------------------------------------

class TestRollingVolumeMean:
    def test_rolling_volume_known(self) -> None:
        """Rolling mean of constant volume should equal that constant."""
        vol = pd.Series([1_000_000.0] * 25, dtype="float64")
        result = rolling_volume_mean(vol, period=10)
        valid = result.dropna()
        for v in valid:
            assert v == pytest.approx(1_000_000.0)

    def test_rolling_volume_length(self) -> None:
        """Output length should match input length."""
        vol = pd.Series(range(30), dtype="float64")
        result = rolling_volume_mean(vol, period=5)
        assert len(result) == 30
