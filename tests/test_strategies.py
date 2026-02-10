"""Tests for the strategy system.

Because the concrete strategy files (sma_crossover.py, rsi_mean_reversion.py,
donchian_breakout.py) are currently stubs, these tests exercise the base
types directly -- constructing SignalResult objects by hand and verifying the
contracts that any strategy implementation must satisfy.

When the concrete strategies are fleshed out, replace the stub-level tests
with full integration tests that call ``strategy.evaluate()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.base import Signal, SignalResult, Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Build an *n*-row OHLCV DataFrame with a random-walk close."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n)
    close = start_price * np.exp(np.cumsum(returns))
    high = close * (1.0 + rng.uniform(0.002, 0.015, size=n))
    low = close * (1.0 - rng.uniform(0.002, 0.015, size=n))
    open_ = low + rng.uniform(0.3, 0.7, size=n) * (high - low)
    volume = rng.integers(500_000, 2_000_000, size=n).astype(float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class _DummyStrategy(Strategy):
    """Minimal concrete strategy for testing the base class contract."""

    @property
    def name(self) -> str:
        return "dummy"

    def compute(self, df, asset, timeframe, params):
        """Return one HOLD signal per bar with strength 0.5."""
        return [
            SignalResult(signal=Signal.HOLD, strength=0.5, explanation={"reason": "test"})
            for _ in range(len(df))
        ]


# ---------------------------------------------------------------------------
# Signal enum
# ---------------------------------------------------------------------------

class TestSignalEnum:
    def test_signal_values(self) -> None:
        """Signal should have exactly BUY, SELL, HOLD members."""
        assert set(Signal) == {Signal.BUY, Signal.SELL, Signal.HOLD}

    def test_signal_string_values(self) -> None:
        assert Signal.BUY.value == "BUY"
        assert Signal.SELL.value == "SELL"
        assert Signal.HOLD.value == "HOLD"


# ---------------------------------------------------------------------------
# SignalResult
# ---------------------------------------------------------------------------

class TestSignalResult:
    def test_signal_result_creation(self) -> None:
        """SignalResult should store signal, strength, and explanation."""
        sr = SignalResult(signal=Signal.BUY, strength=0.8, explanation={"rsi": 25})
        assert sr.signal == Signal.BUY
        assert sr.strength == pytest.approx(0.8)
        assert sr.explanation == {"rsi": 25}

    def test_signal_result_defaults(self) -> None:
        """Default strength should be 0.0 and explanation an empty dict."""
        sr = SignalResult(signal=Signal.HOLD)
        assert sr.strength == 0.0
        assert sr.explanation == {}

    def test_signal_result_immutable(self) -> None:
        """SignalResult is frozen -- attribute assignment should raise."""
        sr = SignalResult(signal=Signal.SELL, strength=0.6)
        with pytest.raises(AttributeError):
            sr.signal = Signal.BUY  # type: ignore[misc]

    def test_signal_strength_bounds(self) -> None:
        """Strength values of 0.0 and 1.0 should both be valid."""
        low = SignalResult(signal=Signal.HOLD, strength=0.0)
        high = SignalResult(signal=Signal.BUY, strength=1.0)
        assert 0.0 <= low.strength <= 1.0
        assert 0.0 <= high.strength <= 1.0


# ---------------------------------------------------------------------------
# Base Strategy abstract contract
# ---------------------------------------------------------------------------

class TestBaseStrategy:
    def test_cannot_instantiate_abstract(self) -> None:
        """Instantiating Strategy directly should raise TypeError."""
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]

    def test_dummy_strategy_name(self) -> None:
        strat = _DummyStrategy()
        assert strat.name == "dummy"

    def test_dummy_strategy_compute(self) -> None:
        df = _make_ohlcv(50)
        results = _DummyStrategy().compute(df, asset="TEST", timeframe="1d", params={})
        assert isinstance(results, list)
        assert len(results) == len(df)
        for result in results:
            assert isinstance(result, SignalResult)
            assert result.signal in set(Signal)
            assert 0.0 <= result.strength <= 1.0


# ---------------------------------------------------------------------------
# Simulated strategy signal generation
#
# These tests validate signal-generation patterns that the concrete
# strategies should follow.  They produce SignalResult objects directly
# so the test suite works even while the strategy files are stubs.
# ---------------------------------------------------------------------------

class TestSMACrossoverSignals:
    """Tests that simulate SMA crossover signal generation logic."""

    @staticmethod
    def _simulate_sma_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> list[SignalResult]:
        """Simulate the SMA crossover logic using indicators.core."""
        from indicators.core import sma as _sma

        fast_ma = _sma(df["close"], period=fast)
        slow_ma = _sma(df["close"], period=slow)

        signals: list[SignalResult] = []
        for i in range(1, len(df)):
            if pd.isna(fast_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]):
                signals.append(SignalResult(signal=Signal.HOLD, strength=0.0))
                continue
            if pd.isna(fast_ma.iloc[i - 1]) or pd.isna(slow_ma.iloc[i - 1]):
                signals.append(SignalResult(signal=Signal.HOLD, strength=0.0))
                continue

            prev_diff = fast_ma.iloc[i - 1] - slow_ma.iloc[i - 1]
            curr_diff = fast_ma.iloc[i] - slow_ma.iloc[i]

            if prev_diff <= 0 and curr_diff > 0:
                strength = min(abs(curr_diff) / df["close"].iloc[i] * 100, 1.0)
                signals.append(SignalResult(signal=Signal.BUY, strength=strength))
            elif prev_diff >= 0 and curr_diff < 0:
                strength = min(abs(curr_diff) / df["close"].iloc[i] * 100, 1.0)
                signals.append(SignalResult(signal=Signal.SELL, strength=strength))
            else:
                signals.append(SignalResult(signal=Signal.HOLD, strength=0.0))
        return signals

    def test_generates_signal_results(self) -> None:
        """Simulated SMA crossover should produce a list of SignalResult."""
        df = _make_ohlcv(120, seed=7)
        signals = self._simulate_sma_crossover(df)
        assert len(signals) > 0
        assert all(isinstance(s, SignalResult) for s in signals)

    def test_signals_are_valid_enum(self) -> None:
        """Every signal should be a valid Signal enum member."""
        df = _make_ohlcv(120, seed=7)
        signals = self._simulate_sma_crossover(df)
        valid_signals = {Signal.BUY, Signal.SELL, Signal.HOLD}
        assert all(s.signal in valid_signals for s in signals)

    def test_strength_bounds(self) -> None:
        """Signal strength should be in [0.0, 1.0]."""
        df = _make_ohlcv(120, seed=7)
        signals = self._simulate_sma_crossover(df)
        assert all(0.0 <= s.strength <= 1.0 for s in signals)


class TestRSIMeanReversionSignals:
    """Tests that simulate RSI mean-reversion signal generation logic."""

    @staticmethod
    def _simulate_rsi_signal(df: pd.DataFrame) -> list[SignalResult]:
        """Simulate RSI mean-reversion: BUY when RSI < 30, SELL when RSI > 70."""
        from indicators.core import rsi as _rsi

        rsi_vals = _rsi(df["close"], period=14)
        signals: list[SignalResult] = []
        for i in range(len(df)):
            val = rsi_vals.iloc[i]
            if pd.isna(val):
                signals.append(SignalResult(signal=Signal.HOLD, strength=0.0))
            elif val < 30:
                strength = min((30.0 - val) / 30.0, 1.0)
                signals.append(SignalResult(signal=Signal.BUY, strength=strength))
            elif val > 70:
                strength = min((val - 70.0) / 30.0, 1.0)
                signals.append(SignalResult(signal=Signal.SELL, strength=strength))
            else:
                signals.append(SignalResult(signal=Signal.HOLD, strength=0.0))
        return signals

    def test_buy_on_oversold(self) -> None:
        """A series with consistent declines should trigger BUY (low RSI)."""
        # Steady downtrend to push RSI below 30.
        prices = [100.0]
        for _ in range(50):
            prices.append(prices[-1] * 0.99)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=len(prices), freq="D"),
                "open": prices,
                "high": [p * 1.005 for p in prices],
                "low": [p * 0.995 for p in prices],
                "close": prices,
                "volume": [1_000_000.0] * len(prices),
            }
        )
        signals = self._simulate_rsi_signal(df)
        buy_signals = [s for s in signals if s.signal == Signal.BUY]
        assert len(buy_signals) > 0, "Expected BUY signals on oversold data"

    def test_sell_on_overbought(self) -> None:
        """A series with consistent gains should trigger SELL (high RSI)."""
        prices = [100.0]
        for _ in range(50):
            prices.append(prices[-1] * 1.01)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=len(prices), freq="D"),
                "open": prices,
                "high": [p * 1.005 for p in prices],
                "low": [p * 0.995 for p in prices],
                "close": prices,
                "volume": [1_000_000.0] * len(prices),
            }
        )
        signals = self._simulate_rsi_signal(df)
        sell_signals = [s for s in signals if s.signal == Signal.SELL]
        assert len(sell_signals) > 0, "Expected SELL signals on overbought data"

    def test_strength_bounds(self) -> None:
        """All signal strengths should be in [0.0, 1.0]."""
        df = _make_ohlcv(100, seed=99)
        signals = self._simulate_rsi_signal(df)
        assert all(0.0 <= s.strength <= 1.0 for s in signals)


class TestDonchianBreakoutSignals:
    """Tests that simulate Donchian breakout signal generation logic."""

    @staticmethod
    def _simulate_donchian_signal(df: pd.DataFrame, period: int = 20) -> list[SignalResult]:
        """Simulate Donchian breakout: BUY when close > upper, SELL when close < lower."""
        from indicators.core import donchian as _donchian

        ch = _donchian(df["high"], df["low"], period=period)
        signals: list[SignalResult] = []
        for i in range(len(df)):
            upper = ch["upper"].iloc[i]
            lower = ch["lower"].iloc[i]
            close = df["close"].iloc[i]

            if pd.isna(upper) or pd.isna(lower):
                signals.append(SignalResult(signal=Signal.HOLD, strength=0.0))
            elif close >= upper:
                width = upper - lower if upper != lower else 1.0
                strength = min(abs(close - upper) / width + 0.5, 1.0)
                signals.append(SignalResult(signal=Signal.BUY, strength=strength))
            elif close <= lower:
                width = upper - lower if upper != lower else 1.0
                strength = min(abs(lower - close) / width + 0.5, 1.0)
                signals.append(SignalResult(signal=Signal.SELL, strength=strength))
            else:
                signals.append(SignalResult(signal=Signal.HOLD, strength=0.0))
        return signals

    def test_buy_on_new_high(self) -> None:
        """A breakout above the channel should trigger BUY."""
        # Flat range then a sharp move up.
        # Use high = close so that the rolling max of highs equals
        # the rolling max of closes.  A breakout close that matches
        # or exceeds the channel upper (close >= upper) triggers BUY.
        n_flat = 25
        n_breakout = 5
        close_flat = [100.0] * n_flat
        close_break = [100.0 + 5.0 * (i + 1) for i in range(n_breakout)]
        all_close = close_flat + close_break
        n = len(all_close)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
                "open": all_close,
                "high": all_close,  # high == close so upper channel equals close
                "low": [c - 1.0 for c in all_close],
                "close": all_close,
                "volume": [1_000_000.0] * n,
            }
        )
        signals = self._simulate_donchian_signal(df, period=20)
        buy_signals = [s for s in signals if s.signal == Signal.BUY]
        assert len(buy_signals) > 0, "Expected BUY signals on breakout above channel"

    def test_sell_on_new_low(self) -> None:
        """A breakdown below the channel should trigger SELL."""
        # Use low = close so that the rolling min of lows equals
        # the rolling min of closes.  A breakdown close that matches
        # or falls below the channel lower (close <= lower) triggers SELL.
        n_flat = 25
        n_break = 5
        close_flat = [100.0] * n_flat
        close_break = [100.0 - 5.0 * (i + 1) for i in range(n_break)]
        all_close = close_flat + close_break
        n = len(all_close)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
                "open": all_close,
                "high": [c + 1.0 for c in all_close],
                "low": all_close,  # low == close so lower channel equals close
                "close": all_close,
                "volume": [1_000_000.0] * n,
            }
        )
        signals = self._simulate_donchian_signal(df, period=20)
        sell_signals = [s for s in signals if s.signal == Signal.SELL]
        assert len(sell_signals) > 0, "Expected SELL signals on breakdown below channel"

    def test_strength_bounds(self) -> None:
        """All signal strengths should be in [0.0, 1.0]."""
        df = _make_ohlcv(100, seed=55)
        signals = self._simulate_donchian_signal(df, period=20)
        assert all(0.0 <= s.strength <= 1.0 for s in signals)
