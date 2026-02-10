"""Tests for backtest metrics module.

The backtest engine (backtest/engine.py) does not exist yet, so these tests
focus on the metric functions and the BacktestResult container that are
already implemented in backtest/metrics.py.

Each metric is tested with hand-crafted equity curves and trade lists
so that expected values can be verified by simple arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    BacktestResult,
    cagr,
    exposure,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _equity_curve(values: list[float], start_date: str = "2023-01-01") -> pd.Series:
    """Build an equity curve Series with a DatetimeIndex."""
    idx = pd.date_range(start_date, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype="float64")


def _flat_equity(value: float = 10_000.0, n: int = 252) -> pd.Series:
    """An equity curve that never changes (no trades)."""
    return _equity_curve([value] * n)


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------

class TestCAGR:
    def test_cagr_positive_return(self) -> None:
        """100% return over exactly 1 year should give CAGR ~ 1.0."""
        curve = _equity_curve([10_000.0, 20_000.0], start_date="2023-01-01")
        # 1 day apart -> very high annualised CAGR; test with 365 days instead
        values = [10_000.0] + [10_000.0] * 364 + [20_000.0]
        curve = _equity_curve(values)
        result = cagr(curve)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_cagr_no_change(self) -> None:
        """Flat equity should give CAGR = 0."""
        result = cagr(_flat_equity())
        assert result == pytest.approx(0.0)

    def test_cagr_negative_return(self) -> None:
        """Losing half the equity should give negative CAGR."""
        values = [10_000.0] + [10_000.0] * 364 + [5_000.0]
        curve = _equity_curve(values)
        result = cagr(curve)
        assert result < 0.0

    def test_cagr_single_point(self) -> None:
        """A single-point curve should return 0.0."""
        curve = _equity_curve([10_000.0])
        assert cagr(curve) == 0.0

    def test_cagr_zero_start(self) -> None:
        """Starting equity of zero should return 0.0 (avoid division by zero)."""
        curve = _equity_curve([0.0, 100.0, 200.0])
        assert cagr(curve) == 0.0


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_sharpe_flat_equity(self) -> None:
        """Flat equity -> zero returns -> Sharpe = 0."""
        result = sharpe_ratio(_flat_equity())
        assert result == pytest.approx(0.0)

    def test_sharpe_positive_for_uptrend(self) -> None:
        """A steadily rising equity curve should have positive Sharpe."""
        values = list(np.linspace(10_000, 12_000, 252))
        curve = _equity_curve(values)
        result = sharpe_ratio(curve)
        assert result > 0.0

    def test_sharpe_single_point(self) -> None:
        """Too few data points should return 0.0."""
        assert sharpe_ratio(_equity_curve([10_000.0])) == 0.0

    def test_sharpe_with_risk_free_rate(self) -> None:
        """Higher risk-free rate should lower the Sharpe ratio."""
        values = list(np.linspace(10_000, 11_000, 252))
        curve = _equity_curve(values)
        sharpe_0 = sharpe_ratio(curve, risk_free_rate=0.0)
        sharpe_5 = sharpe_ratio(curve, risk_free_rate=0.05)
        assert sharpe_0 > sharpe_5


# ---------------------------------------------------------------------------
# Max Drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_max_drawdown_known_value(self) -> None:
        """Curve: 100 -> 120 -> 90 -> 110.  Drawdown = (120-90)/120 = 25%."""
        curve = _equity_curve([100.0, 120.0, 90.0, 110.0])
        result = max_drawdown(curve)
        assert result == pytest.approx(0.25)

    def test_max_drawdown_no_drawdown(self) -> None:
        """A monotonically increasing curve should have zero drawdown."""
        values = list(np.linspace(100, 200, 50))
        curve = _equity_curve(values)
        result = max_drawdown(curve)
        assert result == pytest.approx(0.0)

    def test_max_drawdown_single_point(self) -> None:
        """Single-point curve returns 0.0."""
        assert max_drawdown(_equity_curve([100.0])) == 0.0

    def test_max_drawdown_full_loss(self) -> None:
        """Curve that drops to near zero should approach 100% drawdown."""
        curve = _equity_curve([10_000.0, 10_000.0, 100.0])
        result = max_drawdown(curve)
        assert result == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Win Rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_win_rate_all_winners(self) -> None:
        trades = [{"pnl": 100.0}, {"pnl": 50.0}, {"pnl": 200.0}]
        assert win_rate(trades) == pytest.approx(1.0)

    def test_win_rate_all_losers(self) -> None:
        trades = [{"pnl": -100.0}, {"pnl": -50.0}]
        assert win_rate(trades) == pytest.approx(0.0)

    def test_win_rate_mixed(self) -> None:
        trades = [{"pnl": 100.0}, {"pnl": -50.0}, {"pnl": 200.0}, {"pnl": -10.0}]
        assert win_rate(trades) == pytest.approx(0.5)

    def test_win_rate_no_trades(self) -> None:
        assert win_rate([]) == 0.0


# ---------------------------------------------------------------------------
# Profit Factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_profit_factor_profitable(self) -> None:
        """Gross profit 300, gross loss 100 -> PF = 3.0."""
        trades = [{"pnl": 200.0}, {"pnl": 100.0}, {"pnl": -100.0}]
        assert profit_factor(trades) == pytest.approx(3.0)

    def test_profit_factor_no_losses(self) -> None:
        """All winners -> infinite profit factor."""
        trades = [{"pnl": 100.0}, {"pnl": 50.0}]
        result = profit_factor(trades)
        assert math.isinf(result) and result > 0

    def test_profit_factor_no_wins(self) -> None:
        """All losers -> profit factor 0."""
        trades = [{"pnl": -100.0}]
        assert profit_factor(trades) == pytest.approx(0.0)

    def test_profit_factor_no_trades(self) -> None:
        assert profit_factor([]) == 0.0


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

class TestExposure:
    def test_exposure_full(self) -> None:
        """A single trade spanning all bars -> exposure = 1.0."""
        trades = [{"entry_bar": 0, "exit_bar": 100}]
        assert exposure(trades, total_bars=100) == pytest.approx(1.0)

    def test_exposure_half(self) -> None:
        """A trade spanning half the bars -> exposure = 0.5."""
        trades = [{"entry_bar": 0, "exit_bar": 50}]
        assert exposure(trades, total_bars=100) == pytest.approx(0.5)

    def test_exposure_no_trades(self) -> None:
        assert exposure([], total_bars=100) == 0.0

    def test_exposure_zero_bars(self) -> None:
        trades = [{"entry_bar": 0, "exit_bar": 10}]
        assert exposure(trades, total_bars=0) == 0.0


# ---------------------------------------------------------------------------
# BacktestResult container
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_default_values(self) -> None:
        """BacktestResult should have sensible defaults."""
        result = BacktestResult()
        assert result.initial_capital == 0.0
        assert result.final_equity == 0.0
        assert result.total_trades == 0
        assert result.trades == []

    def test_summary_keys(self) -> None:
        """summary() should contain all expected metric keys."""
        result = BacktestResult(
            initial_capital=10_000.0,
            final_equity=12_000.0,
            cagr=0.20,
            sharpe=1.5,
            max_drawdown=0.10,
            win_rate=0.6,
            profit_factor=2.0,
            exposure=0.4,
            total_trades=20,
            total_bars=252,
        )
        summary = result.summary()
        expected_keys = {
            "initial_capital",
            "final_equity",
            "cagr",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "exposure",
            "total_trades",
            "total_bars",
        }
        assert set(summary.keys()) == expected_keys

    def test_summary_rounding(self) -> None:
        """Metric values in summary should be rounded."""
        result = BacktestResult(cagr=0.123456789, sharpe=1.987654321)
        summary = result.summary()
        assert summary["cagr"] == 0.1235
        assert summary["sharpe"] == 1.9877


# ---------------------------------------------------------------------------
# Simulated backtest scenarios
# ---------------------------------------------------------------------------

class TestBacktestScenarios:
    """End-to-end-style tests using the metric functions together."""

    def test_no_signals_equity_unchanged(self) -> None:
        """With no trades the equity should stay flat and all metrics reflect that."""
        equity = _flat_equity(10_000.0, n=252)
        trades: list[dict] = []

        assert cagr(equity) == pytest.approx(0.0)
        assert sharpe_ratio(equity) == pytest.approx(0.0)
        assert max_drawdown(equity) == pytest.approx(0.0)
        assert win_rate(trades) == 0.0
        assert exposure(trades, total_bars=252) == 0.0

    def test_single_winning_trade(self) -> None:
        """A single winning trade should produce positive CAGR and 100% win rate."""
        # Equity: flat for 50 bars, then jumps +10% at bar 50, flat afterwards.
        values = [10_000.0] * 50 + [11_000.0] * 203
        equity = _equity_curve(values)
        trades = [
            {
                "pnl": 1000.0,
                "entry_bar": 45,
                "exit_bar": 50,
                "entry_price": 100.0,
                "exit_price": 110.0,
            }
        ]

        assert cagr(equity) > 0.0
        assert win_rate(trades) == 1.0
        assert max_drawdown(equity) == pytest.approx(0.0)

    def test_fee_application_reduces_returns(self) -> None:
        """Simulating fees by reducing PnL should lower profit factor."""
        gross_trades = [
            {"pnl": 100.0, "entry_bar": 0, "exit_bar": 10},
            {"pnl": -50.0, "entry_bar": 20, "exit_bar": 30},
        ]
        fee_per_trade = 5.0
        net_trades = [
            {**t, "pnl": t["pnl"] - fee_per_trade} for t in gross_trades
        ]
        pf_gross = profit_factor(gross_trades)
        pf_net = profit_factor(net_trades)
        assert pf_net < pf_gross

    def test_atr_stop_loss_simulation(self) -> None:
        """Simulate an ATR-based stop: if price drops > 2x ATR, exit with a loss."""
        from indicators.core import atr as compute_atr

        rng = np.random.default_rng(42)
        n = 60
        close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)), dtype="float64")
        high = close + rng.uniform(0.2, 1.0, n)
        low = close - rng.uniform(0.2, 1.0, n)

        atr_vals = compute_atr(high, low, close, period=14)
        entry_bar = 30
        entry_price = close.iloc[entry_bar]
        atr_at_entry = atr_vals.iloc[entry_bar]
        stop_level = entry_price - 2.0 * atr_at_entry

        # Walk forward and check if stop triggers.
        stopped_out = False
        exit_bar = None
        for i in range(entry_bar + 1, n):
            if low.iloc[i] <= stop_level:
                stopped_out = True
                exit_bar = i
                break

        if stopped_out:
            exit_price = stop_level
            pnl = exit_price - entry_price
            assert pnl < 0, "Stop loss should produce a negative PnL"
            assert exit_bar is not None and exit_bar > entry_bar
        else:
            # If stop was not triggered, the position is still open.
            # This is valid -- just verify stop_level was set correctly.
            assert stop_level < entry_price
