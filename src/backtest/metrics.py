"""Backtest performance metrics and result container.

All metric functions accept simple Python / numpy structures and return
scalar values.  The :class:`BacktestResult` dataclass bundles every metric
together with the equity curve, trade list, and configuration snapshot so
that a single object fully describes one backtest run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate.

    Assumes the equity curve index is a :class:`~pandas.DatetimeIndex` **or**
    that consecutive entries are one bar apart and the total duration can be
    inferred from the first and last timestamps.

    Args:
        equity_curve: Series of portfolio equity values indexed by datetime.

    Returns:
        Annualised compound growth rate as a decimal (e.g. 0.12 for 12%).
        Returns 0.0 when the curve has fewer than 2 points or the start
        equity is zero.
    """
    if len(equity_curve) < 2:
        return 0.0

    start_val = equity_curve.iloc[0]
    end_val = equity_curve.iloc[-1]

    if start_val <= 0:
        return 0.0

    # Duration in years
    if isinstance(equity_curve.index, pd.DatetimeIndex):
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
    else:
        days = len(equity_curve) - 1  # fallback: assume daily bars

    if days <= 0:
        return 0.0

    years = days / 365.25
    return (end_val / start_val) ** (1.0 / years) - 1.0


def sharpe_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Annualised Sharpe ratio.

    Args:
        equity_curve: Series of portfolio equity values.
        risk_free_rate: Annual risk-free rate as a decimal.
        periods_per_year: Number of bars per year (252 for daily, 252*6.5*4
                          for 15-min, etc.).

    Returns:
        Annualised Sharpe ratio.  Returns 0.0 when there are insufficient
        data points or zero volatility.
    """
    if len(equity_curve) < 2:
        return 0.0

    returns = equity_curve.pct_change().dropna()
    if len(returns) < 2:
        return 0.0

    std = returns.std()
    if std == 0 or math.isnan(std):
        return 0.0

    excess_per_bar = returns.mean() - risk_free_rate / periods_per_year
    return float(excess_per_bar / std * math.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough decline.

    Args:
        equity_curve: Series of portfolio equity values.

    Returns:
        Maximum drawdown as a positive decimal (e.g. 0.25 means 25% decline).
        Returns 0.0 when the curve has fewer than 2 points.
    """
    if len(equity_curve) < 2:
        return 0.0

    cummax = equity_curve.cummax()
    drawdowns = (cummax - equity_curve) / cummax
    result = drawdowns.max()
    return 0.0 if math.isnan(result) else float(result)


def win_rate(trades: list[dict[str, Any]]) -> float:
    """Percentage of profitable trades.

    Args:
        trades: List of trade dicts, each containing a ``'pnl'`` key.

    Returns:
        Win rate as a decimal in [0, 1].  Returns 0.0 when no trades.
    """
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t["pnl"] > 0)
    return winners / len(trades)


def profit_factor(trades: list[dict[str, Any]]) -> float:
    """Gross profit divided by gross loss.

    Args:
        trades: List of trade dicts, each containing a ``'pnl'`` key.

    Returns:
        Profit factor (>1 means profitable overall).
        Returns ``inf`` when there are only winning trades, or 0.0 when
        there are no trades or only losing trades.
    """
    if not trades:
        return 0.0

    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def exposure(trades: list[dict[str, Any]], total_bars: int) -> float:
    """Percentage of time the portfolio was in a position.

    Each trade dict must contain ``'entry_bar'`` and ``'exit_bar'`` integer
    indices.

    Args:
        trades: List of trade dicts.
        total_bars: Total number of bars in the backtest.

    Returns:
        Exposure as a decimal in [0, 1].  Returns 0.0 when there are no
        trades or zero bars.
    """
    if not trades or total_bars <= 0:
        return 0.0

    bars_in_market = sum(
        t.get("exit_bar", 0) - t.get("entry_bar", 0) for t in trades
    )
    return min(bars_in_market / total_bars, 1.0)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Complete result of a single backtest run.

    Attributes:
        equity_curve: Series of portfolio equity indexed by timestamp.
        trades: List of trade dicts with keys such as ``entry_time``,
                ``exit_time``, ``entry_price``, ``exit_price``, ``shares``,
                ``pnl``, ``pnl_pct``, ``entry_bar``, ``exit_bar``, ``side``.
        config: Snapshot of the configuration used (strategy name, params,
                fee preset, risk config, etc.).
        initial_capital: Starting portfolio value.
        final_equity: Ending portfolio value.
        cagr: Compound Annual Growth Rate.
        sharpe: Annualised Sharpe ratio.
        max_drawdown: Maximum peak-to-trough decline.
        win_rate: Fraction of profitable trades.
        profit_factor: Gross profit / gross loss.
        exposure: Fraction of time in a position.
        total_trades: Number of round-trip trades.
        total_bars: Number of candles processed.
    """

    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    initial_capital: float = 0.0
    final_equity: float = 0.0

    cagr: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    exposure: float = 0.0

    total_trades: int = 0
    total_bars: int = 0

    def summary(self) -> dict[str, Any]:
        """Return a plain-dict summary suitable for logging or display."""
        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(self.final_equity, 2),
            "cagr": round(self.cagr, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "exposure": round(self.exposure, 4),
            "total_trades": self.total_trades,
            "total_bars": self.total_bars,
        }
