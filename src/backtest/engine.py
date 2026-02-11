"""Backtesting engine: simulates a strategy over historical OHLCV data.

The engine iterates through candles chronologically, applies strategy signals,
and tracks equity, trades, and drawdowns.  For the MVP it supports long-only
positions with market orders filled at the next bar's open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.config import get_config
from backtest.metrics import (
    BacktestResult,
    cagr,
    exposure,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)
from indicators.core import atr as compute_atr
from strategies import get_strategy
from strategies.base import Signal, SignalResult, Strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fee preset defaults (used when config/config.yaml is not available)
# ---------------------------------------------------------------------------

_DEFAULT_FEE_PRESETS: dict[str, dict[str, float]] = {
    "crypto_major": {
        "fee_pct": 0.001,
        "slippage_pct": 0.0005,
        "spread_pct": 0.0002,
    },
    "liquid_stock": {
        "fee_pct": 0.0,
        "slippage_pct": 0.0003,
        "spread_pct": 0.0001,
    },
    "penny_stock": {
        "fee_pct": 0.0,
        "slippage_pct": 0.005,
        "spread_pct": 0.003,
    },
}


@dataclass
class RiskConfig:
    """Risk-management parameters for a backtest run.

    Attributes:
        stop_loss: Stop-loss mode -- ``"none"``, ``"fixed_percent"``, or
                   ``"atr_multiple"``.
        atr_stop_mult: ATR multiplier for the ATR-based stop.
        fixed_stop_pct: Percentage drop from entry to trigger fixed stop.
        atr_period: Lookback period for ATR calculation.
        position_size_pct: Fraction of current equity to allocate per trade.
    """

    stop_loss: str = "none"
    atr_stop_mult: float = 2.0
    fixed_stop_pct: float = 0.05
    atr_period: int = 14
    position_size_pct: float = 0.10


def _load_fee_preset(preset_name: str) -> dict[str, float]:
    """Resolve a fee preset from config, falling back to built-in defaults."""
    try:
        cfg = get_config()
        presets = cfg.get("fee_presets", {})
        if preset_name in presets:
            return {
                "fee_pct": float(presets[preset_name].get("fee_pct", 0.0)),
                "slippage_pct": float(presets[preset_name].get("slippage_pct", 0.0)),
                "spread_pct": float(presets[preset_name].get("spread_pct", 0.0)),
            }
    except Exception:
        pass

    if preset_name in _DEFAULT_FEE_PRESETS:
        return _DEFAULT_FEE_PRESETS[preset_name].copy()

    logger.warning(
        "Unknown fee preset '%s', falling back to liquid_stock defaults",
        preset_name,
    )
    return _DEFAULT_FEE_PRESETS["liquid_stock"].copy()


def _effective_buy_price(base_price: float, fees: dict[str, float]) -> float:
    """Price actually paid per share when buying (higher than market)."""
    return base_price * (
        1.0 + fees["slippage_pct"] + fees["spread_pct"] / 2.0 + fees["fee_pct"]
    )


def _effective_sell_price(base_price: float, fees: dict[str, float]) -> float:
    """Price actually received per share when selling (lower than market)."""
    return base_price * (
        1.0 - fees["slippage_pct"] - fees["spread_pct"] / 2.0 - fees["fee_pct"]
    )


class BacktestEngine:
    """Long-only backtesting engine.

    Parameters:
        initial_capital: Starting cash balance.
        fee_preset: Name of a fee/slippage/spread preset (e.g.
                    ``"liquid_stock"``).  Custom presets can be defined in
                    ``config/config.yaml`` under ``fee_presets:``.
        risk_config: :class:`RiskConfig` instance.  When *None*, a
                     default (no stop-loss, 10% position sizing) is used.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fee_preset: str = "liquid_stock",
        risk_config: RiskConfig | None = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.fees = _load_fee_preset(fee_preset)
        self.fee_preset = fee_preset
        self.risk = risk_config or RiskConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        params: dict[str, Any],
        asset: str = "UNKNOWN",
        timeframe: str = "1d",
        ml_filter: bool = False,
    ) -> BacktestResult:
        """Execute a full backtest.

        Args:
            df: OHLCV DataFrame with columns ``timestamp``, ``open``,
                ``high``, ``low``, ``close``, ``volume``.
            strategy: A :class:`~strategies.base.Strategy` instance.
            params: Strategy-specific parameter dict forwarded to
                    :meth:`Strategy.evaluate`.
            asset: Asset identifier (for labelling output).
            timeframe: Candle timeframe string (for labelling output).

        Returns:
            A :class:`BacktestResult` containing equity curve, trade list,
            and all computed metrics.
        """
        if df.empty or len(df) < 2:
            logger.warning("Insufficient data for backtest (%d rows)", len(df))
            return self._empty_result(asset, timeframe, strategy.name, params)

        df = df.reset_index(drop=True)

        # Pre-compute all strategy signals in one pass using compute()
        prepared_df = df.set_index("timestamp") if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex) else df
        all_bar_signals: list[SignalResult] = strategy.compute(
            prepared_df, asset, timeframe, params,
        )

        # Optionally filter signals through ML model
        if ml_filter:
            try:
                from ml.scorer import MLScorer
                from ml.features import build_features

                scorer = MLScorer()
                if scorer.load_model(asset, timeframe):
                    feat_df = build_features(df, timeframe)
                    if not feat_df.empty:
                        meta_cols = {"timestamp", "close"}
                        feature_cols = [c for c in feat_df.columns if c not in meta_cols]

                        # Map bar timestamps to feature-row indices so we
                        # look up the correct features for each bar.
                        feat_ts = pd.to_datetime(feat_df["timestamp"])
                        ts_to_feat: dict = {}
                        for fidx in range(len(feat_df)):
                            ts_to_feat[feat_ts.iloc[fidx]] = fidx

                        bar_timestamps = pd.to_datetime(df["timestamp"])

                        for i in range(len(all_bar_signals)):
                            sig = all_bar_signals[i]
                            if sig.signal != Signal.BUY:
                                continue
                            if i >= len(df):
                                continue
                            bar_ts = bar_timestamps.iloc[i]
                            feat_idx = ts_to_feat.get(bar_ts)
                            if feat_idx is None:
                                continue
                            try:
                                X = feat_df[feature_cols].iloc[[feat_idx]]
                                prob = scorer._model.predict_proba(X)
                                if float(prob[0]) < 0.55:
                                    all_bar_signals[i] = SignalResult(
                                        signal=Signal.HOLD,
                                        strength=0.0,
                                        strategy_name=sig.strategy_name,
                                        asset=sig.asset,
                                        timeframe=sig.timeframe,
                                        timestamp=sig.timestamp,
                                        explanation=sig.explanation,
                                    )
                            except Exception:
                                pass
                    logger.info("ML filter applied to backtest signals")
                else:
                    logger.warning("ML filter requested but no model found for %s/%s", asset, timeframe)
            except ImportError:
                logger.warning("ML filter requested but scikit-learn not available")

        # Pre-compute ATR if needed for stop-loss
        atr_series: pd.Series | None = None
        if self.risk.stop_loss == "atr_multiple":
            atr_series = compute_atr(
                df["high"], df["low"], df["close"], period=self.risk.atr_period
            )

        # State
        cash: float = self.initial_capital
        shares: float = 0.0
        entry_price: float = 0.0
        entry_bar: int = 0
        entry_time: pd.Timestamp | None = None

        equity_values: list[float] = []
        equity_timestamps: list[Any] = []
        trades: list[dict[str, Any]] = []

        # Pending order state: signal from bar i fills at bar i+1's open
        pending_signal: Signal | None = None

        for i in range(len(df)):
            row = df.iloc[i]
            price_open = float(row["open"])
            price_high = float(row["high"])
            price_low = float(row["low"])
            price_close = float(row["close"])
            timestamp = row["timestamp"]

            # ---- Execute pending order at this bar's open ----
            if pending_signal is not None and i > 0:
                if pending_signal == Signal.BUY and shares == 0:
                    # Buy: allocate position_size_pct of current equity
                    buy_price = _effective_buy_price(price_open, self.fees)
                    if buy_price > 0:
                        allocation = cash * self.risk.position_size_pct
                        shares = allocation / buy_price
                        cash -= shares * buy_price
                        entry_price = buy_price
                        entry_bar = i
                        entry_time = timestamp

                elif pending_signal == Signal.SELL and shares > 0:
                    # Sell: close entire position
                    sell_price = _effective_sell_price(price_open, self.fees)
                    proceeds = shares * sell_price
                    pnl = proceeds - shares * entry_price
                    pnl_pct = (sell_price / entry_price - 1.0) if entry_price > 0 else 0.0

                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": timestamp,
                            "entry_price": entry_price,
                            "exit_price": sell_price,
                            "shares": shares,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "entry_bar": entry_bar,
                            "exit_bar": i,
                            "side": "long",
                            "exit_reason": "signal",
                        }
                    )
                    cash += proceeds
                    shares = 0.0
                    entry_price = 0.0
                    entry_time = None

                pending_signal = None

            # ---- Check stop-loss (intra-bar, using low) ----
            if shares > 0:
                stopped = False
                if self.risk.stop_loss == "fixed_percent":
                    stop_price = entry_price * (1.0 - self.risk.fixed_stop_pct)
                    if price_low <= stop_price:
                        stopped = True
                        exit_price = _effective_sell_price(stop_price, self.fees)

                elif self.risk.stop_loss == "atr_multiple" and atr_series is not None:
                    atr_val = atr_series.iloc[i]
                    if not np.isnan(atr_val) and atr_val > 0:
                        stop_price = entry_price - self.risk.atr_stop_mult * atr_val
                        if price_low <= stop_price:
                            stopped = True
                            exit_price = _effective_sell_price(stop_price, self.fees)

                if stopped:
                    proceeds = shares * exit_price
                    pnl = proceeds - shares * entry_price
                    pnl_pct = (exit_price / entry_price - 1.0) if entry_price > 0 else 0.0

                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": timestamp,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "shares": shares,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "entry_bar": entry_bar,
                            "exit_bar": i,
                            "side": "long",
                            "exit_reason": f"stop_{self.risk.stop_loss}",
                        }
                    )
                    cash += proceeds
                    shares = 0.0
                    entry_price = 0.0
                    entry_time = None
                    pending_signal = None  # cancel any pending order

            # ---- Look up pre-computed signal for current bar ----
            if i < len(all_bar_signals):
                signal_result = all_bar_signals[i]
                if signal_result.signal in (Signal.BUY, Signal.SELL):
                    pending_signal = signal_result.signal

            # ---- Record equity ----
            equity = cash + shares * price_close
            equity_values.append(equity)
            equity_timestamps.append(timestamp)

        # ---- Build equity curve ----
        equity_curve = pd.Series(
            equity_values,
            index=pd.DatetimeIndex(equity_timestamps),
            name="equity",
            dtype=float,
        )

        total_bars = len(df)

        result = BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            config={
                "asset": asset,
                "timeframe": timeframe,
                "strategy": strategy.name,
                "params": params,
                "fee_preset": self.fee_preset,
                "fees": self.fees,
                "risk": {
                    "stop_loss": self.risk.stop_loss,
                    "atr_stop_mult": self.risk.atr_stop_mult,
                    "fixed_stop_pct": self.risk.fixed_stop_pct,
                    "atr_period": self.risk.atr_period,
                    "position_size_pct": self.risk.position_size_pct,
                },
                "initial_capital": self.initial_capital,
            },
            initial_capital=self.initial_capital,
            final_equity=equity_values[-1] if equity_values else self.initial_capital,
            cagr=cagr(equity_curve),
            sharpe=sharpe_ratio(equity_curve),
            max_drawdown=max_drawdown(equity_curve),
            win_rate=win_rate(trades),
            profit_factor=profit_factor(trades),
            exposure=exposure(trades, total_bars),
            total_trades=len(trades),
            total_bars=total_bars,
        )

        logger.info(
            "Backtest %s | %s | %s: %d trades, CAGR=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
            asset,
            timeframe,
            strategy.name,
            len(trades),
            result.cagr * 100,
            result.sharpe,
            result.max_drawdown * 100,
        )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(
        self,
        asset: str,
        timeframe: str,
        strategy_name: str,
        params: dict[str, Any],
    ) -> BacktestResult:
        """Return a zeroed-out result for insufficient data."""
        return BacktestResult(
            config={
                "asset": asset,
                "timeframe": timeframe,
                "strategy": strategy_name,
                "params": params,
                "fee_preset": self.fee_preset,
                "fees": self.fees,
                "risk": {
                    "stop_loss": self.risk.stop_loss,
                    "atr_stop_mult": self.risk.atr_stop_mult,
                    "fixed_stop_pct": self.risk.fixed_stop_pct,
                    "atr_period": self.risk.atr_period,
                    "position_size_pct": self.risk.position_size_pct,
                },
                "initial_capital": self.initial_capital,
            },
            initial_capital=self.initial_capital,
            final_equity=self.initial_capital,
        )
