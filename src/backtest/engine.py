"""Backtesting engine: simulates a strategy over historical OHLCV data.

The engine iterates through candles chronologically, applies strategy signals,
and tracks equity, trades, and drawdowns.  Supports long-only and long/short
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
from indicators.core import atr as compute_atr, volatility as compute_volatility
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
        allow_short: Enable short selling on SELL signals when flat.
        trailing_stop: Trailing stop mode -- ``"none"``, ``"atr_trail"``,
                       ``"percent_trail"``.
        trailing_atr_mult: ATR multiplier for trailing stop distance.
        trailing_pct: Percentage for trailing stop distance.
        position_sizing: Sizing mode -- ``"fixed"``, ``"volatility_scaled"``.
        vol_target_risk: Target risk per trade for vol-scaled sizing (decimal).
        cooldown_bars: Number of bars to skip after a stop-loss exit.
    """

    stop_loss: str = "none"
    atr_stop_mult: float = 2.0
    fixed_stop_pct: float = 0.05
    atr_period: int = 14
    position_size_pct: float = 0.10
    allow_short: bool = False
    trailing_stop: str = "none"
    trailing_atr_mult: float = 2.0
    trailing_pct: float = 0.05
    position_sizing: str = "fixed"
    vol_target_risk: float = 0.02
    cooldown_bars: int = 0


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
    """Backtesting engine with long/short support, trailing stops, and
    volatility-scaled position sizing.

    Parameters:
        initial_capital: Starting cash balance.
        fee_preset: Name of a fee/slippage/spread preset.
        risk_config: :class:`RiskConfig` instance.
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
        """Execute a full backtest."""
        if df.empty or len(df) < 2:
            logger.warning("Insufficient data for backtest (%d rows)", len(df))
            return self._empty_result(asset, timeframe, strategy.name, params)

        df = df.reset_index(drop=True)

        # Pre-compute all strategy signals in one pass
        prepared_df = (
            df.set_index("timestamp")
            if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex)
            else df
        )
        all_bar_signals: list[SignalResult] = strategy.compute(
            prepared_df, asset, timeframe, params,
        )

        # Optionally filter signals through ML model
        if ml_filter:
            all_bar_signals = self._apply_ml_filter(
                all_bar_signals, df, asset, timeframe,
            )

        # Pre-compute ATR for stop-loss and vol sizing
        atr_series: pd.Series = compute_atr(
            df["high"], df["low"], df["close"], period=self.risk.atr_period,
        )

        # Pre-compute volatility for vol-scaled sizing
        vol_series: pd.Series | None = None
        if self.risk.position_sizing == "volatility_scaled":
            vol_series = compute_volatility(df["close"], period=20)

        # -- Simulation state --
        cash: float = self.initial_capital
        position_size: float = 0.0  # positive = long, negative = short
        entry_price: float = 0.0
        entry_bar: int = 0
        entry_time: pd.Timestamp | None = None
        trailing_stop_price: float = 0.0
        cooldown_remaining: int = 0

        equity_values: list[float] = []
        equity_timestamps: list[Any] = []
        trades: list[dict[str, Any]] = []

        pending_signal: Signal | None = None

        for i in range(len(df)):
            row = df.iloc[i]
            price_open = float(row["open"])
            price_high = float(row["high"])
            price_low = float(row["low"])
            price_close = float(row["close"])
            timestamp = row["timestamp"]
            atr_val = float(atr_series.iloc[i]) if not np.isnan(atr_series.iloc[i]) else 0.0

            # Guard flag: ensures cooldown decrements at most once per bar
            cooldown_decremented: bool = False

            # ---- Execute pending order at this bar's open ----
            if pending_signal is not None and i > 0:
                if cooldown_remaining > 0:
                    pending_signal = None
                    cooldown_remaining -= 1
                    cooldown_decremented = True
                elif pending_signal == Signal.BUY:
                    if position_size < 0:
                        # Close short first
                        buy_price = _effective_buy_price(price_open, self.fees)
                        pnl = abs(position_size) * (entry_price - buy_price)
                        pnl_pct = (entry_price / buy_price - 1.0) if buy_price > 0 else 0.0
                        trades.append(self._make_trade(
                            entry_time, timestamp, entry_price, buy_price,
                            abs(position_size), pnl, pnl_pct,
                            entry_bar, i, "short", "signal",
                        ))
                        cash += pnl + abs(position_size) * entry_price  # return margin
                        position_size = 0.0
                        entry_price = 0.0
                        entry_time = None
                        trailing_stop_price = 0.0

                    if position_size == 0:
                        # Open long
                        buy_price = _effective_buy_price(price_open, self.fees)
                        if buy_price > 0:
                            alloc = self._compute_position_size(
                                cash, buy_price, atr_val, vol_series, i,
                            )
                            position_size = alloc / buy_price
                            cash -= position_size * buy_price
                            entry_price = buy_price
                            entry_bar = i
                            entry_time = timestamp
                            trailing_stop_price = self._init_trailing_stop(
                                buy_price, atr_val, "long",
                            )

                elif pending_signal == Signal.SELL:
                    if position_size > 0:
                        # Close long
                        sell_price = _effective_sell_price(price_open, self.fees)
                        proceeds = position_size * sell_price
                        pnl = proceeds - position_size * entry_price
                        pnl_pct = (sell_price / entry_price - 1.0) if entry_price > 0 else 0.0
                        trades.append(self._make_trade(
                            entry_time, timestamp, entry_price, sell_price,
                            position_size, pnl, pnl_pct,
                            entry_bar, i, "long", "signal",
                        ))
                        cash += proceeds
                        position_size = 0.0
                        entry_price = 0.0
                        entry_time = None
                        trailing_stop_price = 0.0

                    if position_size == 0 and self.risk.allow_short:
                        # Open short
                        sell_price = _effective_sell_price(price_open, self.fees)
                        if sell_price > 0:
                            alloc = self._compute_position_size(
                                cash, sell_price, atr_val, vol_series, i,
                            )
                            short_shares = alloc / sell_price
                            cash -= short_shares * sell_price  # margin collateral
                            position_size = -short_shares
                            entry_price = sell_price
                            entry_bar = i
                            entry_time = timestamp
                            trailing_stop_price = self._init_trailing_stop(
                                sell_price, atr_val, "short",
                            )

                pending_signal = None

            # ---- Update trailing stop ----
            if position_size != 0 and self.risk.trailing_stop != "none":
                trailing_stop_price = self._update_trailing_stop(
                    trailing_stop_price, price_high, price_low, atr_val,
                    "long" if position_size > 0 else "short",
                )

            # ---- Check stop-loss (including trailing) ----
            if position_size != 0:
                stopped, exit_price = self._check_stop(
                    position_size, entry_price, price_high, price_low,
                    atr_val, trailing_stop_price,
                )
                if stopped:
                    side = "long" if position_size > 0 else "short"
                    if side == "long":
                        proceeds = abs(position_size) * exit_price
                        pnl = proceeds - abs(position_size) * entry_price
                        pnl_pct = (exit_price / entry_price - 1.0) if entry_price > 0 else 0.0
                        cash += proceeds
                    else:
                        pnl = abs(position_size) * (entry_price - exit_price)
                        pnl_pct = (entry_price / exit_price - 1.0) if exit_price > 0 else 0.0
                        cash += pnl + abs(position_size) * entry_price

                    stop_reason = "stop_trailing" if self.risk.trailing_stop != "none" else f"stop_{self.risk.stop_loss}"
                    trades.append(self._make_trade(
                        entry_time, timestamp, entry_price, exit_price,
                        abs(position_size), pnl, pnl_pct,
                        entry_bar, i, side, stop_reason,
                    ))
                    position_size = 0.0
                    entry_price = 0.0
                    entry_time = None
                    trailing_stop_price = 0.0
                    pending_signal = None
                    cooldown_remaining = self.risk.cooldown_bars

            # ---- Cooldown tick (when flat) ----
            # Guard: skip if cooldown was already decremented this bar (e.g. pending-signal rejection above)
            if position_size == 0 and cooldown_remaining > 0 and pending_signal is None and not cooldown_decremented:
                cooldown_remaining -= 1

            # ---- Look up pre-computed signal for current bar ----
            if i < len(all_bar_signals):
                signal_result = all_bar_signals[i]
                if signal_result.signal in (Signal.BUY, Signal.SELL):
                    pending_signal = signal_result.signal

            # ---- Record equity ----
            if position_size > 0:
                equity = cash + position_size * price_close
            elif position_size < 0:
                # Short P&L: profit when price drops
                unrealised = abs(position_size) * (entry_price - price_close)
                margin = abs(position_size) * entry_price
                equity = cash + margin + unrealised
            else:
                equity = cash
            equity_values.append(equity)
            equity_timestamps.append(timestamp)

        # ---- Force-close any open position at the last bar's close ----
        if position_size != 0 and len(df) > 0:
            last_row = df.iloc[-1]
            last_close = float(last_row["close"])
            last_timestamp = last_row["timestamp"]
            last_bar_idx = len(df) - 1
            side = "long" if position_size > 0 else "short"

            if side == "long":
                exit_price = _effective_sell_price(last_close, self.fees)
                proceeds = position_size * exit_price
                pnl = proceeds - position_size * entry_price
                pnl_pct = (exit_price / entry_price - 1.0) if entry_price > 0 else 0.0
                cash += proceeds
            else:
                exit_price = _effective_buy_price(last_close, self.fees)
                pnl = abs(position_size) * (entry_price - exit_price)
                pnl_pct = (entry_price / exit_price - 1.0) if exit_price > 0 else 0.0
                cash += pnl + abs(position_size) * entry_price

            trades.append(self._make_trade(
                entry_time, last_timestamp, entry_price, exit_price,
                abs(position_size), pnl, pnl_pct,
                entry_bar, last_bar_idx, side, "end_of_data",
            ))

            # Update the final equity value to reflect the now-realised cash
            if equity_values:
                equity_values[-1] = cash

            position_size = 0.0

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
                    "allow_short": self.risk.allow_short,
                    "trailing_stop": self.risk.trailing_stop,
                    "position_sizing": self.risk.position_sizing,
                    "cooldown_bars": self.risk.cooldown_bars,
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
            asset, timeframe, strategy.name,
            len(trades), result.cagr * 100, result.sharpe, result.max_drawdown * 100,
        )

        return result

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _compute_position_size(
        self,
        cash: float,
        price: float,
        atr_val: float,
        vol_series: pd.Series | None,
        bar_idx: int,
    ) -> float:
        """Compute dollar amount to allocate for a new position."""
        if self.risk.position_sizing == "volatility_scaled" and atr_val > 0:
            # Risk-based: allocate so that 1-ATR move = vol_target_risk * equity
            risk_per_share = atr_val
            target_risk_dollars = cash * self.risk.vol_target_risk
            dollar_alloc = target_risk_dollars / (risk_per_share / price)
            return min(dollar_alloc, cash * 0.95)  # cap at 95% of cash

        # Fixed percentage
        return cash * self.risk.position_size_pct

    # ------------------------------------------------------------------
    # Trailing stop management
    # ------------------------------------------------------------------

    def _init_trailing_stop(
        self, entry_price: float, atr_val: float, side: str,
    ) -> float:
        """Initialise trailing stop price at entry."""
        if self.risk.trailing_stop == "atr_trail" and atr_val > 0:
            dist = self.risk.trailing_atr_mult * atr_val
            return (entry_price - dist) if side == "long" else (entry_price + dist)
        elif self.risk.trailing_stop == "percent_trail":
            dist = entry_price * self.risk.trailing_pct
            return (entry_price - dist) if side == "long" else (entry_price + dist)
        return 0.0

    def _update_trailing_stop(
        self,
        current_stop: float,
        bar_high: float,
        bar_low: float,
        atr_val: float,
        side: str,
    ) -> float:
        """Ratchet trailing stop in the direction of profit."""
        if self.risk.trailing_stop == "atr_trail" and atr_val > 0:
            dist = self.risk.trailing_atr_mult * atr_val
            if side == "long":
                new_stop = bar_high - dist
                return max(current_stop, new_stop)
            else:
                new_stop = bar_low + dist
                return min(current_stop, new_stop) if current_stop > 0 else new_stop
        elif self.risk.trailing_stop == "percent_trail":
            if side == "long":
                new_stop = bar_high * (1.0 - self.risk.trailing_pct)
                return max(current_stop, new_stop)
            else:
                new_stop = bar_low * (1.0 + self.risk.trailing_pct)
                return min(current_stop, new_stop) if current_stop > 0 else new_stop
        return current_stop

    # ------------------------------------------------------------------
    # Stop-loss checking
    # ------------------------------------------------------------------

    def _check_stop(
        self,
        position_size: float,
        entry_price: float,
        bar_high: float,
        bar_low: float,
        atr_val: float,
        trailing_stop_price: float,
    ) -> tuple[bool, float]:
        """Check if any stop-loss condition is triggered.

        Returns:
            (stopped, exit_price) tuple.
        """
        side = "long" if position_size > 0 else "short"

        # Trailing stop (checked first — takes priority)
        if self.risk.trailing_stop != "none" and trailing_stop_price > 0:
            if side == "long" and bar_low <= trailing_stop_price:
                return True, _effective_sell_price(trailing_stop_price, self.fees)
            if side == "short" and bar_high >= trailing_stop_price:
                return True, _effective_buy_price(trailing_stop_price, self.fees)

        # Fixed percentage stop
        if self.risk.stop_loss == "fixed_percent":
            if side == "long":
                stop_price = entry_price * (1.0 - self.risk.fixed_stop_pct)
                if bar_low <= stop_price:
                    return True, _effective_sell_price(stop_price, self.fees)
            else:
                stop_price = entry_price * (1.0 + self.risk.fixed_stop_pct)
                if bar_high >= stop_price:
                    return True, _effective_buy_price(stop_price, self.fees)

        # ATR-based stop
        if self.risk.stop_loss == "atr_multiple" and atr_val > 0:
            if side == "long":
                stop_price = entry_price - self.risk.atr_stop_mult * atr_val
                if bar_low <= stop_price:
                    return True, _effective_sell_price(stop_price, self.fees)
            else:
                stop_price = entry_price + self.risk.atr_stop_mult * atr_val
                if bar_high >= stop_price:
                    return True, _effective_buy_price(stop_price, self.fees)

        return False, 0.0

    # ------------------------------------------------------------------
    # ML filter
    # ------------------------------------------------------------------

    def _apply_ml_filter(
        self,
        all_bar_signals: list[SignalResult],
        df: pd.DataFrame,
        asset: str,
        timeframe: str,
    ) -> list[SignalResult]:
        """Suppress BUY signals that fail ML confidence threshold."""
        try:
            from ml.scorer import MLScorer
            from ml.features import build_features

            scorer = MLScorer()
            if not scorer.load_model(asset, timeframe):
                logger.warning("ML filter requested but no model for %s/%s", asset, timeframe)
                return all_bar_signals

            # Use threshold and feature list saved with the model
            threshold = scorer._model.metadata.get("threshold", 0.50)
            selected_features = scorer._model.metadata.get("selected_features", None)

            feat_df = build_features(df, timeframe)
            if feat_df.empty:
                return all_bar_signals

            meta_cols = {"timestamp", "close"}
            if selected_features:
                # Use only the features the model was trained on
                feature_cols = [c for c in selected_features if c in feat_df.columns]
            else:
                feature_cols = [c for c in feat_df.columns if c not in meta_cols]

            feat_ts = pd.to_datetime(feat_df["timestamp"])
            ts_to_feat: dict = {feat_ts.iloc[j]: j for j in range(len(feat_df))}
            bar_timestamps = pd.to_datetime(df["timestamp"])

            ml_gated = 0
            ml_passed = 0
            ml_errors = 0

            for idx in range(len(all_bar_signals)):
                sig = all_bar_signals[idx]
                if sig.signal != Signal.BUY or idx >= len(df):
                    continue
                bar_ts = bar_timestamps.iloc[idx]
                feat_idx = ts_to_feat.get(bar_ts)
                if feat_idx is None:
                    continue
                try:
                    X = feat_df[feature_cols].iloc[[feat_idx]]
                    prob = scorer._model.predict_proba(X)
                    if float(prob[0]) < threshold:
                        all_bar_signals[idx] = SignalResult(
                            signal=Signal.HOLD, strength=0.0,
                            strategy_name=sig.strategy_name, asset=sig.asset,
                            timeframe=sig.timeframe, timestamp=sig.timestamp,
                            explanation=sig.explanation,
                        )
                        ml_gated += 1
                    else:
                        ml_passed += 1
                except Exception as exc:
                    ml_errors += 1
                    logger.warning(
                        "ML filter inference error at bar %d (%s) for %s/%s — "
                        "signal passed through unfiltered: %s",
                        idx, bar_ts, asset, timeframe, exc,
                        exc_info=True,
                    )
                    # Pass-through: leave original signal intact (conservative fallback)

            logger.info(
                "ML filter applied (threshold=%.2f, features=%d) — "
                "gated=%d passed=%d errors=%d",
                threshold, len(feature_cols),
                ml_gated, ml_passed, ml_errors,
            )
            if ml_errors:
                logger.warning(
                    "ML filter had %d inference error(s) for %s/%s; "
                    "those bars were passed through unfiltered.",
                    ml_errors, asset, timeframe,
                )
        except ImportError:
            logger.warning("ML filter requested but scikit-learn not available")

        return all_bar_signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_trade(
        entry_time: Any,
        exit_time: Any,
        entry_price: float,
        exit_price: float,
        shares: float,
        pnl: float,
        pnl_pct: float,
        entry_bar: int,
        exit_bar: int,
        side: str,
        exit_reason: str,
    ) -> dict[str, Any]:
        return {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "entry_bar": entry_bar,
            "exit_bar": exit_bar,
            "side": side,
            "exit_reason": exit_reason,
        }

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
