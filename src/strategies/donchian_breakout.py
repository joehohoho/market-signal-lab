"""Donchian Channel Breakout strategy with ADX regime filtering.

Generates BUY when the close breaks above the upper Donchian channel with
above-average volume, RSI filter, and trending regime (ADX > threshold).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators.core import (
    adx as calc_adx,
    donchian,
    rolling_volume_mean,
    rsi as calc_rsi,
)
from strategies.base import Signal, SignalResult, Strategy

_DEFAULT_PARAMS: dict[str, Any] = {
    "channel_period": 20,
    "volume_ma_period": 20,
    "volume_mult": 1.5,
    "regime_filter": True,
    "adx_threshold": 20.0,
}


class DonchianBreakoutStrategy(Strategy):
    """Donchian channel breakout with volume and regime confirmation.

    **BUY** when close breaks above the *previous* bar's upper channel **and**
    volume exceeds ``volume_mult`` times rolling average **and** RSI < 80
    **and** ADX > threshold or ADX is rising (trend starting).

    **SELL** when close breaks below the *previous* bar's lower channel **and**
    volume exceeds ``volume_mult`` times rolling average.

    **HOLD** otherwise.
    """

    @property
    def name(self) -> str:
        return "donchian_breakout"

    def compute(
        self,
        df: pd.DataFrame,
        asset: str,
        timeframe: str,
        params: dict[str, Any],
    ) -> list[SignalResult]:
        p = {**_DEFAULT_PARAMS, **params}
        channel_period: int = int(p["channel_period"])
        volume_ma_period: int = int(p["volume_ma_period"])
        volume_mult: float = float(p["volume_mult"])
        regime_filter: bool = bool(p["regime_filter"])
        adx_threshold: float = float(p["adx_threshold"])

        close: pd.Series = df["close"]
        volume: pd.Series = df["volume"]

        channels = donchian(df["high"], df["low"], channel_period)
        upper: pd.Series = channels["upper"]
        lower: pd.Series = channels["lower"]
        mid: pd.Series = channels["mid"]

        avg_volume: pd.Series = rolling_volume_mean(volume, volume_ma_period)
        rsi_values: pd.Series = calc_rsi(close, 14)

        # Use previous bar's channel boundaries
        prev_upper = upper.shift(1)
        prev_lower = lower.shift(1)

        # Vectorized conditions
        volume_ok = avg_volume.notna() & (volume > volume_mult * avg_volume)
        rsi_ok = rsi_values.isna() | (rsi_values < 80)

        buy_mask = (
            prev_upper.notna()
            & (close > prev_upper)
            & volume_ok
            & rsi_ok
        )
        sell_mask = (
            prev_lower.notna()
            & (close < prev_lower)
            & volume_ok
        )

        # Regime filter: breakouts work in trending or trend-starting markets
        if regime_filter:
            adx_data = calc_adx(df["high"], df["low"], close, period=14)
            adx_values = adx_data["adx"]
            adx_prev = adx_values.shift(1)
            # Allow signals when ADX > threshold OR ADX is rising (trend starting)
            trend_or_starting = (
                (adx_values.notna() & (adx_values > adx_threshold))
                | (adx_values.notna() & adx_prev.notna() & (adx_values > adx_prev))
            )
            buy_mask = buy_mask & trend_or_starting
            sell_mask = sell_mask & trend_or_starting

        # Vectorized strength: breakout magnitude / channel width
        channel_width = (prev_upper - prev_lower).replace(0, float("nan"))
        buy_strength = ((close - prev_upper) / channel_width).clip(0.0, 1.0)
        sell_strength = ((prev_lower - close) / channel_width).clip(0.0, 1.0)

        # Build results
        signals: list[SignalResult] = []
        timestamps = df.index

        for i in range(len(df)):
            ts = pd.Timestamp(timestamps[i])

            if buy_mask.iloc[i]:
                strength = float(buy_strength.iloc[i]) if pd.notna(buy_strength.iloc[i]) else 0.5
                signals.append(SignalResult(
                    signal=Signal.BUY, strength=strength,
                    strategy_name=self.name, asset=asset,
                    timeframe=timeframe, timestamp=ts,
                    explanation={
                        "upper": float(prev_upper.iloc[i]) if pd.notna(prev_upper.iloc[i]) else None,
                        "lower": float(prev_lower.iloc[i]) if pd.notna(prev_lower.iloc[i]) else None,
                        "mid": float(mid.iloc[i]) if pd.notna(mid.iloc[i]) else None,
                        "close": float(close.iloc[i]),
                        "volume": float(volume.iloc[i]),
                        "avg_volume": float(avg_volume.iloc[i]) if pd.notna(avg_volume.iloc[i]) else None,
                        "rsi": float(rsi_values.iloc[i]) if pd.notna(rsi_values.iloc[i]) else None,
                        "volume_mult": volume_mult,
                    },
                ))
            elif sell_mask.iloc[i]:
                strength = float(sell_strength.iloc[i]) if pd.notna(sell_strength.iloc[i]) else 0.5
                signals.append(SignalResult(
                    signal=Signal.SELL, strength=strength,
                    strategy_name=self.name, asset=asset,
                    timeframe=timeframe, timestamp=ts,
                    explanation={
                        "upper": float(prev_upper.iloc[i]) if pd.notna(prev_upper.iloc[i]) else None,
                        "lower": float(prev_lower.iloc[i]) if pd.notna(prev_lower.iloc[i]) else None,
                        "mid": float(mid.iloc[i]) if pd.notna(mid.iloc[i]) else None,
                        "close": float(close.iloc[i]),
                        "volume": float(volume.iloc[i]),
                        "avg_volume": float(avg_volume.iloc[i]) if pd.notna(avg_volume.iloc[i]) else None,
                        "rsi": float(rsi_values.iloc[i]) if pd.notna(rsi_values.iloc[i]) else None,
                        "volume_mult": volume_mult,
                    },
                ))
            else:
                signals.append(SignalResult(
                    signal=Signal.HOLD, strength=0.0,
                    strategy_name=self.name, asset=asset,
                    timeframe=timeframe, timestamp=ts,
                    explanation={},
                ))

        return signals
