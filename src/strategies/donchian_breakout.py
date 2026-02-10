"""Donchian Channel Breakout strategy.

Generates BUY when the close breaks above the upper Donchian channel with
above-average volume, and SELL when it breaks below the lower channel.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators.core import donchian, rolling_volume_mean
from strategies.base import Signal, SignalResult, Strategy

_DEFAULT_PARAMS: dict[str, Any] = {
    "channel_period": 20,
    "volume_ma_period": 20,
    "volume_mult": 1.5,
}


class DonchianBreakoutStrategy(Strategy):
    """Classic Donchian channel breakout with volume filter.

    **BUY** when close breaks above the *previous* bar's upper channel **and**
    current volume exceeds ``volume_mult`` times the rolling average volume.

    **SELL** when close breaks below the *previous* bar's lower channel.

    **HOLD** otherwise.

    Strength is based on the magnitude of the breakout relative to the
    channel width.
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

        close: pd.Series = df["close"]
        volume: pd.Series = df["volume"]

        channels: dict[str, pd.Series] = donchian(
            df["high"], df["low"], channel_period,
        )
        upper: pd.Series = channels["upper"]
        lower: pd.Series = channels["lower"]
        mid: pd.Series = channels["mid"]

        avg_volume: pd.Series = rolling_volume_mean(volume, volume_ma_period)

        # Use previous bar's channel boundaries so the current bar's extremes
        # don't leak into its own reference level.
        prev_upper = upper.shift(1)
        prev_lower = lower.shift(1)

        signals: list[SignalResult] = []
        for i in range(len(df)):
            ts = pd.Timestamp(df.index[i])
            close_val = close.iloc[i]
            vol_val = volume.iloc[i]
            avg_vol_val = avg_volume.iloc[i]
            upper_val = prev_upper.iloc[i]
            lower_val = prev_lower.iloc[i]
            mid_val = mid.iloc[i]

            channel_width: float | None = (
                float(upper_val - lower_val)
                if (pd.notna(upper_val) and pd.notna(lower_val))
                else None
            )

            explanation: dict[str, Any] = {
                "upper": float(upper_val) if pd.notna(upper_val) else None,
                "lower": float(lower_val) if pd.notna(lower_val) else None,
                "mid": float(mid_val) if pd.notna(mid_val) else None,
                "close": float(close_val) if pd.notna(close_val) else None,
                "volume": float(vol_val) if pd.notna(vol_val) else None,
                "avg_volume": (
                    float(avg_vol_val) if pd.notna(avg_vol_val) else None
                ),
                "volume_mult": volume_mult,
            }

            if (
                pd.notna(upper_val)
                and pd.notna(avg_vol_val)
                and close_val > upper_val
                and vol_val > volume_mult * avg_vol_val
            ):
                # Strength: breakout magnitude relative to channel width.
                if channel_width and channel_width > 0:
                    breakout_magnitude = close_val - upper_val
                    strength = min(breakout_magnitude / channel_width, 1.0)
                else:
                    strength = 0.5
                signals.append(
                    SignalResult(
                        signal=Signal.BUY,
                        strength=strength,
                        strategy_name=self.name,
                        asset=asset,
                        timeframe=timeframe,
                        timestamp=ts,
                        explanation=explanation,
                    )
                )
            elif pd.notna(lower_val) and close_val < lower_val:
                # Strength: breakdown magnitude relative to channel width.
                if channel_width and channel_width > 0:
                    breakdown_magnitude = lower_val - close_val
                    strength = min(breakdown_magnitude / channel_width, 1.0)
                else:
                    strength = 0.5
                signals.append(
                    SignalResult(
                        signal=Signal.SELL,
                        strength=strength,
                        strategy_name=self.name,
                        asset=asset,
                        timeframe=timeframe,
                        timestamp=ts,
                        explanation=explanation,
                    )
                )
            else:
                signals.append(
                    SignalResult(
                        signal=Signal.HOLD,
                        strength=0.0,
                        strategy_name=self.name,
                        asset=asset,
                        timeframe=timeframe,
                        timestamp=ts,
                        explanation=explanation,
                    )
                )

        return signals
