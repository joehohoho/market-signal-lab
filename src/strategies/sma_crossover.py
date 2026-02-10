"""SMA Crossover strategy.

Generates BUY when the fast SMA crosses above the slow SMA (with an ATR
volatility filter), and SELL when the fast SMA crosses below.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators.core import atr as calc_atr
from indicators.core import sma
from strategies.base import Signal, SignalResult, Strategy

_DEFAULT_PARAMS: dict[str, Any] = {
    "fast_period": 10,
    "slow_period": 30,
    "atr_period": 14,
    "atr_filter_mult": 1.0,
}


class SMACrossoverStrategy(Strategy):
    """Simple Moving-Average crossover with ATR volatility filter.

    **BUY** when the fast SMA crosses *above* the slow SMA **and** the current
    ATR exceeds ``atr_filter_mult`` times its own rolling mean (confirming a
    volatile-enough market for trend-following).

    **SELL** when the fast SMA crosses *below* the slow SMA.

    **HOLD** otherwise.

    Strength is proportional to the normalised distance between the two
    moving averages.
    """

    @property
    def name(self) -> str:
        return "sma_crossover"

    def compute(
        self,
        df: pd.DataFrame,
        asset: str,
        timeframe: str,
        params: dict[str, Any],
    ) -> list[SignalResult]:
        p = {**_DEFAULT_PARAMS, **params}
        fast_period: int = int(p["fast_period"])
        slow_period: int = int(p["slow_period"])
        atr_period: int = int(p["atr_period"])
        atr_filter_mult: float = float(p["atr_filter_mult"])

        close: pd.Series = df["close"]
        fast_sma: pd.Series = sma(close, fast_period)
        slow_sma: pd.Series = sma(close, slow_period)

        atr_values: pd.Series = calc_atr(df["high"], df["low"], close, atr_period)
        avg_atr: pd.Series = atr_values.rolling(
            window=atr_period, min_periods=atr_period,
        ).mean()

        # Crossover detection: compare current vs previous relative positions.
        fast_above = (fast_sma > slow_sma).fillna(False)
        fast_above_prev = fast_above.shift(1).fillna(False)

        cross_up = fast_above & ~fast_above_prev
        cross_down = ~fast_above & fast_above_prev

        # Normalised MA spread for strength calculation.
        spread = (fast_sma - slow_sma).abs()
        max_spread = spread.rolling(window=slow_period, min_periods=1).max()
        normalised_spread = (spread / max_spread.replace(0, 1)).clip(0.0, 1.0)

        signals: list[SignalResult] = []
        for i in range(len(df)):
            ts = pd.Timestamp(df.index[i])
            f_sma = fast_sma.iloc[i]
            s_sma = slow_sma.iloc[i]
            atr_val = atr_values.iloc[i]
            avg_atr_val = avg_atr.iloc[i]
            strength = (
                float(normalised_spread.iloc[i])
                if pd.notna(normalised_spread.iloc[i])
                else 0.0
            )

            explanation: dict[str, Any] = {
                "fast_sma": float(f_sma) if pd.notna(f_sma) else None,
                "slow_sma": float(s_sma) if pd.notna(s_sma) else None,
                "atr": float(atr_val) if pd.notna(atr_val) else None,
                "avg_atr": float(avg_atr_val) if pd.notna(avg_atr_val) else None,
                "atr_filter_mult": atr_filter_mult,
                "crossover": None,
            }

            if (
                cross_up.iloc[i]
                and pd.notna(atr_val)
                and pd.notna(avg_atr_val)
                and atr_val > atr_filter_mult * avg_atr_val
            ):
                explanation["crossover"] = "bullish"
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
            elif cross_down.iloc[i]:
                explanation["crossover"] = "bearish"
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
