"""RSI Mean Reversion strategy.

Generates BUY when RSI *recovers* from oversold (crosses back above
threshold) in an uptrend, and SELL when RSI *drops* from overbought
(crosses back below threshold) in a downtrend.  Both directions are
filtered by trend to avoid counter-trend entries.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators.core import bollinger_bands
from indicators.core import rsi as compute_rsi
from indicators.core import sma
from strategies.base import Signal, SignalResult, Strategy

_DEFAULT_PARAMS: dict[str, Any] = {
    "rsi_period": 14,
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    "trend_ma_period": 200,
}


class RSIMeanReversionStrategy(Strategy):
    """Buy oversold recoveries in uptrends, sell overbought reversals in downtrends.

    **BUY** when RSI *crosses back above* the oversold threshold (recovery
    confirmation) **and** price is above the trend moving average **and**
    price is near the lower Bollinger Band.

    **SELL** when RSI *crosses back below* the overbought threshold (reversal
    confirmation) **and** price is below the trend moving average.

    **HOLD** otherwise.

    Waiting for the RSI to cross *back through* the threshold confirms
    that momentum is actually reversing, avoiding "catching a falling knife."
    """

    @property
    def name(self) -> str:
        return "rsi_mean_reversion"

    def compute(
        self,
        df: pd.DataFrame,
        asset: str,
        timeframe: str,
        params: dict[str, Any],
    ) -> list[SignalResult]:
        p = {**_DEFAULT_PARAMS, **params}
        rsi_period: int = int(p["rsi_period"])
        rsi_oversold: float = float(p["rsi_oversold"])
        rsi_overbought: float = float(p["rsi_overbought"])
        trend_ma_period: int = int(p["trend_ma_period"])

        close: pd.Series = df["close"]
        rsi_values: pd.Series = compute_rsi(close, rsi_period)
        trend_ma: pd.Series = sma(close, trend_ma_period)

        # Bollinger Bands for mean-reversion proximity check
        bb = bollinger_bands(close, period=20, num_std=2.0)
        bb_lower: pd.Series = bb["lower"]

        # RSI crossover detection (recovery / reversal)
        rsi_below_oversold = (rsi_values < rsi_oversold).fillna(False)
        rsi_above_overbought = (rsi_values > rsi_overbought).fillna(False)
        prev_below = rsi_below_oversold.shift(1).fillna(False)
        prev_above = rsi_above_overbought.shift(1).fillna(False)

        # Recovery: was below oversold, now above it
        rsi_recovery = prev_below & ~rsi_below_oversold
        # Reversal: was above overbought, now below it
        rsi_reversal = prev_above & ~rsi_above_overbought

        signals: list[SignalResult] = []
        for i in range(len(df)):
            ts = pd.Timestamp(df.index[i])
            rsi_val = rsi_values.iloc[i]
            ma_val = trend_ma.iloc[i]
            close_val = close.iloc[i]
            bb_low_val = bb_lower.iloc[i]

            explanation: dict[str, Any] = {
                "rsi": float(rsi_val) if pd.notna(rsi_val) else None,
                "trend_ma": float(ma_val) if pd.notna(ma_val) else None,
                "close": float(close_val) if pd.notna(close_val) else None,
                "bb_lower": float(bb_low_val) if pd.notna(bb_low_val) else None,
                "rsi_oversold": rsi_oversold,
                "rsi_overbought": rsi_overbought,
            }

            if (
                rsi_recovery.iloc[i]
                and pd.notna(rsi_val)
                and pd.notna(ma_val)
                and close_val > ma_val
                and pd.notna(bb_low_val)
                and close_val < bb_low_val * 1.02
            ):
                distance = rsi_oversold - rsi_val
                strength = min(abs(distance) / rsi_oversold, 1.0)
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
            elif (
                rsi_reversal.iloc[i]
                and pd.notna(rsi_val)
                and pd.notna(ma_val)
                and close_val < ma_val
            ):
                distance = rsi_val - rsi_overbought
                max_distance = 100.0 - rsi_overbought
                strength = (
                    min(abs(distance) / max_distance, 1.0)
                    if max_distance > 0
                    else 1.0
                )
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
