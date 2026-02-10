"""RSI Mean Reversion strategy.

Generates BUY when RSI dips below the oversold threshold (and price is above
the long-term trend MA), and SELL when RSI rises above the overbought
threshold.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

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
    """Buy oversold dips in an uptrend, sell overbought conditions.

    **BUY** when RSI drops below the oversold threshold **and** price is above
    the trend moving average (buying the dip in an uptrend).

    **SELL** when RSI rises above the overbought threshold.

    **HOLD** otherwise.

    Strength is proportional to how far RSI has pushed past its trigger
    threshold, normalised to ``[0, 1]``.
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

        signals: list[SignalResult] = []
        for i in range(len(df)):
            ts = pd.Timestamp(df.index[i])
            rsi_val = rsi_values.iloc[i]
            ma_val = trend_ma.iloc[i]
            close_val = close.iloc[i]

            explanation: dict[str, Any] = {
                "rsi": float(rsi_val) if pd.notna(rsi_val) else None,
                "trend_ma": float(ma_val) if pd.notna(ma_val) else None,
                "close": float(close_val) if pd.notna(close_val) else None,
                "rsi_oversold": rsi_oversold,
                "rsi_overbought": rsi_overbought,
            }

            if (
                pd.notna(rsi_val)
                and pd.notna(ma_val)
                and rsi_val < rsi_oversold
                and close_val > ma_val
            ):
                # How far below oversold RSI is, normalised.
                # E.g. RSI=10 with oversold=30 -> distance=20, max ~30 -> ~0.67
                distance = rsi_oversold - rsi_val
                strength = min(distance / rsi_oversold, 1.0)
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
            elif pd.notna(rsi_val) and rsi_val > rsi_overbought:
                # How far above overbought RSI is, normalised.
                distance = rsi_val - rsi_overbought
                max_distance = 100.0 - rsi_overbought
                strength = (
                    min(distance / max_distance, 1.0)
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
