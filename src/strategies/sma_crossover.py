"""SMA Crossover strategy.

Generates BUY when the fast SMA crosses above the slow SMA with ATR
volatility filter and MACD confirmation.  SELL requires the crossover
plus ATR confirmation (symmetric filtering).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators.core import atr as calc_atr
from indicators.core import macd as calc_macd
from indicators.core import rsi as calc_rsi
from indicators.core import sma
from strategies.base import Signal, SignalResult, Strategy

_DEFAULT_PARAMS: dict[str, Any] = {
    "fast_period": 10,
    "slow_period": 30,
    "atr_period": 14,
    "atr_filter_mult": 1.0,
}


class SMACrossoverStrategy(Strategy):
    """Moving-Average crossover with MACD confirmation and symmetric filters.

    **BUY** when the fast SMA crosses *above* the slow SMA **and**:
    - ATR exceeds ``atr_filter_mult`` times its rolling mean (volatility), *and*
    - MACD histogram is positive (momentum confirms the crossover), *and*
    - RSI is below 70 (not already overbought).

    **SELL** when the fast SMA crosses *below* the slow SMA **and**:
    - ATR exceeds ``atr_filter_mult`` times its rolling mean (volatility), *and*
    - MACD histogram is negative (momentum confirms the breakdown).

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

        # MACD for momentum confirmation
        macd_data = calc_macd(close)
        macd_hist: pd.Series = macd_data["histogram"]

        # RSI to avoid buying into overbought conditions
        rsi_values: pd.Series = calc_rsi(close, 14)

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
            hist_val = macd_hist.iloc[i]
            rsi_val = rsi_values.iloc[i]
            strength = (
                float(normalised_spread.iloc[i])
                if pd.notna(normalised_spread.iloc[i])
                else 0.0
            )

            atr_ok = (
                pd.notna(atr_val)
                and pd.notna(avg_atr_val)
                and atr_val > atr_filter_mult * avg_atr_val
            )

            explanation: dict[str, Any] = {
                "fast_sma": float(f_sma) if pd.notna(f_sma) else None,
                "slow_sma": float(s_sma) if pd.notna(s_sma) else None,
                "atr": float(atr_val) if pd.notna(atr_val) else None,
                "avg_atr": float(avg_atr_val) if pd.notna(avg_atr_val) else None,
                "macd_hist": float(hist_val) if pd.notna(hist_val) else None,
                "rsi": float(rsi_val) if pd.notna(rsi_val) else None,
                "atr_filter_mult": atr_filter_mult,
                "crossover": None,
            }

            if (
                cross_up.iloc[i]
                and atr_ok
                and pd.notna(hist_val) and hist_val > 0
                and (pd.isna(rsi_val) or rsi_val < 70)
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
            elif (
                cross_down.iloc[i]
                and atr_ok
                and pd.notna(hist_val) and hist_val < 0
            ):
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
