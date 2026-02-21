"""RSI Mean Reversion strategy with ADX regime filtering.

Generates BUY when RSI *recovers* from oversold in an uptrend and
ranging/weak-trend regime (ADX < threshold).  SELL when RSI *drops*
from overbought.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators.core import adx as calc_adx
from indicators.core import rsi as compute_rsi
from indicators.core import sma
from strategies.base import Signal, SignalResult, Strategy

_DEFAULT_PARAMS: dict[str, Any] = {
    "rsi_period": 14,
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    "trend_ma_period": 200,
    "regime_filter": True,
    "adx_threshold": 25.0,
}


class RSIMeanReversionStrategy(Strategy):
    """Buy oversold recoveries in low-ADX (ranging) markets.

    **BUY** when RSI *crosses back above* the oversold threshold **and**
    price is above the trend moving average **and** ADX < threshold
    (market is ranging — mean reversion works best here).

    **SELL** when RSI *crosses back below* the overbought threshold.

    **HOLD** otherwise.
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
        regime_filter: bool = bool(p["regime_filter"])
        adx_threshold: float = float(p["adx_threshold"])

        close: pd.Series = df["close"]
        rsi_values: pd.Series = compute_rsi(close, rsi_period)
        trend_ma: pd.Series = sma(close, trend_ma_period)

        # Vectorized RSI crossover detection
        rsi_below_oversold = (rsi_values < rsi_oversold).fillna(False)
        rsi_above_overbought = (rsi_values > rsi_overbought).fillna(False)
        prev_below = rsi_below_oversold.shift(1).fillna(False)
        prev_above = rsi_above_overbought.shift(1).fillna(False)

        rsi_recovery = prev_below & ~rsi_below_oversold
        rsi_reversal = prev_above & ~rsi_above_overbought

        # Vectorized signal masks
        buy_mask = (
            rsi_recovery
            & rsi_values.notna()
            & trend_ma.notna()
            & (close > trend_ma)
        )
        sell_mask = rsi_reversal & rsi_values.notna()

        # Regime filter: mean reversion works in ranging markets
        if regime_filter:
            adx_data = calc_adx(df["high"], df["low"], close, period=14)
            adx_values = adx_data["adx"]
            range_mask = adx_values.notna() & (adx_values < adx_threshold)
            buy_mask = buy_mask & range_mask
            # Sell signals don't need regime filter (exit is time-sensitive)

        # Vectorized strength computation
        buy_strength = ((rsi_oversold - rsi_values).abs() / rsi_oversold).clip(0.0, 1.0)
        sell_strength_max = 100.0 - rsi_overbought
        sell_strength = (
            ((rsi_values - rsi_overbought).abs() / sell_strength_max).clip(0.0, 1.0)
            if sell_strength_max > 0
            else pd.Series(1.0, index=df.index)
        )

        # Build results
        signals: list[SignalResult] = []
        timestamps = df.index

        for i in range(len(df)):
            ts = pd.Timestamp(timestamps[i])

            if buy_mask.iloc[i]:
                signals.append(SignalResult(
                    signal=Signal.BUY,
                    strength=float(buy_strength.iloc[i]),
                    strategy_name=self.name, asset=asset,
                    timeframe=timeframe, timestamp=ts,
                    explanation={
                        "rsi": float(rsi_values.iloc[i]) if pd.notna(rsi_values.iloc[i]) else None,
                        "trend_ma": float(trend_ma.iloc[i]) if pd.notna(trend_ma.iloc[i]) else None,
                        "close": float(close.iloc[i]),
                        "rsi_oversold": rsi_oversold,
                        "rsi_overbought": rsi_overbought,
                    },
                ))
            elif sell_mask.iloc[i]:
                signals.append(SignalResult(
                    signal=Signal.SELL,
                    strength=float(sell_strength.iloc[i]),
                    strategy_name=self.name, asset=asset,
                    timeframe=timeframe, timestamp=ts,
                    explanation={
                        "rsi": float(rsi_values.iloc[i]) if pd.notna(rsi_values.iloc[i]) else None,
                        "trend_ma": float(trend_ma.iloc[i]) if pd.notna(trend_ma.iloc[i]) else None,
                        "close": float(close.iloc[i]),
                        "rsi_oversold": rsi_oversold,
                        "rsi_overbought": rsi_overbought,
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
