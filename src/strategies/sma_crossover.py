"""SMA Crossover strategy with ADX regime filtering.

Generates BUY when the fast SMA crosses above the slow SMA with MACD
momentum confirmation and trending regime (ADX > threshold).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators.core import adx as calc_adx
from indicators.core import atr as calc_atr
from indicators.core import macd as calc_macd
from indicators.core import sma
from strategies.base import Signal, SignalResult, Strategy

_DEFAULT_PARAMS: dict[str, Any] = {
    "fast_period": 10,
    "slow_period": 30,
    "atr_period": 14,
    "atr_filter_mult": 1.0,
    "regime_filter": True,
    "adx_threshold": 20.0,
}


class SMACrossoverStrategy(Strategy):
    """Moving-Average crossover with MACD momentum and regime confirmation.

    **BUY** when the fast SMA crosses *above* the slow SMA **and** the MACD
    histogram is positive **and** ADX > threshold (market is trending).

    **SELL** when the fast SMA crosses *below* the slow SMA **and** the MACD
    histogram is negative **and** ADX > threshold.

    **HOLD** otherwise.
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
        regime_filter: bool = bool(p["regime_filter"])
        adx_threshold: float = float(p["adx_threshold"])

        close: pd.Series = df["close"]
        fast_sma: pd.Series = sma(close, fast_period)
        slow_sma: pd.Series = sma(close, slow_period)

        atr_values: pd.Series = calc_atr(df["high"], df["low"], close, atr_period)

        macd_data = calc_macd(close)
        macd_hist: pd.Series = macd_data["histogram"]

        # Vectorized crossover detection
        fast_above = (fast_sma > slow_sma).fillna(False)
        fast_above_prev = fast_above.shift(1).fillna(False)
        cross_up = fast_above & ~fast_above_prev
        cross_down = ~fast_above & fast_above_prev

        # ATR volatility filter: only signal when ATR is above
        # atr_filter_mult × its rolling mean (filters low-volatility chop).
        # Window matches slow_period so the baseline is contextually relevant.
        atr_rolling_mean: pd.Series = atr_values.rolling(
            window=slow_period, min_periods=atr_period
        ).mean()
        atr_filter_mask: pd.Series = (
            atr_values.notna()
            & atr_rolling_mean.notna()
            & (atr_values > atr_filter_mult * atr_rolling_mean)
        )

        # Vectorized signal masks
        buy_mask = cross_up & macd_hist.notna() & (macd_hist > 0) & atr_filter_mask
        sell_mask = cross_down & macd_hist.notna() & (macd_hist < 0) & atr_filter_mask

        # Regime filter: only signal in trending markets
        if regime_filter:
            adx_data = calc_adx(df["high"], df["low"], close, period=14)
            adx_values = adx_data["adx"]
            trend_mask = adx_values.notna() & (adx_values > adx_threshold)
            buy_mask = buy_mask & trend_mask
            sell_mask = sell_mask & trend_mask

        # Vectorized strength
        spread = (fast_sma - slow_sma).abs()
        max_spread = spread.rolling(window=slow_period, min_periods=1).max()
        normalised_spread = (spread / max_spread.replace(0, 1)).clip(0.0, 1.0)

        # Build results
        signals: list[SignalResult] = []
        timestamps = df.index

        for i in range(len(df)):
            ts = pd.Timestamp(timestamps[i])

            if buy_mask.iloc[i]:
                strength = float(normalised_spread.iloc[i]) if pd.notna(normalised_spread.iloc[i]) else 0.0
                signals.append(SignalResult(
                    signal=Signal.BUY, strength=strength,
                    strategy_name=self.name, asset=asset,
                    timeframe=timeframe, timestamp=ts,
                    explanation={
                        "fast_sma": float(fast_sma.iloc[i]) if pd.notna(fast_sma.iloc[i]) else None,
                        "slow_sma": float(slow_sma.iloc[i]) if pd.notna(slow_sma.iloc[i]) else None,
                        "macd_hist": float(macd_hist.iloc[i]) if pd.notna(macd_hist.iloc[i]) else None,
                        "atr": float(atr_values.iloc[i]) if pd.notna(atr_values.iloc[i]) else None,
                        "atr_rolling_mean": float(atr_rolling_mean.iloc[i]) if pd.notna(atr_rolling_mean.iloc[i]) else None,
                        "atr_filter_mult": atr_filter_mult,
                        "crossover": "bullish",
                    },
                ))
            elif sell_mask.iloc[i]:
                strength = float(normalised_spread.iloc[i]) if pd.notna(normalised_spread.iloc[i]) else 0.0
                signals.append(SignalResult(
                    signal=Signal.SELL, strength=strength,
                    strategy_name=self.name, asset=asset,
                    timeframe=timeframe, timestamp=ts,
                    explanation={
                        "fast_sma": float(fast_sma.iloc[i]) if pd.notna(fast_sma.iloc[i]) else None,
                        "slow_sma": float(slow_sma.iloc[i]) if pd.notna(slow_sma.iloc[i]) else None,
                        "macd_hist": float(macd_hist.iloc[i]) if pd.notna(macd_hist.iloc[i]) else None,
                        "atr": float(atr_values.iloc[i]) if pd.notna(atr_values.iloc[i]) else None,
                        "atr_rolling_mean": float(atr_rolling_mean.iloc[i]) if pd.notna(atr_rolling_mean.iloc[i]) else None,
                        "atr_filter_mult": atr_filter_mult,
                        "crossover": "bearish",
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
