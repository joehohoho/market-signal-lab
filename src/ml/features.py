"""Feature engineering for ML models.

All features use only past data (no lookahead bias).  The ``build_features``
function computes a set of technical features from an OHLCV DataFrame, and
``build_target`` creates a binary classification target based on forward
returns.
"""

from __future__ import annotations

import pandas as pd

from indicators.core import (
    atr,
    macd,
    rolling_volume_mean,
    rsi,
    sma,
    volatility,
)


def build_features(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    """Compute ML features from an OHLCV DataFrame.

    Every feature is computed using only data available at or before the
    current bar -- no future information leaks into the feature set.

    Args:
        df: OHLCV DataFrame with columns ``timestamp``, ``open``, ``high``,
            ``low``, ``close``, ``volume``.
        timeframe: Candle interval string (e.g. ``"1d"``, ``"1h"``).  When
            the timeframe is daily (``"1d"``), a ``day_of_week`` feature is
            added.

    Returns:
        A DataFrame containing feature columns plus ``timestamp`` and
        ``close``.  Rows with NaN values (from insufficient lookback) are
        dropped.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    features = pd.DataFrame(index=df.index)

    # Preserve timestamp and close for downstream use.
    features["timestamp"] = df["timestamp"]
    features["close"] = close

    # ---- Momentum --------------------------------------------------------
    features["rsi_14"] = rsi(close, period=14)

    macd_result = macd(close)
    features["macd_histogram"] = macd_result["histogram"]

    # ---- Volatility ------------------------------------------------------
    atr_14 = atr(high, low, close, period=14)
    features["atr_14_norm"] = atr_14 / close  # normalised by close price

    features["volatility_20"] = volatility(close, period=20)

    # ---- Trend (SMA ratios) ----------------------------------------------
    sma_20 = sma(close, period=20)
    sma_50 = sma(close, period=50)
    features["close_sma20_ratio"] = close / sma_20
    features["close_sma50_ratio"] = close / sma_50

    # ---- Volume ----------------------------------------------------------
    vol_mean_20 = rolling_volume_mean(volume, period=20)
    features["volume_ratio"] = volume / vol_mean_20

    # ---- Lagged returns --------------------------------------------------
    features["return_1"] = close.pct_change(1)
    features["return_5"] = close.pct_change(5)
    features["return_10"] = close.pct_change(10)

    # ---- Lagged volatility -----------------------------------------------
    features["volatility_5"] = volatility(close, period=5)

    # ---- Calendar --------------------------------------------------------
    if timeframe == "1d":
        ts = pd.to_datetime(df["timestamp"])
        features["day_of_week"] = ts.dt.dayofweek

    # Drop rows with any NaN (insufficient lookback at the start).
    features = features.dropna().reset_index(drop=True)

    return features


def build_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Build a binary classification target from forward returns.

    Args:
        df: DataFrame that must contain a ``close`` column.
        horizon: Number of bars to look forward for the return
            calculation.

    Returns:
        A :class:`pd.Series` named ``'target'`` with values ``1`` (positive
        forward return) or ``0`` (zero or negative).  The last ``horizon``
        rows will be NaN (no future data available).
    """
    forward_return = df["close"].shift(-horizon) / df["close"] - 1.0
    target = (forward_return > 0).astype(float)
    # Mark rows without enough forward data as NaN.
    target.iloc[-horizon:] = float("nan")
    target.name = "target"
    return target
