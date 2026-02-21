"""Feature engineering for ML models.

All features use only past data (no lookahead bias).  The ``build_features``
function computes a set of technical features from an OHLCV DataFrame, and
``build_target`` creates classification targets based on forward returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.core import (
    adx,
    atr,
    bollinger_bandwidth,
    bollinger_pctb,
    cmf,
    macd,
    obv,
    rolling_volume_mean,
    rolling_vwap,
    rsi,
    sma,
    stochastic_rsi,
    volatility,
)


def build_features(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    """Compute ML features from an OHLCV DataFrame.

    Every feature is computed using only data available at or before the
    current bar -- no future information leaks into the feature set.

    Args:
        df: OHLCV DataFrame with columns ``timestamp``, ``open``, ``high``,
            ``low``, ``close``, ``volume``.
        timeframe: Candle interval string (e.g. ``"1d"``, ``"1h"``).

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

    stoch_rsi = stochastic_rsi(close, rsi_period=14, stoch_period=14)
    features["stoch_rsi_k"] = stoch_rsi["k"]
    features["stoch_rsi_d"] = stoch_rsi["d"]

    # ---- Trend Strength --------------------------------------------------
    adx_result = adx(high, low, close, period=14)
    features["adx"] = adx_result["adx"]
    features["plus_di"] = adx_result["plus_di"]
    features["minus_di"] = adx_result["minus_di"]

    # ---- Volatility ------------------------------------------------------
    atr_14 = atr(high, low, close, period=14)
    features["atr_14_norm"] = atr_14 / close  # normalised by close price

    features["volatility_20"] = volatility(close, period=20)
    features["volatility_5"] = volatility(close, period=5)

    features["bb_width"] = bollinger_bandwidth(close, period=20)
    features["bb_pctb"] = bollinger_pctb(close, period=20)

    # ---- Trend (SMA ratios) ----------------------------------------------
    sma_20 = sma(close, period=20)
    sma_50 = sma(close, period=50)
    sma_200 = sma(close, period=200)
    features["close_sma20_ratio"] = close / sma_20
    features["close_sma50_ratio"] = close / sma_50
    features["close_sma200_ratio"] = close / sma_200

    # ---- Volume ----------------------------------------------------------
    vol_mean_20 = rolling_volume_mean(volume, period=20)
    features["volume_ratio"] = volume / vol_mean_20

    # Volume trend: short-term vs long-term
    vol_mean_5 = rolling_volume_mean(volume, period=5)
    features["volume_trend"] = vol_mean_5 / vol_mean_20

    # ---- On-Balance Volume -----------------------------------------------
    obv_series = obv(close, volume)
    obv_sma = sma(obv_series, period=20)
    # OBV ratio to its own moving average (normalised)
    features["obv_ratio"] = obv_series / obv_sma.replace(0, np.nan)

    # ---- Chaikin Money Flow ----------------------------------------------
    features["cmf_20"] = cmf(high, low, close, volume, period=20)

    # ---- VWAP deviation --------------------------------------------------
    rvwap = rolling_vwap(high, low, close, volume, period=20)
    features["vwap_deviation"] = (close - rvwap) / rvwap

    # ---- Lagged returns --------------------------------------------------
    features["return_1"] = close.pct_change(1)
    features["return_5"] = close.pct_change(5)
    features["return_10"] = close.pct_change(10)
    features["return_20"] = close.pct_change(20)
    features["return_60"] = close.pct_change(60)

    # ---- Price distance from recent extremes -----------------------------
    features["dist_from_20d_high"] = close / high.rolling(20).max() - 1.0
    features["dist_from_20d_low"] = close / low.rolling(20).min() - 1.0

    # ---- Range / ATR ratio (intraday volatility regime) ------------------
    features["range_atr_ratio"] = (high - low) / atr_14.replace(0, np.nan)

    # ---- Calendar --------------------------------------------------------
    if timeframe == "1d":
        ts = pd.to_datetime(df["timestamp"])
        features["day_of_week"] = ts.dt.dayofweek

    # ---- Alternative data (if present in df) -----------------------------
    if "fear_greed" in df.columns:
        features["fear_greed"] = df["fear_greed"]
    if "funding_rate" in df.columns:
        features["funding_rate"] = df["funding_rate"]

    # Drop rows with any NaN (insufficient lookback at the start).
    features = features.dropna().reset_index(drop=True)

    return features


def build_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Build a binary classification target from forward returns.

    Args:
        df: DataFrame that must contain a ``close`` column.
        horizon: Number of bars to look forward.

    Returns:
        A Series named ``'target'`` with 1 (positive) or 0 (negative/zero).
    """
    forward_return = df["close"].shift(-horizon) / df["close"] - 1.0
    target = (forward_return > 0).astype(float)
    target.iloc[-horizon:] = float("nan")
    target.name = "target"
    return target


def build_target_atr_scaled(
    df: pd.DataFrame,
    horizon: int = 5,
    atr_threshold: float = 0.5,
) -> pd.Series:
    """Build an ATR-scaled ternary target.

    Instead of raw up/down, classifies moves relative to volatility:
    - 1.0: forward return > atr_threshold * normalised ATR (strong move up)
    - 0.0: forward return < -atr_threshold * normalised ATR (strong move down)
    - 0.5: within noise band (not a meaningful move)

    This is better for BTC because a 1% move is noise in high-vol and
    significant in low-vol.

    Args:
        df: DataFrame with ``close``, ``high``, ``low`` columns.
        horizon: Forward-looking window in bars.
        atr_threshold: Multiplier of normalised ATR for the significance
            threshold.

    Returns:
        Series with values 0.0, 0.5, or 1.0.  Last *horizon* rows are NaN.
    """
    close = df["close"]
    atr_14 = atr(df["high"], df["low"], close, period=14)
    atr_norm = atr_14 / close  # normalised ATR (fraction of price)

    forward_return = close.shift(-horizon) / close - 1.0
    threshold = atr_threshold * atr_norm

    target = pd.Series(0.5, index=df.index, name="target")  # neutral default
    target[forward_return > threshold] = 1.0   # strong up
    target[forward_return < -threshold] = 0.0  # strong down
    target.iloc[-horizon:] = float("nan")

    return target
