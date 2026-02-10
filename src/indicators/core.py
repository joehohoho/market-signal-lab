"""Technical indicators implemented as pure functions on pandas Series/DataFrames.

All indicators use only numpy and pandas -- no external TA libraries.
Each function accepts pandas Series input and returns pandas Series (or a dict
of Series for multi-output indicators).  NaN values are naturally produced at
the beginning of each output where insufficient lookback data exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Trend / Moving Averages
# ---------------------------------------------------------------------------


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    Args:
        series: Price or value series.
        period: Lookback window length.

    Returns:
        A Series of the rolling arithmetic mean.  The first ``period - 1``
        values will be NaN.
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average.

    Uses the standard EMA formula with ``span = period`` (decay factor
    ``alpha = 2 / (period + 1)``).

    Args:
        series: Price or value series.
        period: Span for the exponential weighting.

    Returns:
        A Series of the exponentially weighted moving average.  The first
        ``period - 1`` values will be NaN.
    """
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing method.

    Wilder's smoothing is equivalent to an EMA with ``alpha = 1 / period``
    (i.e. ``com = period - 1``).

    Args:
        series: Price series (typically close prices).
        period: Lookback period (default 14).

    Returns:
        RSI values between 0 and 100.  The first ``period`` values will be
        NaN.
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing: EMA with alpha = 1/period  =>  com = period - 1
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss is zero, RSI is 100 (all gains, no losses).
    result = result.where(avg_loss != 0, 100.0)

    # Ensure the first `period` values are NaN (insufficient lookback).
    result.iloc[:period] = np.nan

    return result


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """Moving Average Convergence Divergence.

    Args:
        series: Price series (typically close prices).
        fast:   Fast EMA period (default 12).
        slow:   Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        A dict with keys:
            - ``'macd'``:      MACD line (fast EMA - slow EMA).
            - ``'signal'``:    Signal line (EMA of the MACD line).
            - ``'histogram'``: MACD histogram (MACD - signal).
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range for a single period.

    TR = max(high - low,  |high - prev_close|,  |low - prev_close|)

    Args:
        high:  High price series.
        low:   Low price series.
        close: Close price series.

    Returns:
        True Range series.  The first value will be NaN because there is no
        previous close for the initial bar.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range using Wilder's smoothing.

    Args:
        high:   High price series.
        low:    Low price series.
        close:  Close price series.
        period: Smoothing period (default 14).

    Returns:
        ATR series.  The first ``period`` values will be NaN.
    """
    tr = true_range(high, low, close)
    return tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()


def volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling annualisation-free volatility (standard deviation of returns).

    Computes the rolling standard deviation of log returns over the given
    window.  To annualise, multiply the result by ``sqrt(252)`` for daily
    data.

    Args:
        close:  Close price series.
        period: Rolling window length (default 20).

    Returns:
        Rolling standard deviation of log returns.  The first ``period``
        values will be NaN.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=period, min_periods=period).std()


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_mult: float = 2.0,
) -> dict[str, pd.Series]:
    """Bollinger Bands.

    Args:
        close:    Close price series.
        period:   SMA lookback period (default 20).
        std_mult: Number of standard deviations for the bands (default 2.0).

    Returns:
        A dict with keys:
            - ``'upper'``: Upper band (mid + std_mult * rolling_std).
            - ``'mid'``:   Middle band (SMA).
            - ``'lower'``: Lower band (mid - std_mult * rolling_std).
    """
    mid = sma(close, period)
    rolling_std = close.rolling(window=period, min_periods=period).std()

    return {
        "upper": mid + std_mult * rolling_std,
        "mid": mid,
        "lower": mid - std_mult * rolling_std,
    }


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------


def donchian(
    high: pd.Series, low: pd.Series, period: int = 20
) -> dict[str, pd.Series]:
    """Donchian Channel.

    Args:
        high:   High price series.
        low:    Low price series.
        period: Lookback window (default 20).

    Returns:
        A dict with keys:
            - ``'upper'``: Rolling max of highs.
            - ``'lower'``: Rolling min of lows.
            - ``'mid'``:   Midpoint of upper and lower.
    """
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    mid = (upper + lower) / 2.0

    return {
        "upper": upper,
        "lower": lower,
        "mid": mid,
    }


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def rolling_volume_mean(volume: pd.Series, period: int = 20) -> pd.Series:
    """Rolling mean of volume.

    Args:
        volume: Volume series.
        period: Rolling window length (default 20).

    Returns:
        Rolling arithmetic mean of volume.  The first ``period - 1`` values
        will be NaN.
    """
    return volume.rolling(window=period, min_periods=period).mean()
