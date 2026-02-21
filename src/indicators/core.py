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


# ---------------------------------------------------------------------------
# Trend Strength
# ---------------------------------------------------------------------------


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> dict[str, pd.Series]:
    """Average Directional Index (ADX) with +DI / -DI components.

    ADX measures trend *strength* regardless of direction.  Values above 25
    indicate a strong trend; below 20 indicates ranging / choppy conditions.

    Args:
        high:   High price series.
        low:    Low price series.
        close:  Close price series.
        period: Smoothing period (default 14).

    Returns:
        A dict with keys ``'adx'``, ``'plus_di'``, ``'minus_di'``.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # Wilder's smoothing (EMA with alpha = 1/period)
    tr = true_range(high, low, close)
    atr_smooth = tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    # Directional Indicators
    plus_di = 100.0 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_smooth / atr_smooth.replace(0, np.nan)

    # DX and ADX
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx_val = dx.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    # NaN the warmup period
    adx_val.iloc[: 2 * period] = np.nan
    plus_di.iloc[:period] = np.nan
    minus_di.iloc[:period] = np.nan

    return {"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di}


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price (cumulative within session).

    Uses the typical price (H+L+C)/3 weighted by volume.  For daily data
    this gives the cumulative VWAP from the start of the series; for
    intraday data callers should reset at session boundaries.

    Args:
        high:   High price series.
        low:    Low price series.
        close:  Close price series.
        volume: Volume series.

    Returns:
        Cumulative VWAP series.
    """
    typical_price = (high + low + close) / 3.0
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def rolling_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Rolling VWAP over a fixed window.

    More useful than cumulative VWAP for daily charts since it doesn't
    anchor to the start of the series.

    Args:
        high:   High price series.
        low:    Low price series.
        close:  Close price series.
        volume: Volume series.
        period: Rolling window (default 20).

    Returns:
        Rolling VWAP series.
    """
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    return (
        tp_vol.rolling(window=period, min_periods=period).sum()
        / volume.rolling(window=period, min_periods=period).sum().replace(0, np.nan)
    )


# ---------------------------------------------------------------------------
# On-Balance Volume
# ---------------------------------------------------------------------------


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume.

    Cumulative sum of volume weighted by close-to-close direction.
    Rising OBV confirms uptrend; divergence from price warns of reversal.

    Args:
        close:  Close price series.
        volume: Volume series.

    Returns:
        OBV series.
    """
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (volume * direction).cumsum()


# ---------------------------------------------------------------------------
# Stochastic RSI
# ---------------------------------------------------------------------------


def stochastic_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> dict[str, pd.Series]:
    """Stochastic RSI (%K and %D).

    Applies the Stochastic oscillator formula to RSI values instead of
    price.  More responsive than standard RSI in fast-moving crypto markets.

    Args:
        close:       Close price series.
        rsi_period:  RSI lookback (default 14).
        stoch_period: Stochastic lookback on RSI values (default 14).
        smooth_k:    Smoothing for %K line (default 3).
        smooth_d:    Smoothing for %D line (default 3).

    Returns:
        A dict with keys ``'k'`` and ``'d'``, both in [0, 100].
    """
    rsi_vals = rsi(close, rsi_period)
    rsi_min = rsi_vals.rolling(window=stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi_vals.rolling(window=stoch_period, min_periods=stoch_period).max()
    rsi_range = (rsi_max - rsi_min).replace(0, np.nan)

    stoch = (rsi_vals - rsi_min) / rsi_range * 100.0
    k = stoch.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=smooth_d, min_periods=smooth_d).mean()

    return {"k": k, "d": d}


# ---------------------------------------------------------------------------
# Chaikin Money Flow
# ---------------------------------------------------------------------------


def cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Chaikin Money Flow.

    Measures buying/selling pressure over a rolling window.  Positive CMF
    indicates accumulation, negative indicates distribution.

    Args:
        high:   High price series.
        low:    Low price series.
        close:  Close price series.
        volume: Volume series.
        period: Rolling window (default 20).

    Returns:
        CMF series in [-1, 1].
    """
    hl_range = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    mfm = mfm.fillna(0.0)
    mf_volume = mfm * volume
    return (
        mf_volume.rolling(window=period, min_periods=period).sum()
        / volume.rolling(window=period, min_periods=period).sum().replace(0, np.nan)
    )


# ---------------------------------------------------------------------------
# Bollinger Band Width & %B
# ---------------------------------------------------------------------------


def bollinger_bandwidth(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    """Bollinger Band Width (squeeze detection).

    Low bandwidth indicates a volatility squeeze — BTC often explodes
    after periods of low bandwidth.

    Args:
        close:    Close price series.
        period:   SMA lookback (default 20).
        std_mult: Standard deviation multiplier (default 2.0).

    Returns:
        Bandwidth as (upper - lower) / mid.
    """
    bands = bollinger_bands(close, period, std_mult)
    mid = bands["mid"].replace(0, np.nan)
    return (bands["upper"] - bands["lower"]) / mid


def bollinger_pctb(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    """Bollinger %B — position of close within the bands.

    0 = at lower band, 1 = at upper band, 0.5 = at midline.
    Values outside [0, 1] indicate breakouts.

    Args:
        close:    Close price series.
        period:   SMA lookback (default 20).
        std_mult: Standard deviation multiplier (default 2.0).

    Returns:
        %B series.
    """
    bands = bollinger_bands(close, period, std_mult)
    band_range = (bands["upper"] - bands["lower"]).replace(0, np.nan)
    return (close - bands["lower"]) / band_range
