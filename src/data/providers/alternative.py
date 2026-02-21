"""Alternative data providers for BTC market analysis.

Fetches non-price data that is highly predictive for crypto markets:
- Fear & Greed Index (alternative.me API)
- Funding rates (Binance public API)
- BTC dominance (CoinGecko public API)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Fear & Greed Index
# ---------------------------------------------------------------------------


def fetch_fear_greed(limit: int = 365) -> pd.DataFrame:
    """Fetch Bitcoin Fear & Greed Index from alternative.me.

    The index ranges from 0 (Extreme Fear) to 100 (Extreme Greed).
    Strong contrarian signal: buy at Extreme Fear, caution at Extreme Greed.

    Args:
        limit: Number of daily readings to fetch (max ~365 available).

    Returns:
        DataFrame with columns ``timestamp``, ``value`` (0-100),
        ``classification`` (e.g. "Extreme Fear", "Greed").
        Empty DataFrame on failure.
    """
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        logger.exception("Failed to fetch Fear & Greed Index")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    rows = []
    for entry in data:
        ts = datetime.fromtimestamp(int(entry["timestamp"]), tz=timezone.utc)
        rows.append(
            {
                "timestamp": ts,
                "value": int(entry["value"]),
                "classification": entry.get("value_classification", ""),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Binance Funding Rates
# ---------------------------------------------------------------------------

_BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_funding_rates(
    symbol: str = "BTCUSDT",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch perpetual futures funding rates from Binance.

    Positive funding = longs pay shorts (bullish crowding).
    Negative funding = shorts pay longs (bearish crowding).
    Extreme values are strong contrarian signals.

    Args:
        symbol: Binance futures symbol (default ``"BTCUSDT"``).
        limit:  Number of 8-hour readings (max 1000).

    Returns:
        DataFrame with columns ``timestamp``, ``funding_rate`` (decimal),
        ``symbol``.  Empty DataFrame on failure.
    """
    params: dict[str, Any] = {"symbol": symbol, "limit": min(limit, 1000)}
    try:
        resp = httpx.get(_BINANCE_FUNDING_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception("Failed to fetch funding rates for %s", symbol)
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    rows = []
    for entry in data:
        ts = datetime.fromtimestamp(int(entry["fundingTime"]) / 1000, tz=timezone.utc)
        rows.append(
            {
                "timestamp": ts,
                "funding_rate": float(entry["fundingRate"]),
                "symbol": entry["symbol"],
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def resample_funding_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 8-hourly funding rates to daily averages.

    Args:
        df: Funding rate DataFrame from :func:`fetch_funding_rates`.

    Returns:
        Daily DataFrame with ``timestamp`` and ``funding_rate`` (daily avg).
    """
    if df.empty:
        return df
    df = df.set_index("timestamp")
    daily = df["funding_rate"].resample("1D").mean().reset_index()
    daily.columns = ["timestamp", "funding_rate"]
    return daily.dropna()


# ---------------------------------------------------------------------------
# BTC Dominance (CoinGecko)
# ---------------------------------------------------------------------------

_COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"


def fetch_btc_dominance() -> dict[str, Any]:
    """Fetch current BTC market dominance from CoinGecko.

    Rising dominance = risk-off (money flowing to BTC from alts).
    Falling dominance = risk-on (altcoin season).

    Returns:
        Dict with ``btc_dominance`` (float, 0-100),
        ``total_market_cap_usd`` (float), ``timestamp`` (datetime).
        Empty dict on failure.
    """
    try:
        resp = httpx.get(_COINGECKO_GLOBAL_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("data", {})
    except Exception:
        logger.exception("Failed to fetch BTC dominance")
        return {}

    market_cap_pct = data.get("market_cap_percentage", {})
    total_market_cap = data.get("total_market_cap", {})

    return {
        "btc_dominance": market_cap_pct.get("btc", 0.0),
        "total_market_cap_usd": total_market_cap.get("usd", 0.0),
        "timestamp": datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# Convenience: merge alternative data with OHLCV
# ---------------------------------------------------------------------------


def merge_fear_greed_with_ohlcv(
    ohlcv: pd.DataFrame,
    fg_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join Fear & Greed data onto an OHLCV DataFrame by date.

    If *fg_df* is None, fetches fresh data.  Adds ``fear_greed`` column.

    Args:
        ohlcv: OHLCV DataFrame with ``timestamp`` column.
        fg_df: Pre-fetched Fear & Greed data, or None to fetch.

    Returns:
        OHLCV DataFrame with ``fear_greed`` column added (NaN where missing).
    """
    if fg_df is None:
        fg_df = fetch_fear_greed()

    if fg_df.empty:
        ohlcv["fear_greed"] = float("nan")
        return ohlcv

    # Align by date
    fg_df = fg_df.copy()
    fg_df["date"] = pd.to_datetime(fg_df["timestamp"]).dt.date

    ohlcv = ohlcv.copy()
    ts_col = "timestamp"
    ohlcv["_date"] = pd.to_datetime(ohlcv[ts_col]).dt.date

    fg_map = dict(zip(fg_df["date"], fg_df["value"]))
    ohlcv["fear_greed"] = ohlcv["_date"].map(fg_map).astype(float)
    ohlcv = ohlcv.drop(columns=["_date"])

    return ohlcv


def merge_funding_with_ohlcv(
    ohlcv: pd.DataFrame,
    funding_df: pd.DataFrame | None = None,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """Left-join daily funding rate onto an OHLCV DataFrame.

    Args:
        ohlcv: OHLCV DataFrame with ``timestamp`` column.
        funding_df: Pre-fetched daily funding data, or None to fetch + resample.
        symbol: Binance futures symbol if fetching.

    Returns:
        OHLCV DataFrame with ``funding_rate`` column added.
    """
    if funding_df is None:
        raw = fetch_funding_rates(symbol=symbol)
        funding_df = resample_funding_daily(raw)

    if funding_df.empty:
        ohlcv["funding_rate"] = float("nan")
        return ohlcv

    funding_df = funding_df.copy()
    funding_df["date"] = pd.to_datetime(funding_df["timestamp"]).dt.date

    ohlcv = ohlcv.copy()
    ohlcv["_date"] = pd.to_datetime(ohlcv["timestamp"]).dt.date

    fr_map = dict(zip(funding_df["date"], funding_df["funding_rate"]))
    ohlcv["funding_rate"] = ohlcv["_date"].map(fr_map).astype(float)
    ohlcv = ohlcv.drop(columns=["_date"])

    return ohlcv
