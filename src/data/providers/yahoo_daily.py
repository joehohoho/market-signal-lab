"""Yahoo Finance public chart API provider for daily equity OHLCV data."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.logging import get_logger
from data.providers.base import DataProvider

logger = get_logger(__name__)

_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Yahoo interval parameter values for supported timeframes.
_INTERVAL_MAP: dict[str, str] = {
    "1d": "1d",
    "1wk": "1wk",
}

_RATE_LIMIT_SECONDS = 1.0

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class YahooDailyProvider(DataProvider):
    """Fetch daily (or weekly) OHLCV candles from the Yahoo Finance chart API.

    This provider works with US equities (``AAPL``), Canadian stocks
    (``SHOP.TO``), ETFs, indices, and other Yahoo-supported tickers.
    No authentication is required.

    Only ``1d`` and ``1wk`` intervals are supported.
    """

    def __init__(self) -> None:
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    def provider_name(self) -> str:
        return "yahoo_daily"

    def supported_timeframes(self) -> list[str]:
        return list(_INTERVAL_MAP.keys())

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        self._validate_timeframe(timeframe)

        interval = _INTERVAL_MAP[timeframe]
        period1 = self._to_unix(start)
        period2 = self._to_unix(end)

        logger.info(
            "Yahoo: fetching %s %s from %s to %s",
            symbol,
            timeframe,
            start.isoformat(),
            end.isoformat(),
        )

        data = self._request_chart(symbol, interval, period1, period2)
        if data is None:
            logger.warning("Yahoo: no data returned for %s", symbol)
            return self._make_dataframe([])

        rows = self._parse_chart_response(data)

        logger.info("Yahoo: %d candles for %s %s", len(rows), symbol, timeframe)
        return self._make_dataframe(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_unix(dt: datetime) -> int:
        """Convert a datetime to a Unix timestamp (seconds)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _request_chart(
        self,
        symbol: str,
        interval: str,
        period1: int,
        period2: int,
    ) -> dict[str, Any] | None:
        """Execute one chart API request with rate-limiting and retries.

        Args:
            symbol: Ticker as Yahoo expects it (e.g. ``"AAPL"``, ``"SHOP.TO"``).
            interval: Yahoo interval string (``"1d"`` or ``"1wk"``).
            period1: Start Unix timestamp.
            period2: End Unix timestamp.

        Returns:
            The first element of the ``chart.result`` array, or ``None`` on
            error or empty response.
        """
        self._rate_limit()

        url = _BASE_URL.format(symbol=symbol)
        params: dict[str, Any] = {
            "period1": period1,
            "period2": period2,
            "interval": interval,
            "includePrePost": "false",
            "events": "",
        }
        headers = {"User-Agent": _USER_AGENT}

        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()

        body = resp.json()

        chart = body.get("chart", {})
        error = chart.get("error")
        if error:
            logger.error("Yahoo API error for %s: %s", symbol, error)
            return None

        results = chart.get("result")
        if not results:
            return None

        return results[0]

    @staticmethod
    def _parse_chart_response(data: dict[str, Any]) -> list[list]:
        """Extract OHLCV rows from a Yahoo chart result object.

        Args:
            data: A single element from ``chart.result``.

        Returns:
            List of ``[timestamp, open, high, low, close, volume]`` lists.
        """
        timestamps = data.get("timestamp")
        if not timestamps:
            return []

        indicators = data.get("indicators", {})
        quote_list = indicators.get("quote", [])
        if not quote_list:
            return []

        quote = quote_list[0]
        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        rows: list[list] = []
        for i, ts in enumerate(timestamps):
            o = opens[i] if i < len(opens) else None
            h = highs[i] if i < len(highs) else None
            lo = lows[i] if i < len(lows) else None
            c = closes[i] if i < len(closes) else None
            v = volumes[i] if i < len(volumes) else None

            # Skip candles with missing price data (e.g. holidays)
            if any(val is None for val in (o, h, lo, c)):
                continue

            rows.append(
                [
                    datetime.fromtimestamp(ts, tz=timezone.utc),
                    float(o),
                    float(h),
                    float(lo),
                    float(c),
                    float(v) if v is not None else 0.0,
                ]
            )

        return rows

    def _rate_limit(self) -> None:
        """Block until at least ``_RATE_LIMIT_SECONDS`` since the last request."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _RATE_LIMIT_SECONDS:
            time.sleep(_RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.monotonic()
