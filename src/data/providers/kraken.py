"""Kraken public REST API provider for crypto OHLCV data."""

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

# ---------------------------------------------------------------------------
# Symbol mapping: unified "BASE-USD" -> Kraken pair name
# ---------------------------------------------------------------------------
_SYMBOL_MAP: dict[str, str] = {
    "BTC-USD": "XBTUSD",
    "ETH-USD": "ETHUSD",
    "SOL-USD": "SOLUSD",
    "ADA-USD": "ADAUSD",
    "DOGE-USD": "XDGUSD",
    "XRP-USD": "XXRPZUSD",
    "DOT-USD": "DOTUSD",
    "AVAX-USD": "AVAXUSD",
    "LINK-USD": "LINKUSD",
    "MATIC-USD": "MATICUSD",
    "LTC-USD": "XLTCZUSD",
    "ATOM-USD": "ATOMUSD",
    "UNI-USD": "UNIUSD",
    "AAVE-USD": "AAVEUSD",
    "ALGO-USD": "ALGOUSD",
}

# ---------------------------------------------------------------------------
# Timeframe mapping: human-readable -> Kraken interval (minutes)
# ---------------------------------------------------------------------------
_TIMEFRAME_MAP: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}

_BASE_URL = "https://api.kraken.com/0/public/OHLC"
_MAX_CANDLES_PER_REQUEST = 720
_RATE_LIMIT_SECONDS = 1.0


class KrakenProvider(DataProvider):
    """Fetch crypto OHLCV candles from the Kraken public REST API.

    No authentication is required.  The provider maps unified symbols
    (``BTC-USD``) to Kraken pair names (``XBTUSD``) and handles pagination
    (720 candle limit per request) automatically.
    """

    def __init__(self) -> None:
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    def provider_name(self) -> str:
        return "kraken"

    def supported_timeframes(self) -> list[str]:
        return list(_TIMEFRAME_MAP.keys())

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        self._validate_timeframe(timeframe)

        kraken_pair = self._resolve_symbol(symbol)
        interval = _TIMEFRAME_MAP[timeframe]

        start_ts = int(start.replace(tzinfo=timezone.utc).timestamp()) if start.tzinfo is None else int(start.timestamp())
        end_ts = int(end.replace(tzinfo=timezone.utc).timestamp()) if end.tzinfo is None else int(end.timestamp())

        all_rows: list[list] = []
        since = start_ts

        logger.info(
            "Kraken: fetching %s %s from %s to %s",
            symbol,
            timeframe,
            start.isoformat(),
            end.isoformat(),
        )

        while since < end_ts:
            data = self._request_ohlc(kraken_pair, interval, since)
            if not data:
                logger.debug("Kraken: no more data returned for since=%s", since)
                break

            new_rows = 0
            for candle in data:
                candle_ts = int(candle[0])
                if candle_ts > end_ts:
                    break
                all_rows.append(
                    [
                        datetime.fromtimestamp(candle_ts, tz=timezone.utc),
                        float(candle[1]),  # open
                        float(candle[2]),  # high
                        float(candle[3]),  # low
                        float(candle[4]),  # close
                        float(candle[6]),  # volume (index 6 in Kraken response)
                    ]
                )
                new_rows += 1

            if new_rows == 0:
                break

            # Move 'since' forward to the last timestamp we received
            last_ts = int(data[-1][0])
            if last_ts <= since:
                # Safety: avoid infinite loop if API keeps returning same data
                break
            since = last_ts

            logger.debug(
                "Kraken: fetched %d candles, total so far %d, since=%s",
                new_rows,
                len(all_rows),
                since,
            )

        logger.info("Kraken: total %d candles for %s %s", len(all_rows), symbol, timeframe)
        return self._make_dataframe(all_rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_symbol(self, symbol: str) -> str:
        """Map a unified symbol to Kraken's pair name."""
        pair = _SYMBOL_MAP.get(symbol)
        if pair is None:
            # Fall back: strip the dash and hope Kraken recognises it
            pair = symbol.replace("-", "")
            logger.warning(
                "Kraken: no explicit mapping for '%s', trying '%s'",
                symbol,
                pair,
            )
        return pair

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _request_ohlc(
        self,
        pair: str,
        interval: int,
        since: int,
    ) -> list[list[Any]]:
        """Execute a single OHLC request with rate limiting and retries.

        Args:
            pair: Kraken pair name (e.g. ``"XBTUSD"``).
            interval: Candle interval in minutes.
            since: Unix timestamp to fetch candles after.

        Returns:
            List of candle arrays from the Kraken response, or an empty list
            on error.
        """
        self._rate_limit()

        params: dict[str, Any] = {
            "pair": pair,
            "interval": interval,
            "since": since,
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.get(_BASE_URL, params=params)
            resp.raise_for_status()

        body = resp.json()

        errors = body.get("error", [])
        if errors:
            logger.error("Kraken API error: %s", errors)
            return []

        result = body.get("result", {})
        # The result dict has the pair key and a "last" key; we want the pair key.
        for key, value in result.items():
            if key == "last":
                continue
            if isinstance(value, list):
                return value

        return []

    def _rate_limit(self) -> None:
        """Block until at least ``_RATE_LIMIT_SECONDS`` since the last request."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _RATE_LIMIT_SECONDS:
            time.sleep(_RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.monotonic()
