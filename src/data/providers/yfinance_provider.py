"""Yahoo Finance provider using the yfinance library.

Provides full historical OHLCV data for equities, ETFs, crypto, and
other Yahoo-supported tickers.  Unlike the raw chart API, the yfinance
library handles session cookies and rate limiting automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from app.logging import get_logger
from data.providers.base import DataProvider

logger = get_logger(__name__)


class YFinanceProvider(DataProvider):
    """Fetch OHLCV data via the ``yfinance`` library.

    Supports intraday (15m, 1h, 4h) and daily/weekly timeframes.

    Note: yfinance intraday history is limited — roughly 60 days for
    15m and 730 days for 1h.  The 4h timeframe is built by fetching 1h
    data and resampling.
    """

    # Map our canonical timeframes to yfinance interval strings.
    _TF_MAP: dict[str, str] = {
        "15m": "15m",
        "1h": "60m",
        "4h": "60m",   # fetch 1h, resample to 4h
        "1d": "1d",
        "1wk": "1wk",
    }

    def provider_name(self) -> str:
        return "yfinance"

    def supported_timeframes(self) -> list[str]:
        return list(self._TF_MAP.keys())

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        self._validate_timeframe(timeframe)

        interval = self._TF_MAP[timeframe]
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        logger.info(
            "yfinance: fetching %s %s from %s to %s",
            symbol, timeframe, start_str, end_str,
        )

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_str, end=end_str, interval=interval, auto_adjust=True)

        if df.empty:
            logger.warning("yfinance: no data returned for %s", symbol)
            return self._make_dataframe([])

        # Resample to 4h if needed
        if timeframe == "4h":
            df = self._resample_to_4h(df)
            if df.empty:
                return self._make_dataframe([])

        rows = []
        for ts, row in df.iterrows():
            rows.append([
                ts.to_pydatetime().replace(tzinfo=timezone.utc),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                float(row["Volume"]),
            ])

        logger.info("yfinance: %d candles for %s %s", len(rows), symbol, timeframe)
        return self._make_dataframe(rows)

    @staticmethod
    def _resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1-hour OHLCV data to 4-hour bars."""
        return df.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna(subset=["Open"])
