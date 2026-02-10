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

    Supports daily and weekly timeframes with full history.
    """

    def provider_name(self) -> str:
        return "yfinance"

    def supported_timeframes(self) -> list[str]:
        return ["1d", "1wk"]

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        self._validate_timeframe(timeframe)

        interval = "1d" if timeframe == "1d" else "1wk"
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
