"""Abstract base class for OHLCV data providers."""

from __future__ import annotations

import abc
from datetime import datetime

import pandas as pd


class DataProvider(abc.ABC):
    """Base interface every data provider must implement.

    Concrete providers fetch OHLCV candle data from a specific source and
    return it as a :class:`pandas.DataFrame` with the following columns:

    * ``timestamp`` -- timezone-aware UTC :class:`datetime64[ns, UTC]`
    * ``open``      -- float64
    * ``high``      -- float64
    * ``low``       -- float64
    * ``close``     -- float64
    * ``volume``    -- float64
    """

    # Canonical column order shared by all providers.
    COLUMNS: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data for *symbol* over [*start*, *end*].

        Args:
            symbol: Unified symbol name (e.g. ``"BTC-USD"``, ``"AAPL"``).
            timeframe: Candle interval such as ``"1d"``, ``"1h"``, ``"15m"``.
            start: Inclusive start of the requested range (UTC).
            end: Inclusive end of the requested range (UTC).

        Returns:
            A :class:`~pandas.DataFrame` with columns ``timestamp``, ``open``,
            ``high``, ``low``, ``close``, ``volume`` sorted by ``timestamp``
            ascending.  The ``timestamp`` column must be timezone-aware UTC.
        """
        ...

    @abc.abstractmethod
    def supported_timeframes(self) -> list[str]:
        """Return the list of timeframe strings this provider supports."""
        ...

    @abc.abstractmethod
    def provider_name(self) -> str:
        """Return a short, unique identifier for this provider (e.g. ``"kraken"``)."""
        ...

    # ------------------------------------------------------------------
    # Helpers available to all providers
    # ------------------------------------------------------------------

    def _validate_timeframe(self, timeframe: str) -> None:
        """Raise :class:`ValueError` if *timeframe* is not supported."""
        supported = self.supported_timeframes()
        if timeframe not in supported:
            raise ValueError(
                f"Timeframe '{timeframe}' is not supported by {self.provider_name()}. "
                f"Supported: {supported}"
            )

    @staticmethod
    def _make_dataframe(rows: list[list], tz_aware: bool = True) -> pd.DataFrame:
        """Build a standardised DataFrame from raw row data.

        Args:
            rows: List of ``[timestamp, open, high, low, close, volume]`` lists.
                  ``timestamp`` may be a :class:`datetime`, Unix-seconds int/float,
                  or any value convertible by :func:`pd.to_datetime`.
            tz_aware: If *True* (default), localise the timestamp column to UTC.

        Returns:
            A sorted DataFrame with correct dtypes.
        """
        if not rows:
            return pd.DataFrame(columns=DataProvider.COLUMNS)

        df = pd.DataFrame(rows, columns=DataProvider.COLUMNS)

        # Coerce timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=tz_aware)

        # Numeric columns
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
