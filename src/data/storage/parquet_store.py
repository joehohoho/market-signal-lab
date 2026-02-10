"""Parquet-backed OHLCV candle storage using PyArrow."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_DEFAULT_PARQUET_DIR = "data/candles"

# Canonical column order and dtypes for every candle DataFrame.
CANDLE_COLUMNS: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]
CANDLE_DTYPES: dict[str, str] = {
    "timestamp": "datetime64[ns]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

# PyArrow schema kept in sync with the pandas dtypes above.
_PA_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ns")),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
    ]
)


def _safe_filename(symbol: str, timeframe: str) -> str:
    """Build a parquet filename, replacing ``/`` with ``_`` in the symbol."""
    safe_symbol = symbol.replace("/", "_")
    return f"{safe_symbol}_{timeframe}.parquet"


class ParquetStore:
    """Store and load OHLCV candle data as Parquet files.

    One file per (symbol, timeframe) pair, located under *parquet_dir*.
    """

    def __init__(self, parquet_dir: str = _DEFAULT_PARQUET_DIR) -> None:
        self._parquet_dir = Path(parquet_dir)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Append *df* to the existing file, deduplicating by timestamp.

        Parameters
        ----------
        symbol : str
            Asset symbol (e.g. ``"BTC-USD"``).
        timeframe : str
            Candle interval (e.g. ``"1d"``, ``"15m"``).
        df : pd.DataFrame
            Must contain at least the columns in :data:`CANDLE_COLUMNS`.

        Returns
        -------
        int
            Total row count after save.
        """
        df = self._normalise(df)
        path = self._path_for(symbol, timeframe)

        if path.exists():
            existing = self._read_raw(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.drop_duplicates(subset="timestamp", keep="last", inplace=True)
            combined.sort_values("timestamp", inplace=True)
            combined.reset_index(drop=True, inplace=True)
        else:
            combined = df.sort_values("timestamp").reset_index(drop=True)

        table = pa.Table.from_pandas(combined, schema=_PA_SCHEMA, preserve_index=False)
        pq.write_table(table, path)
        logger.debug(
            "Saved %d rows for %s/%s -> %s", len(combined), symbol, timeframe, path
        )
        return len(combined)

    def load(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load candle data, optionally filtered to [*start*, *end*].

        Parameters
        ----------
        symbol : str
        timeframe : str
        start : datetime, optional
            Inclusive lower bound on ``timestamp``.
        end : datetime, optional
            Inclusive upper bound on ``timestamp``.

        Returns
        -------
        pd.DataFrame
            Empty DataFrame with the correct schema if no data exists.
        """
        path = self._path_for(symbol, timeframe)
        if not path.exists():
            logger.debug("No parquet file for %s/%s at %s", symbol, timeframe, path)
            return self._empty_frame()

        df = self._read_raw(path)

        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]

        return df.reset_index(drop=True)

    def delete(self, symbol: str, timeframe: str) -> bool:
        """Remove the parquet file for a (symbol, timeframe) pair.

        Returns ``True`` if the file existed and was deleted.
        """
        path = self._path_for(symbol, timeframe)
        if path.exists():
            path.unlink()
            logger.debug("Deleted parquet file %s", path)
            return True
        return False

    def exists(self, symbol: str, timeframe: str) -> bool:
        """Check whether a parquet file exists for this pair."""
        return self._path_for(symbol, timeframe).exists()

    def row_count(self, symbol: str, timeframe: str) -> int:
        """Return the number of rows stored, or 0 if the file is absent."""
        path = self._path_for(symbol, timeframe)
        if not path.exists():
            return 0
        meta = pq.read_metadata(path)
        return meta.num_rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_for(self, symbol: str, timeframe: str) -> Path:
        return self._parquet_dir / _safe_filename(symbol, timeframe)

    @staticmethod
    def _read_raw(path: Path) -> pd.DataFrame:
        """Read a parquet file into a DataFrame with canonical dtypes."""
        table = pq.read_table(path)
        df = table.to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        return df

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and cast *df* to the canonical schema."""
        missing = set(CANDLE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        df = df[CANDLE_COLUMNS].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype("float64")
        return df

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        """Return an empty DataFrame with the canonical schema."""
        return pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in CANDLE_DTYPES.items()}
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    @property
    def parquet_dir(self) -> Path:
        """Return the resolved parquet directory path."""
        return self._parquet_dir

    def __repr__(self) -> str:
        return f"ParquetStore(parquet_dir={str(self._parquet_dir)!r})"
