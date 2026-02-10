"""DuckDB-based metadata store for tracking ingested market data."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "data/msl.duckdb"


class DuckDBStore:
    """Manages asset and candle metadata in a local DuckDB database.

    Tables
    ------
    assets
        symbol TEXT PRIMARY KEY, asset_class TEXT, exchange TEXT
    candle_metadata
        (symbol, timeframe) composite PK, first_ts TIMESTAMP, last_ts TIMESTAMP, row_count BIGINT
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con: duckdb.DuckDBPyConnection = duckdb.connect(str(self._db_path))
        self._init_tables()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        """Create tables if they do not already exist."""
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                symbol      TEXT PRIMARY KEY,
                asset_class TEXT NOT NULL,
                exchange    TEXT
            );
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS candle_metadata (
                symbol    TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                first_ts  TIMESTAMP NOT NULL,
                last_ts   TIMESTAMP NOT NULL,
                row_count BIGINT NOT NULL DEFAULT 0,
                PRIMARY KEY (symbol, timeframe)
            );
        """)
        logger.debug("DuckDB tables initialised at %s", self._db_path)

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._con.close()

    # ------------------------------------------------------------------
    # Assets CRUD
    # ------------------------------------------------------------------

    def upsert_asset(
        self,
        symbol: str,
        asset_class: str,
        exchange: Optional[str] = None,
    ) -> None:
        """Insert or update an asset record."""
        self._con.execute(
            """
            INSERT INTO assets (symbol, asset_class, exchange)
            VALUES (?, ?, ?)
            ON CONFLICT (symbol) DO UPDATE
                SET asset_class = EXCLUDED.asset_class,
                    exchange    = EXCLUDED.exchange;
            """,
            [symbol, asset_class, exchange],
        )
        logger.debug("Upserted asset %s (%s)", symbol, asset_class)

    def get_asset(self, symbol: str) -> Optional[dict]:
        """Return a single asset dict or ``None``."""
        result = self._con.execute(
            "SELECT symbol, asset_class, exchange FROM assets WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if result is None:
            return None
        return {"symbol": result[0], "asset_class": result[1], "exchange": result[2]}

    def list_assets(self, asset_class: Optional[str] = None) -> list[dict]:
        """Return all assets, optionally filtered by *asset_class*."""
        if asset_class is not None:
            rows = self._con.execute(
                "SELECT symbol, asset_class, exchange FROM assets WHERE asset_class = ? ORDER BY symbol",
                [asset_class],
            ).fetchall()
        else:
            rows = self._con.execute(
                "SELECT symbol, asset_class, exchange FROM assets ORDER BY symbol"
            ).fetchall()
        return [
            {"symbol": r[0], "asset_class": r[1], "exchange": r[2]} for r in rows
        ]

    def delete_asset(self, symbol: str) -> bool:
        """Delete an asset and its associated candle metadata.

        Returns ``True`` if the asset existed.
        """
        existing = self.get_asset(symbol)
        if existing is None:
            return False
        self._con.execute(
            "DELETE FROM candle_metadata WHERE symbol = ?", [symbol]
        )
        self._con.execute("DELETE FROM assets WHERE symbol = ?", [symbol])
        logger.debug("Deleted asset %s and its candle metadata", symbol)
        return True

    # ------------------------------------------------------------------
    # Candle metadata CRUD
    # ------------------------------------------------------------------

    def upsert_candle_metadata(
        self,
        symbol: str,
        timeframe: str,
        first_ts: datetime,
        last_ts: datetime,
        row_count: int,
    ) -> None:
        """Insert or update the candle metadata for a (symbol, timeframe) pair."""
        self._con.execute(
            """
            INSERT INTO candle_metadata (symbol, timeframe, first_ts, last_ts, row_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (symbol, timeframe) DO UPDATE
                SET first_ts  = EXCLUDED.first_ts,
                    last_ts   = EXCLUDED.last_ts,
                    row_count = EXCLUDED.row_count;
            """,
            [symbol, timeframe, first_ts, last_ts, row_count],
        )
        logger.debug(
            "Upserted candle_metadata %s/%s  [%s -> %s, %d rows]",
            symbol,
            timeframe,
            first_ts,
            last_ts,
            row_count,
        )

    def get_candle_metadata(
        self, symbol: str, timeframe: str
    ) -> Optional[dict]:
        """Return metadata dict for one (symbol, timeframe) or ``None``."""
        result = self._con.execute(
            "SELECT symbol, timeframe, first_ts, last_ts, row_count "
            "FROM candle_metadata WHERE symbol = ? AND timeframe = ?",
            [symbol, timeframe],
        ).fetchone()
        if result is None:
            return None
        return {
            "symbol": result[0],
            "timeframe": result[1],
            "first_ts": result[2],
            "last_ts": result[3],
            "row_count": result[4],
        }

    def list_candle_metadata(
        self, symbol: Optional[str] = None
    ) -> list[dict]:
        """Return candle metadata rows, optionally filtered by *symbol*."""
        if symbol is not None:
            rows = self._con.execute(
                "SELECT symbol, timeframe, first_ts, last_ts, row_count "
                "FROM candle_metadata WHERE symbol = ? ORDER BY symbol, timeframe",
                [symbol],
            ).fetchall()
        else:
            rows = self._con.execute(
                "SELECT symbol, timeframe, first_ts, last_ts, row_count "
                "FROM candle_metadata ORDER BY symbol, timeframe"
            ).fetchall()
        return [
            {
                "symbol": r[0],
                "timeframe": r[1],
                "first_ts": r[2],
                "last_ts": r[3],
                "row_count": r[4],
            }
            for r in rows
        ]

    def delete_candle_metadata(self, symbol: str, timeframe: str) -> bool:
        """Delete a single candle metadata row.

        Returns ``True`` if the row existed.
        """
        existing = self.get_candle_metadata(symbol, timeframe)
        if existing is None:
            return False
        self._con.execute(
            "DELETE FROM candle_metadata WHERE symbol = ? AND timeframe = ?",
            [symbol, timeframe],
        )
        logger.debug("Deleted candle_metadata %s/%s", symbol, timeframe)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> Path:
        """Return the resolved database file path."""
        return self._db_path

    def __repr__(self) -> str:
        return f"DuckDBStore(db_path={str(self._db_path)!r})"
