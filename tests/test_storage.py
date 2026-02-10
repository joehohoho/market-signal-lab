"""Tests for storage modules: ParquetStore and DuckDBStore.

All tests use pytest's tmp_path fixture for full isolation -- no
production data directories are touched.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from data.storage.parquet_store import CANDLE_COLUMNS, ParquetStore
from data.storage.duckdb_store import DuckDBStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_candles(n: int, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a small OHLCV DataFrame with *n* rows."""
    import numpy as np

    rng = np.random.default_rng(0)
    ts = pd.date_range(start, periods=n, freq="D")
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.integers(100_000, 500_000, size=n).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# ParquetStore
# ---------------------------------------------------------------------------

class TestParquetStoreSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_storage_dir) -> None:
        """Save a DataFrame, load it back, and verify equality."""
        store = ParquetStore(parquet_dir=str(tmp_storage_dir))
        original = _sample_candles(20)

        row_count = store.save("BTC-USD", "1d", original)
        assert row_count == 20

        loaded = store.load("BTC-USD", "1d")
        assert len(loaded) == 20
        assert list(loaded.columns) == CANDLE_COLUMNS
        pd.testing.assert_frame_equal(
            loaded[["open", "high", "low", "close", "volume"]].reset_index(drop=True),
            original[["open", "high", "low", "close", "volume"]].reset_index(drop=True),
            check_dtype=True,
        )

    def test_exists_and_row_count(self, tmp_storage_dir) -> None:
        """exists() and row_count() should reflect stored data."""
        store = ParquetStore(parquet_dir=str(tmp_storage_dir))
        assert not store.exists("ETH-USD", "1d")
        assert store.row_count("ETH-USD", "1d") == 0

        store.save("ETH-USD", "1d", _sample_candles(15))
        assert store.exists("ETH-USD", "1d")
        assert store.row_count("ETH-USD", "1d") == 15


class TestParquetStoreAppend:
    def test_append_deduplicates(self, tmp_storage_dir) -> None:
        """Saving overlapping data should deduplicate by timestamp."""
        store = ParquetStore(parquet_dir=str(tmp_storage_dir))

        batch1 = _sample_candles(10, start="2024-01-01")
        batch2 = _sample_candles(10, start="2024-01-08")  # overlaps 3 days

        store.save("AAPL", "1d", batch1)
        total = store.save("AAPL", "1d", batch2)

        loaded = store.load("AAPL", "1d")
        # 10 + 10 - 3 overlap = 17 unique timestamps
        assert total == 17
        assert len(loaded) == 17

    def test_append_preserves_order(self, tmp_storage_dir) -> None:
        """Data should be sorted by timestamp after append."""
        store = ParquetStore(parquet_dir=str(tmp_storage_dir))

        # Save later dates first, then earlier dates.
        later = _sample_candles(5, start="2024-02-01")
        earlier = _sample_candles(5, start="2024-01-01")

        store.save("TEST", "1d", later)
        store.save("TEST", "1d", earlier)

        loaded = store.load("TEST", "1d")
        timestamps = loaded["timestamp"].tolist()
        assert timestamps == sorted(timestamps)


class TestParquetStoreDateFilter:
    def test_load_with_start_date(self, tmp_storage_dir) -> None:
        """Loading with a start date should exclude earlier rows."""
        store = ParquetStore(parquet_dir=str(tmp_storage_dir))
        store.save("MSFT", "1d", _sample_candles(30, start="2024-01-01"))

        loaded = store.load("MSFT", "1d", start=datetime(2024, 1, 15))
        assert len(loaded) > 0
        assert loaded["timestamp"].min() >= pd.Timestamp("2024-01-15")

    def test_load_with_end_date(self, tmp_storage_dir) -> None:
        """Loading with an end date should exclude later rows."""
        store = ParquetStore(parquet_dir=str(tmp_storage_dir))
        store.save("MSFT", "1d", _sample_candles(30, start="2024-01-01"))

        loaded = store.load("MSFT", "1d", end=datetime(2024, 1, 15))
        assert len(loaded) > 0
        assert loaded["timestamp"].max() <= pd.Timestamp("2024-01-15")

    def test_load_with_date_range(self, tmp_storage_dir) -> None:
        """Loading with both start and end should return only the window."""
        store = ParquetStore(parquet_dir=str(tmp_storage_dir))
        store.save("MSFT", "1d", _sample_candles(60, start="2024-01-01"))

        loaded = store.load(
            "MSFT", "1d",
            start=datetime(2024, 1, 15),
            end=datetime(2024, 2, 15),
        )
        assert len(loaded) > 0
        assert loaded["timestamp"].min() >= pd.Timestamp("2024-01-15")
        assert loaded["timestamp"].max() <= pd.Timestamp("2024-02-15")

    def test_load_nonexistent_returns_empty(self, tmp_storage_dir) -> None:
        """Loading a symbol that was never saved should return an empty DataFrame."""
        store = ParquetStore(parquet_dir=str(tmp_storage_dir))
        loaded = store.load("NOPE", "1d")
        assert loaded.empty
        assert list(loaded.columns) == CANDLE_COLUMNS


class TestParquetStoreDelete:
    def test_delete_existing(self, tmp_storage_dir) -> None:
        """delete() should remove the file and return True."""
        store = ParquetStore(parquet_dir=str(tmp_storage_dir))
        store.save("DEL", "1d", _sample_candles(5))
        assert store.delete("DEL", "1d") is True
        assert not store.exists("DEL", "1d")

    def test_delete_nonexistent(self, tmp_storage_dir) -> None:
        """delete() on a missing file should return False."""
        store = ParquetStore(parquet_dir=str(tmp_storage_dir))
        assert store.delete("NOPE", "1d") is False


# ---------------------------------------------------------------------------
# DuckDBStore
# ---------------------------------------------------------------------------

class TestDuckDBStoreMetadata:
    def test_upsert_and_get_asset(self, tmp_storage_dir) -> None:
        """Upserting an asset and retrieving it should round-trip correctly."""
        db_path = str(tmp_storage_dir / "test.duckdb")
        store = DuckDBStore(db_path=db_path)
        try:
            store.upsert_asset("BTC-USD", "crypto", exchange="kraken")
            asset = store.get_asset("BTC-USD")

            assert asset is not None
            assert asset["symbol"] == "BTC-USD"
            assert asset["asset_class"] == "crypto"
            assert asset["exchange"] == "kraken"
        finally:
            store.close()

    def test_list_assets(self, tmp_storage_dir) -> None:
        """list_assets() should return all inserted assets."""
        db_path = str(tmp_storage_dir / "test.duckdb")
        store = DuckDBStore(db_path=db_path)
        try:
            store.upsert_asset("AAPL", "equity")
            store.upsert_asset("BTC-USD", "crypto")
            store.upsert_asset("MSFT", "equity")

            all_assets = store.list_assets()
            assert len(all_assets) == 3

            equities = store.list_assets(asset_class="equity")
            assert len(equities) == 2
            assert all(a["asset_class"] == "equity" for a in equities)
        finally:
            store.close()

    def test_delete_asset(self, tmp_storage_dir) -> None:
        """Deleting an asset should remove it and return True."""
        db_path = str(tmp_storage_dir / "test.duckdb")
        store = DuckDBStore(db_path=db_path)
        try:
            store.upsert_asset("DEL-ME", "crypto")
            assert store.delete_asset("DEL-ME") is True
            assert store.get_asset("DEL-ME") is None
            assert store.delete_asset("DEL-ME") is False
        finally:
            store.close()

    def test_candle_metadata_crud(self, tmp_storage_dir) -> None:
        """Candle metadata upsert, get, list, and delete should all work."""
        db_path = str(tmp_storage_dir / "test.duckdb")
        store = DuckDBStore(db_path=db_path)
        try:
            store.upsert_candle_metadata(
                symbol="BTC-USD",
                timeframe="1d",
                first_ts=datetime(2024, 1, 1),
                last_ts=datetime(2024, 6, 30),
                row_count=181,
            )

            meta = store.get_candle_metadata("BTC-USD", "1d")
            assert meta is not None
            assert meta["symbol"] == "BTC-USD"
            assert meta["timeframe"] == "1d"
            assert meta["row_count"] == 181

            # List should include the entry.
            all_meta = store.list_candle_metadata()
            assert len(all_meta) == 1

            # Delete should succeed.
            assert store.delete_candle_metadata("BTC-USD", "1d") is True
            assert store.get_candle_metadata("BTC-USD", "1d") is None
        finally:
            store.close()

    def test_upsert_overwrites(self, tmp_storage_dir) -> None:
        """Upserting the same (symbol, timeframe) should update, not duplicate."""
        db_path = str(tmp_storage_dir / "test.duckdb")
        store = DuckDBStore(db_path=db_path)
        try:
            store.upsert_candle_metadata("X", "1d", datetime(2024, 1, 1), datetime(2024, 3, 1), 60)
            store.upsert_candle_metadata("X", "1d", datetime(2024, 1, 1), datetime(2024, 6, 1), 150)

            meta = store.get_candle_metadata("X", "1d")
            assert meta is not None
            assert meta["row_count"] == 150

            all_meta = store.list_candle_metadata(symbol="X")
            assert len(all_meta) == 1
        finally:
            store.close()

    def test_get_nonexistent_returns_none(self, tmp_storage_dir) -> None:
        """Querying a missing asset/metadata should return None."""
        db_path = str(tmp_storage_dir / "test.duckdb")
        store = DuckDBStore(db_path=db_path)
        try:
            assert store.get_asset("NOPE") is None
            assert store.get_candle_metadata("NOPE", "1d") is None
        finally:
            store.close()
