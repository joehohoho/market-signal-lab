"""Data ingestion orchestrator.

Fetches OHLCV data from a provider, stores it to Parquet via
:class:`~data.storage.ParquetStore`, and registers metadata in DuckDB via
:class:`~data.storage.DuckDBStore`.  Includes gap detection so that
incremental ingestion fills only the missing portions of a time range.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from app.config import get_config
from app.logging import get_logger
from data.providers import DataProvider, get_provider
from data.storage import DuckDBStore, ParquetStore

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Asset-class detection
# ---------------------------------------------------------------------------

_CRYPTO_SUFFIXES = ("-USD", "-USDT", "-USDC", "-EUR", "-GBP", "-CAD", "-BTC", "-ETH")


def _detect_asset_class(symbol: str) -> str:
    """Heuristically determine whether *symbol* is crypto or equity.

    Returns ``"crypto"`` or ``"equity"``.
    """
    upper = symbol.upper()
    for suffix in _CRYPTO_SUFFIXES:
        if upper.endswith(suffix):
            return "crypto"
    return "equity"


def _default_provider_for(asset_class: str) -> str:
    """Look up the default provider name from config for *asset_class*."""
    try:
        providers_cfg: dict[str, Any] = get_config("providers")
    except KeyError:
        # Fallback when config section is absent.
        return "kraken" if asset_class == "crypto" else "yahoo_daily"

    section = providers_cfg.get(asset_class, {})
    if isinstance(section, dict):
        return section.get("default", "kraken" if asset_class == "crypto" else "yahoo_daily")
    return str(section)


# ---------------------------------------------------------------------------
# Store singletons (lazy-initialised from config)
# ---------------------------------------------------------------------------

_parquet_store: ParquetStore | None = None
_duckdb_store: DuckDBStore | None = None


def _get_parquet_store() -> ParquetStore:
    """Return the singleton :class:`ParquetStore`, creating it on first use."""
    global _parquet_store
    if _parquet_store is None:
        try:
            storage_cfg = get_config("storage")
            parquet_dir = storage_cfg.get("parquet_dir", "data/candles")
        except KeyError:
            parquet_dir = "data/candles"
        _parquet_store = ParquetStore(parquet_dir=parquet_dir)
    return _parquet_store


def _get_duckdb_store() -> DuckDBStore:
    """Return the singleton :class:`DuckDBStore`, creating it on first use."""
    global _duckdb_store
    if _duckdb_store is None:
        try:
            storage_cfg = get_config("storage")
            db_path = storage_cfg.get("db_path", "data/msl.duckdb")
        except KeyError:
            db_path = "data/msl.duckdb"
        _duckdb_store = DuckDBStore(db_path=db_path)
    return _duckdb_store


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def _detect_gaps(
    existing: pd.DataFrame | None,
    start: datetime,
    end: datetime,
    timeframe: str,
) -> list[tuple[datetime, datetime]]:
    """Find time gaps in *existing* data within [*start*, *end*].

    Returns a list of ``(gap_start, gap_end)`` tuples that need to be
    fetched.  If there is no existing data the entire range is returned.

    The gap detection uses a simple heuristic: any break between consecutive
    timestamps that exceeds twice the expected candle interval is treated as
    a gap.
    """
    if existing is None or existing.empty:
        return [(start, end)]

    # Estimate expected interval from timeframe string
    interval_seconds = _timeframe_to_seconds(timeframe)
    threshold = timedelta(seconds=interval_seconds * 2.5)

    ts = existing["timestamp"].sort_values()
    start_utc = pd.Timestamp(start, tz="UTC")
    end_utc = pd.Timestamp(end, tz="UTC")

    gaps: list[tuple[datetime, datetime]] = []

    # Gap before existing data?
    if ts.iloc[0] > start_utc + threshold:
        gaps.append((start, ts.iloc[0].to_pydatetime()))

    # Gaps in the middle
    diffs = ts.diff()
    for i in range(1, len(diffs)):
        if diffs.iloc[i] > threshold:
            gap_start = ts.iloc[i - 1].to_pydatetime()
            gap_end = ts.iloc[i].to_pydatetime()
            gaps.append((gap_start, gap_end))

    # Gap after existing data?
    if ts.iloc[-1] < end_utc - threshold:
        gaps.append((ts.iloc[-1].to_pydatetime(), end))

    return gaps


def _timeframe_to_seconds(timeframe: str) -> int:
    """Convert a timeframe string to an approximate number of seconds."""
    multipliers = {
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
        "wk": 604800,
    }
    # e.g. "15m" -> number=15, unit="m"
    for unit, factor in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if timeframe.endswith(unit):
            try:
                number = int(timeframe[: -len(unit)])
                return number * factor
            except ValueError:
                break
    # Fallback: 1 day
    return 86400


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_asset(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime | None = None,
    provider_name: str | None = None,
) -> pd.DataFrame:
    """Ingest OHLCV data for a single asset, filling gaps in existing data.

    Workflow:
        1. Detect asset class (crypto vs equity) from the symbol.
        2. Select the appropriate provider from config (or use *provider_name*).
        3. Load existing data from :class:`~data.storage.ParquetStore`.
        4. Detect gaps between *start*/*end* and existing timestamps.
        5. Fetch missing data from the provider for each gap.
        6. Merge, deduplicate, and save back to Parquet.
        7. Update candle metadata in :class:`~data.storage.DuckDBStore`.

    Args:
        symbol: Unified symbol (e.g. ``"BTC-USD"``, ``"AAPL"``, ``"SHOP.TO"``).
        timeframe: Candle interval (e.g. ``"1d"``, ``"15m"``).
        start: Start of the desired date range (UTC).
        end: End of the desired date range.  Defaults to *now* (UTC).
        provider_name: Explicit provider name.  When ``None`` the provider is
            chosen automatically based on asset class and config.

    Returns:
        The complete :class:`~pandas.DataFrame` of OHLCV data for the
        requested range (union of existing + newly fetched data).
    """
    if end is None:
        end = datetime.now(timezone.utc)

    # Ensure tz-aware
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    asset_class = _detect_asset_class(symbol)

    if provider_name is None:
        provider_name = _default_provider_for(asset_class)

    logger.info(
        "Ingesting %s [%s] from %s to %s via %s",
        symbol,
        timeframe,
        start.date(),
        end.date(),
        provider_name,
    )

    provider: DataProvider = get_provider(provider_name)
    pq_store = _get_parquet_store()

    # Load existing data and detect gaps
    existing = pq_store.load(symbol, timeframe)
    existing_or_none = existing if not existing.empty else None
    gaps = _detect_gaps(existing_or_none, start, end, timeframe)

    if not gaps:
        logger.info("No gaps detected for %s %s -- data is up to date", symbol, timeframe)
        return existing

    logger.info("Detected %d gap(s) to fill for %s %s", len(gaps), symbol, timeframe)

    # Fetch data for each gap
    fetched_frames: list[pd.DataFrame] = []
    for gap_start, gap_end in gaps:
        logger.info("  Fetching gap: %s -> %s", gap_start, gap_end)
        try:
            df = provider.fetch_ohlcv(symbol, timeframe, gap_start, gap_end)
            if not df.empty:
                fetched_frames.append(df)
        except Exception:
            logger.exception(
                "Failed to fetch %s %s for gap %s-%s",
                symbol,
                timeframe,
                gap_start,
                gap_end,
            )

    if not fetched_frames:
        if existing_or_none is not None:
            logger.info("No new data fetched for %s %s, returning existing", symbol, timeframe)
            return existing
        logger.warning("No data available for %s %s", symbol, timeframe)
        return pd.DataFrame(columns=DataProvider.COLUMNS)

    # Concatenate all newly fetched frames into one DataFrame
    new_data = pd.concat(fetched_frames, ignore_index=True)

    # Save through ParquetStore (it handles merge + dedup internally)
    total_rows = pq_store.save(symbol, timeframe, new_data)

    # Update DuckDB metadata
    try:
        db_store = _get_duckdb_store()
        db_store.upsert_asset(symbol, asset_class)
        # Re-load the final merged data to get accurate first/last timestamps
        final = pq_store.load(symbol, timeframe)
        if not final.empty:
            db_store.upsert_candle_metadata(
                symbol=symbol,
                timeframe=timeframe,
                first_ts=final["timestamp"].min().to_pydatetime(),
                last_ts=final["timestamp"].max().to_pydatetime(),
                row_count=total_rows,
            )
    except Exception as exc:
        logger.warning("Could not update DuckDB metadata: %s", exc)

    logger.info(
        "Ingestion complete for %s %s: %d total rows",
        symbol,
        timeframe,
        total_rows,
    )

    return pq_store.load(symbol, timeframe)
