"""Bulk import of local CSV OHLCV files into ParquetStore + DuckDBStore.

Designed for Kraken-exported CSV data where files follow the naming
convention ``{PAIR}_{MINUTES}.csv`` (e.g. ``XBTUSD_60.csv``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from app.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Timeframe mapping: CSV minutes -> app timeframe string
# ---------------------------------------------------------------------------

TIMEFRAME_MAP: dict[int, str] = {
    1: "1m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    240: "4h",
    720: "12h",
    1440: "1d",
}

_VALID_MINUTES: set[int] = set(TIMEFRAME_MAP.keys())

# ---------------------------------------------------------------------------
# Symbol parsing
# ---------------------------------------------------------------------------

# Known quote currencies, ordered longest-first for greedy suffix matching.
KNOWN_QUOTES: tuple[str, ...] = (
    "USDT", "USDC", "PYUSD",
    "USD", "EUR", "GBP", "AUD", "CAD", "JPY", "CHF", "AED", "DAI",
    "XBT", "ETH", "BTC",
)

# Normalise Kraken-specific base names to standard tickers.
BASE_NORMALIZE: dict[str, str] = {"XBT": "BTC"}

# Default base currencies to import.
DEFAULT_BASE_FILTER: set[str] = {"BTC", "XBT", "ETH"}

# Filename regex: captures PAIR and MINUTES from "{PAIR}_{MINUTES}.csv"
_FILENAME_RE = re.compile(r"^([A-Za-z0-9.]+)_(\d+)\.csv$")

# CSV column names (Kraken export format).
_CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "trades"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedFile:
    """A parsed CSV filename with resolved app-level identifiers."""

    path: Path
    raw_pair: str       # e.g. "XBTUSD"
    minutes: int        # e.g. 60
    base: str           # e.g. "XBT" (raw)
    quote: str          # e.g. "USD"
    app_symbol: str     # e.g. "BTC-USD" (normalised)
    app_timeframe: str  # e.g. "1h"


@dataclass
class ImportResult:
    """Summary of a single file import."""

    app_symbol: str
    app_timeframe: str
    source_file: str
    rows_imported: int
    total_rows: int
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _split_pair(pair: str) -> tuple[Optional[str], Optional[str]]:
    """Split a concatenated pair like ``XBTUSD`` into ``("XBT", "USD")``.

    Tries known quote currencies as suffixes, longest first.
    Returns ``(None, None)`` if no known quote matches.
    """
    for quote in KNOWN_QUOTES:
        if pair.endswith(quote) and len(pair) > len(quote):
            base = pair[: -len(quote)]
            return base, quote
    return None, None


def parse_csv_filename(filename: str) -> Optional[ParsedFile]:
    """Parse a CSV filename into a :class:`ParsedFile`.

    Returns ``None`` if the filename does not match the expected pattern,
    the timeframe is unrecognised, or the pair cannot be split.
    """
    match = _FILENAME_RE.match(filename)
    if not match:
        return None

    raw_pair = match.group(1).upper()
    try:
        minutes = int(match.group(2))
    except ValueError:
        return None

    if minutes not in _VALID_MINUTES:
        return None

    base, quote = _split_pair(raw_pair)
    if base is None:
        return None

    normalised_base = BASE_NORMALIZE.get(base, base)
    normalised_quote = BASE_NORMALIZE.get(quote, quote)

    return ParsedFile(
        path=Path(),  # placeholder — caller replaces with real path
        raw_pair=raw_pair,
        minutes=minutes,
        base=base,
        quote=quote,
        app_symbol=f"{normalised_base}-{normalised_quote}",
        app_timeframe=TIMEFRAME_MAP[minutes],
    )


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def scan_csv_directory(
    directory: Path,
    base_filter: Optional[set[str]] = None,
) -> list[ParsedFile]:
    """Scan *directory* for CSV files matching the Kraken naming convention.

    Args:
        directory: Path containing CSV files.
        base_filter: Only include files whose **raw** base currency
            (before normalisation) is in this set.  Defaults to
            :data:`DEFAULT_BASE_FILTER`.

    Returns:
        Sorted list of :class:`ParsedFile` objects.
    """
    if base_filter is None:
        base_filter = DEFAULT_BASE_FILTER

    results: list[ParsedFile] = []
    for csv_path in sorted(directory.glob("*.csv")):
        parsed = parse_csv_filename(csv_path.name)
        if parsed is None:
            continue
        if base_filter and parsed.base not in base_filter:
            continue
        # Replace placeholder path with the real one.
        results.append(ParsedFile(
            path=csv_path,
            raw_pair=parsed.raw_pair,
            minutes=parsed.minutes,
            base=parsed.base,
            quote=parsed.quote,
            app_symbol=parsed.app_symbol,
            app_timeframe=parsed.app_timeframe,
        ))

    results.sort(key=lambda p: (p.app_symbol, p.minutes))
    return results


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_csv_file(path: Path) -> pd.DataFrame:
    """Read a Kraken-format CSV file into an app-compatible DataFrame.

    Handles files with or without a header row, drops the ``trades``
    column, and converts Unix-epoch timestamps to ``datetime64[ns, UTC]``.
    Returns an empty DataFrame for zero-byte files.
    """
    if path.stat().st_size == 0:
        logger.warning("Skipping empty file: %s", path)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Peek at the first line to detect a header row.
    with open(path, "r") as fh:
        first_line = fh.readline().strip()

    has_header = first_line.startswith("timestamp")

    df = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else _CSV_COLUMNS,
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
    )

    df.drop(columns=["trades"], errors="ignore", inplace=True)

    # Convert Unix epoch seconds → tz-aware UTC datetime.
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    return df


# ---------------------------------------------------------------------------
# Import orchestrator
# ---------------------------------------------------------------------------

def import_local_csv(
    directory: Path,
    base_filter: Optional[set[str]] = None,
    dry_run: bool = False,
    progress_callback: Optional[Callable] = None,
) -> list[ImportResult]:
    """Bulk-import CSV files from *directory* into the app data store.

    Args:
        directory: Path containing Kraken-format CSV files.
        base_filter: Raw base currencies to include (e.g. ``{"BTC", "XBT", "ETH"}``).
        dry_run: If ``True``, scan and parse only — no data is written.
        progress_callback: Optional ``callback(parsed_file, index, total)``.

    Returns:
        List of :class:`ImportResult` objects.
    """
    from data.ingest import _get_duckdb_store, _get_parquet_store

    files = scan_csv_directory(directory, base_filter=base_filter)

    if not files:
        logger.warning("No matching CSV files found in %s", directory)
        return []

    logger.info("Found %d CSV files to import from %s", len(files), directory)

    if dry_run:
        return [
            ImportResult(
                app_symbol=f.app_symbol,
                app_timeframe=f.app_timeframe,
                source_file=f.path.name,
                rows_imported=0,
                total_rows=0,
                success=True,
            )
            for f in files
        ]

    pq_store = _get_parquet_store()
    db_store = _get_duckdb_store()
    results: list[ImportResult] = []

    for idx, parsed in enumerate(files):
        if progress_callback:
            progress_callback(parsed, idx, len(files))

        try:
            df = read_csv_file(parsed.path)

            if df.empty:
                results.append(ImportResult(
                    app_symbol=parsed.app_symbol,
                    app_timeframe=parsed.app_timeframe,
                    source_file=parsed.path.name,
                    rows_imported=0,
                    total_rows=pq_store.row_count(parsed.app_symbol, parsed.app_timeframe),
                    success=True,
                ))
                continue

            rows_read = len(df)
            total_rows = pq_store.save(parsed.app_symbol, parsed.app_timeframe, df)

            # Update DuckDB metadata (mirrors pattern in data/ingest.py:267-281).
            _update_metadata(db_store, pq_store, parsed.app_symbol, parsed.app_timeframe)

            results.append(ImportResult(
                app_symbol=parsed.app_symbol,
                app_timeframe=parsed.app_timeframe,
                source_file=parsed.path.name,
                rows_imported=rows_read,
                total_rows=total_rows,
                success=True,
            ))

            logger.info(
                "Imported %s -> %s/%s: %d rows read, %d total",
                parsed.path.name,
                parsed.app_symbol,
                parsed.app_timeframe,
                rows_read,
                total_rows,
            )

        except Exception as exc:
            logger.error("Failed to import %s: %s", parsed.path.name, exc)
            results.append(ImportResult(
                app_symbol=parsed.app_symbol,
                app_timeframe=parsed.app_timeframe,
                source_file=parsed.path.name,
                rows_imported=0,
                total_rows=0,
                success=False,
                error=str(exc),
            ))

    return results


def _update_metadata(
    db_store: object,
    pq_store: object,
    symbol: str,
    timeframe: str,
) -> None:
    """Update DuckDB asset and candle_metadata after a successful import."""
    try:
        db_store.upsert_asset(symbol, "crypto", exchange="kraken")
        final = pq_store.load(symbol, timeframe)
        if not final.empty:
            db_store.upsert_candle_metadata(
                symbol=symbol,
                timeframe=timeframe,
                first_ts=final["timestamp"].min().to_pydatetime(),
                last_ts=final["timestamp"].max().to_pydatetime(),
                row_count=len(final),
            )
    except Exception as exc:
        logger.warning("Could not update DuckDB metadata for %s/%s: %s", symbol, timeframe, exc)
