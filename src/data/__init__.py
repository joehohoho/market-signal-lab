"""Data layer -- providers, storage, and ingestion.

Quick usage::

    from data import ingest_asset, get_provider

    # Ingest with automatic provider selection
    df = ingest_asset("BTC-USD", "1d", start=datetime(2023, 1, 1))

    # Or use a provider directly
    provider = get_provider("kraken")
    df = provider.fetch_ohlcv("BTC-USD", "1d", start, end)
"""

from data.ingest import ingest_asset
from data.providers import get_provider

__all__ = [
    "ingest_asset",
    "get_provider",
]
