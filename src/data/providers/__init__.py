"""Data providers for OHLCV market data.

Use :func:`get_provider` to obtain a provider instance by name::

    from data.providers import get_provider

    provider = get_provider("kraken")
    df = provider.fetch_ohlcv("BTC-USD", "1d", start, end)
"""

from __future__ import annotations

from data.providers.base import DataProvider
from data.providers.kraken import KrakenProvider
from data.providers.yahoo_daily import YahooDailyProvider

# Registry mapping provider name -> class.
_PROVIDER_REGISTRY: dict[str, type[DataProvider]] = {
    "kraken": KrakenProvider,
    "yahoo_daily": YahooDailyProvider,
}


def get_provider(name: str) -> DataProvider:
    """Instantiate and return a data provider by name.

    Args:
        name: Provider identifier (e.g. ``"kraken"``, ``"yahoo_daily"``).

    Returns:
        An instance of the requested :class:`DataProvider`.

    Raises:
        ValueError: If no provider is registered under *name*.
    """
    cls = _PROVIDER_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider '{name}'. Available providers: {available}"
        )
    return cls()


__all__ = [
    "DataProvider",
    "KrakenProvider",
    "YahooDailyProvider",
    "get_provider",
]
