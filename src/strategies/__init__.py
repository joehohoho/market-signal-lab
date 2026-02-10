"""Strategy registry.

Import strategies by name via :func:`get_strategy`, or iterate
:data:`STRATEGY_REGISTRY` for the full set.
"""

from __future__ import annotations

from strategies.base import Signal, SignalResult, Strategy
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.sma_crossover import SMACrossoverStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "sma_crossover": SMACrossoverStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}


def get_strategy(name: str) -> Strategy:
    """Instantiate a strategy by its registered name.

    Args:
        name: Strategy identifier (e.g. ``"sma_crossover"``).

    Returns:
        An instance of the requested :class:`Strategy`.

    Raises:
        ValueError: If no strategy is registered under *name*.
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return cls()


__all__ = [
    "DonchianBreakoutStrategy",
    "RSIMeanReversionStrategy",
    "SMACrossoverStrategy",
    "Signal",
    "SignalResult",
    "Strategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
]
