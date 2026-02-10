"""Base types for the strategy system.

Every strategy produces :class:`SignalResult` objects that carry the trading
signal, its strength, and an explanation dict describing the reasoning.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Signal(Enum):
    """Discrete trading signal emitted by a strategy."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class SignalResult:
    """Immutable result produced by a strategy for a single bar.

    Attributes:
        signal: The discrete trading action.
        strength: Confidence / magnitude in the range [0.0, 1.0].
                  0.0 means no conviction, 1.0 means maximum conviction.
        strategy_name: Identifier of the strategy that produced this signal.
        asset: Ticker / symbol the signal refers to (e.g. ``"BTC-USD"``).
        timeframe: Candle interval (e.g. ``"1d"``, ``"15m"``).
        timestamp: Bar timestamp.
        explanation: Free-form dict describing why the signal was generated
                     (e.g. indicator values, threshold comparisons).
    """

    signal: Signal
    strength: float = 0.0
    strategy_name: str = ""
    asset: str = ""
    timeframe: str = ""
    timestamp: pd.Timestamp = pd.Timestamp("NaT")
    explanation: dict[str, Any] = field(default_factory=dict)


class Strategy(abc.ABC):
    """Abstract base class for all strategies.

    Subclasses **must** implement :meth:`compute` and expose a ``name``
    property.  :meth:`compute` receives a full OHLCV DataFrame and returns one
    :class:`SignalResult` per bar.  :meth:`latest_signal` is a convenience
    wrapper that returns only the most recent signal.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short, unique identifier for this strategy (e.g. ``"sma_crossover"``)."""
        ...

    @abc.abstractmethod
    def compute(
        self,
        df: pd.DataFrame,
        asset: str,
        timeframe: str,
        params: dict[str, Any],
    ) -> list[SignalResult]:
        """Compute signals for every bar in *df*.

        Parameters
        ----------
        df:
            OHLCV DataFrame with at least ``open``, ``high``, ``low``,
            ``close``, ``volume`` columns and a DatetimeIndex.
        asset:
            Ticker / symbol identifier.
        timeframe:
            Candle timeframe string (e.g. ``"1d"``).
        params:
            Strategy-specific parameter overrides.

        Returns
        -------
        list[SignalResult]
            One :class:`SignalResult` per bar (same length as *df*).
        """
        ...

    def latest_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        timeframe: str,
        params: dict[str, Any],
    ) -> SignalResult:
        """Return the signal for the most recent bar only.

        The default implementation calls :meth:`compute` and returns the last
        element.  Subclasses may override for efficiency.
        """
        signals = self.compute(df, asset, timeframe, params)
        return signals[-1]
