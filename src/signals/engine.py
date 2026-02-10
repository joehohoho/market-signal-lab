"""Signal evaluation engine.

The :class:`SignalEngine` runs strategies across a watchlist of assets,
loading OHLCV data from Parquet files, and collects the latest signal for
each ``(asset, timeframe, strategy)`` combination.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from data.storage.parquet_store import ParquetStore
from strategies import STRATEGY_REGISTRY, get_strategy
from strategies.base import SignalResult

logger = logging.getLogger(__name__)

_DEFAULT_PARQUET_DIR = "data/candles"


class SignalEngine:
    """Orchestrates strategy evaluation across a watchlist of assets.

    For each ``(asset, timeframe)`` pair the engine loads OHLCV data from a
    Parquet store, runs every requested strategy, and collects the latest
    :class:`SignalResult`.

    Args:
        parquet_dir: Root directory containing Parquet candle files.
    """

    def __init__(self, parquet_dir: str = _DEFAULT_PARQUET_DIR) -> None:
        self._parquet_dir = Path(parquet_dir)
        self._store = ParquetStore(parquet_dir)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_candles(self, asset: str, timeframe: str) -> "pd.DataFrame | None":
        """Load OHLCV candles from ParquetStore.

        Returns ``None`` if no data exists or is empty.
        """
        df = self._store.load(asset, timeframe)

        if df.empty:
            logger.warning("No/empty candle data for %s/%s", asset, timeframe)
            return None

        # Ensure a DatetimeIndex.
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        elif "date" in df.columns:
            df = df.set_index("date")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        return df

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has a DatetimeIndex for strategy compute()."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                return df.set_index("timestamp")
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_watchlist(
        self,
        watchlist_config: list[dict[str, Any]],
        strategies_config: dict[str, dict[str, dict[str, Any]]],
    ) -> list[SignalResult]:
        """Run strategies across every asset/timeframe in the watchlist.

        Parameters
        ----------
        watchlist_config:
            A list of dicts, each with at least ``"asset"`` and
            ``"timeframes"`` keys.  Example::

                [
                    {"asset": "BTC-USD", "timeframes": ["1d", "15m"]},
                    {"asset": "AAPL", "timeframes": ["1d"]},
                ]

        strategies_config:
            Nested dict of ``strategy_name -> timeframe -> params``.
            Example::

                {
                    "sma_crossover": {
                        "1d": {"fast_period": 10, "slow_period": 30, ...},
                    },
                }

        Returns
        -------
        list[SignalResult]
            Latest signals for every ``(asset, timeframe, strategy)``
            combination, sorted by strength in descending order.
        """
        all_signals: list[SignalResult] = []

        for entry in watchlist_config:
            asset: str = entry["asset"]
            timeframes: list[str] = entry.get("timeframes", ["1d"])

            for tf in timeframes:
                df = self._load_candles(asset, tf)
                if df is None or len(df) < 2:
                    logger.info(
                        "Skipping %s/%s -- insufficient data", asset, tf,
                    )
                    continue

                for strategy_name, tf_params_map in strategies_config.items():
                    params: dict[str, Any] = tf_params_map.get(tf, {})

                    # Prefer optimized params when available (lazy import
                    # to avoid circular imports).
                    from backtest.optimizer import ParameterOptimizer

                    optimized = ParameterOptimizer.load_optimized(
                        strategy_name, asset, tf,
                    )
                    if optimized is not None:
                        params = {**params, **optimized}

                    try:
                        strategy = get_strategy(strategy_name)
                    except ValueError:
                        logger.warning(
                            "Unknown strategy '%s', skipping", strategy_name,
                        )
                        continue

                    try:
                        result = strategy.latest_signal(df, asset, tf, params)
                        all_signals.append(result)
                        logger.debug(
                            "%s | %s/%s -> %s (%.2f)",
                            strategy_name,
                            asset,
                            tf,
                            result.signal.value,
                            result.strength,
                        )
                    except Exception:
                        logger.exception(
                            "Error running %s on %s/%s",
                            strategy_name,
                            asset,
                            tf,
                        )

        # Sort by strength descending so the strongest signals come first.
        all_signals.sort(key=lambda s: s.strength, reverse=True)
        return all_signals

    def run_single(
        self,
        df: pd.DataFrame,
        asset: str,
        timeframe: str = "1d",
        strategy_names: list[str] | None = None,
        strategies_config: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> list[SignalResult]:
        """Run strategies on an already-loaded DataFrame.

        This is useful when you already have OHLCV data in memory and don't
        need the Parquet loading step.

        Parameters
        ----------
        df:
            OHLCV DataFrame with a DatetimeIndex.
        asset:
            Ticker symbol.
        timeframe:
            Candle interval string.
        strategy_names:
            Subset of strategies to evaluate.  ``None`` runs all registered.
        strategies_config:
            Optional param overrides (same structure as :meth:`run_watchlist`).

        Returns
        -------
        list[SignalResult]
            Latest signal from each strategy, sorted by strength descending.
        """
        if df.empty or len(df) < 2:
            return []

        prepared = self._prepare_df(df)
        names = strategy_names or list(STRATEGY_REGISTRY.keys())
        cfg = strategies_config or {}
        results: list[SignalResult] = []

        for name in names:
            try:
                strategy = get_strategy(name)
            except ValueError:
                logger.warning("Skipping unknown strategy: %s", name)
                continue

            tf_params_map = cfg.get(name, {})
            params: dict[str, Any] = tf_params_map.get(timeframe, {})

            try:
                result = strategy.latest_signal(
                    prepared, asset, timeframe, params,
                )
                results.append(result)
            except Exception:
                logger.exception(
                    "Error running %s on %s/%s", name, asset, timeframe,
                )

        results.sort(key=lambda s: s.strength, reverse=True)
        return results

    def compute_all_bars(
        self,
        df: pd.DataFrame,
        asset: str,
        timeframe: str = "1d",
        strategy_name: str = "sma_crossover",
        params: dict[str, Any] | None = None,
    ) -> list[SignalResult]:
        """Evaluate a single strategy across all bars (for backtesting).

        Args:
            df: OHLCV DataFrame sorted by timestamp ascending.
            asset: Asset symbol.
            timeframe: Candle interval string.
            strategy_name: Which strategy to evaluate.
            params: Strategy parameter overrides.

        Returns:
            A list of :class:`SignalResult`, one per bar.
        """
        if df.empty:
            return []

        prepared = self._prepare_df(df)
        strategy = get_strategy(strategy_name)
        return strategy.compute(prepared, asset, timeframe, params or {})
