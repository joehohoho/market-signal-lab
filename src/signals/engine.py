"""Signal evaluation engine with multi-timeframe confirmation.

The :class:`SignalEngine` runs strategies across a watchlist of assets,
loading OHLCV data from Parquet files, and collects the latest signal for
each ``(asset, timeframe, strategy)`` combination.

Multi-timeframe mode requires directional alignment between a higher
timeframe (bias) and lower timeframe (entry trigger).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from data.storage.parquet_store import ParquetStore
from strategies import STRATEGY_REGISTRY, get_strategy
from strategies.base import Signal, SignalResult

logger = logging.getLogger(__name__)

_DEFAULT_PARQUET_DIR = "data/candles"

# Higher-timeframe mapping for multi-timeframe confirmation
_HIGHER_TF_MAP: dict[str, str] = {
    "1m": "15m",
    "5m": "1h",
    "15m": "4h",
    "30m": "4h",
    "1h": "1d",
    "4h": "1d",
    "1d": "1w",
}


class SignalEngine:
    """Orchestrates strategy evaluation across a watchlist of assets.

    Supports single-timeframe and multi-timeframe confirmation modes.
    """

    def __init__(self, parquet_dir: str = _DEFAULT_PARQUET_DIR) -> None:
        self._parquet_dir = Path(parquet_dir)
        self._store = ParquetStore(parquet_dir)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_candles(self, asset: str, timeframe: str) -> pd.DataFrame | None:
        """Load OHLCV candles from ParquetStore."""
        df = self._store.load(asset, timeframe)

        if df.empty:
            logger.warning("No/empty candle data for %s/%s", asset, timeframe)
            return None

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
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                return df.set_index("timestamp")
        return df

    # ------------------------------------------------------------------
    # Multi-timeframe bias
    # ------------------------------------------------------------------

    def _get_higher_tf_bias(
        self,
        asset: str,
        timeframe: str,
        strategies_config: dict[str, dict[str, dict[str, Any]]],
    ) -> Signal | None:
        """Determine directional bias from the higher timeframe.

        Loads the higher-TF data, runs all strategies, and returns the
        consensus direction.  BUY if majority bullish, SELL if majority
        bearish, HOLD if mixed/neutral.

        Returns None if higher-TF data is unavailable.
        """
        higher_tf = _HIGHER_TF_MAP.get(timeframe)
        if not higher_tf:
            return None

        df = self._load_candles(asset, higher_tf)
        if df is None or len(df) < 30:
            return None

        # Run only the active strategies (those in strategies_config) on higher TF.
        # Iterating STRATEGY_REGISTRY would silently include out-of-run strategies,
        # breaking reproducibility.
        buy_count = 0
        sell_count = 0

        for strategy_name in strategies_config:
            try:
                strategy = get_strategy(strategy_name)
                tf_params = strategies_config[strategy_name].get(higher_tf, {})
                result = strategy.latest_signal(df, asset, higher_tf, tf_params)
                if result.signal == Signal.BUY:
                    buy_count += 1
                elif result.signal == Signal.SELL:
                    sell_count += 1
            except Exception:
                continue

        if buy_count > sell_count and buy_count > 0:
            return Signal.BUY
        elif sell_count > buy_count and sell_count > 0:
            return Signal.SELL
        return Signal.HOLD

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_watchlist(
        self,
        watchlist_config: list[dict[str, Any]],
        strategies_config: dict[str, dict[str, dict[str, Any]]],
        multi_timeframe: bool = False,
    ) -> list[SignalResult]:
        """Run strategies across every asset/timeframe in the watchlist.

        Parameters
        ----------
        watchlist_config:
            A list of dicts with ``"asset"`` and ``"timeframes"`` keys.
        strategies_config:
            Nested dict of ``strategy_name -> timeframe -> params``.
        multi_timeframe:
            When True, signals are filtered by higher-timeframe directional
            alignment.  BUY signals require bullish higher-TF bias; SELL
            signals require bearish higher-TF bias.

        Returns
        -------
        list[SignalResult]
            Sorted by strength descending.
        """
        all_signals: list[SignalResult] = []

        for entry in watchlist_config:
            asset: str = entry["asset"]
            timeframes: list[str] = entry.get("timeframes", ["1d"])

            for tf in timeframes:
                df = self._load_candles(asset, tf)
                if df is None or len(df) < 2:
                    logger.info("Skipping %s/%s -- insufficient data", asset, tf)
                    continue

                # Get higher-TF bias if multi-timeframe enabled
                htf_bias: Signal | None = None
                if multi_timeframe:
                    htf_bias = self._get_higher_tf_bias(asset, tf, strategies_config)

                for strategy_name, tf_params_map in strategies_config.items():
                    params: dict[str, Any] = tf_params_map.get(tf, {})

                    from backtest.optimizer import ParameterOptimizer
                    optimized = ParameterOptimizer.load_optimized(strategy_name, asset, tf)
                    if optimized is not None:
                        params = {**params, **optimized}

                    try:
                        strategy = get_strategy(strategy_name)
                    except ValueError:
                        logger.warning("Unknown strategy '%s', skipping", strategy_name)
                        continue

                    try:
                        result = strategy.latest_signal(df, asset, tf, params)

                        # Multi-timeframe filter
                        if multi_timeframe and htf_bias is not None:
                            if result.signal == Signal.BUY and htf_bias == Signal.SELL:
                                result = SignalResult(
                                    signal=Signal.HOLD, strength=0.0,
                                    strategy_name=result.strategy_name,
                                    asset=asset, timeframe=tf,
                                    timestamp=result.timestamp,
                                    explanation={
                                        **result.explanation,
                                        "mtf_filtered": True,
                                        "htf_bias": "bearish",
                                    },
                                )
                            elif result.signal == Signal.SELL and htf_bias == Signal.BUY:
                                result = SignalResult(
                                    signal=Signal.HOLD, strength=0.0,
                                    strategy_name=result.strategy_name,
                                    asset=asset, timeframe=tf,
                                    timestamp=result.timestamp,
                                    explanation={
                                        **result.explanation,
                                        "mtf_filtered": True,
                                        "htf_bias": "bullish",
                                    },
                                )

                        all_signals.append(result)
                        logger.debug(
                            "%s | %s/%s -> %s (%.2f)%s",
                            strategy_name, asset, tf,
                            result.signal.value, result.strength,
                            f" [HTF bias: {htf_bias.value}]" if htf_bias else "",
                        )
                    except Exception:
                        logger.exception(
                            "Error running %s on %s/%s", strategy_name, asset, tf,
                        )

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
        """Run strategies on an already-loaded DataFrame."""
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
                result = strategy.latest_signal(prepared, asset, timeframe, params)
                results.append(result)
            except Exception:
                logger.exception("Error running %s on %s/%s", name, asset, timeframe)

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
        """Evaluate a single strategy across all bars (for backtesting)."""
        if df.empty:
            return []

        prepared = self._prepare_df(df)
        strategy = get_strategy(strategy_name)
        return strategy.compute(prepared, asset, timeframe, params or {})
