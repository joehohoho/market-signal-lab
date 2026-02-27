"""Parameter optimizer: grid-search strategy parameters via repeated backtests.

The :class:`ParameterOptimizer` iterates over pre-defined parameter grids for
each strategy, runs a :class:`BacktestEngine` for every combination, and ranks
results by Sharpe ratio (primary) and CAGR (secondary).
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.engine import BacktestEngine
from strategies import get_strategy

logger = logging.getLogger(__name__)

# Project root: three levels up from this file (src/backtest/optimizer.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OPTIMIZED_DIR = _PROJECT_ROOT / "data" / "optimized"

# ---------------------------------------------------------------------------
# Parameter grids per strategy
# ---------------------------------------------------------------------------

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "sma_crossover": {
        "fast_period": [5, 8, 10, 12, 15],
        "slow_period": [20, 25, 30, 40, 50],
        "atr_filter_mult": [0.5, 0.8, 1.0, 1.2, 1.5],
        "adx_threshold": [15.0, 20.0, 25.0],
    },
    "rsi_mean_reversion": {
        "rsi_period": [10, 12, 14, 16, 20],
        "rsi_oversold": [20, 25, 30, 35],
        "rsi_overbought": [65, 70, 75, 80],
        "adx_threshold": [20.0, 25.0, 30.0],
    },
    "donchian_breakout": {
        "channel_period": [10, 15, 20, 25, 30],
        "volume_mult": [1.0, 1.3, 1.5, 2.0],
        "adx_threshold": [15.0, 20.0, 25.0],
    },
}


class ParameterOptimizer:
    """Grid-search optimizer for strategy parameters.

    Args:
        strategy_name: Registered strategy name (e.g. ``"sma_crossover"``).
        asset: Asset symbol (e.g. ``"BTC-USD"``).
        timeframe: Candle interval (e.g. ``"1d"``).
        df: OHLCV DataFrame with columns ``timestamp``, ``open``, ``high``,
            ``low``, ``close``, ``volume``.
        fee_preset: Fee preset name for the backtest engine.
        top_n: Number of top results to keep.
    """

    def __init__(
        self,
        strategy_name: str,
        asset: str,
        timeframe: str,
        df: pd.DataFrame,
        fee_preset: str = "crypto_major",
        top_n: int = 5,
    ) -> None:
        self.strategy_name = strategy_name
        self.asset = asset
        self.timeframe = timeframe
        self.df = df
        self.fee_preset = fee_preset
        self.top_n = top_n

        self._results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        """Execute the grid search and return ranked results.

        Returns:
            A list of dicts (up to *top_n*) sorted by Sharpe (primary) and
            CAGR (secondary), each containing::

                {
                    "params": {...},
                    "sharpe": float,
                    "cagr": float,
                    "max_drawdown": float,
                    "win_rate": float,
                    "profit_factor": float,
                    "total_trades": int,
                }
        """
        grid = PARAM_GRIDS.get(self.strategy_name)
        if grid is None:
            available = ", ".join(sorted(PARAM_GRIDS.keys()))
            raise ValueError(
                f"No parameter grid defined for '{self.strategy_name}'. "
                f"Available: {available}"
            )

        strategy = get_strategy(self.strategy_name)
        engine = BacktestEngine(fee_preset=self.fee_preset)

        # Build all param combinations via cartesian product
        param_names = list(grid.keys())
        param_values = [grid[name] for name in param_names]
        combos = list(itertools.product(*param_values))

        logger.info(
            "Optimizing %s on %s/%s: %d combinations",
            self.strategy_name,
            self.asset,
            self.timeframe,
            len(combos),
        )

        results: list[dict[str, Any]] = []

        for combo in combos:
            params = dict(zip(param_names, combo))

            try:
                result = engine.run(
                    df=self.df,
                    strategy=strategy,
                    params=params,
                    asset=self.asset,
                    timeframe=self.timeframe,
                )
            except Exception:
                logger.exception(
                    "Backtest failed for params %s, skipping", params,
                )
                continue

            # Filter out results with fewer than 3 trades (not meaningful)
            if result.total_trades < 3:
                continue

            results.append(
                {
                    "params": params,
                    "sharpe": result.sharpe,
                    "cagr": result.cagr,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "total_trades": result.total_trades,
                }
            )

        # Rank by Sharpe (primary, descending), CAGR (secondary, descending)
        results.sort(key=lambda r: (r["sharpe"], r["cagr"]), reverse=True)

        self._results = results[: self.top_n]
        return self._results

    def save_best(self) -> Path | None:
        """Save the top result to ``data/optimized/{strategy}_{asset}_{tf}.json``.

        Returns:
            The path to the saved JSON file, or ``None`` if no results exist.
        """
        if not self._results:
            logger.warning("No results to save -- run the optimizer first.")
            return None

        _OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

        safe_asset = self.asset.replace("/", "_")
        filename = f"{self.strategy_name}_{safe_asset}_{self.timeframe}.json"
        path = _OPTIMIZED_DIR / filename

        best = self._results[0]
        payload = {
            "strategy": self.strategy_name,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "fee_preset": self.fee_preset,
            "params": best["params"],
            "sharpe": best["sharpe"],
            "cagr": best["cagr"],
            "max_drawdown": best["max_drawdown"],
            "win_rate": best["win_rate"],
            "profit_factor": best["profit_factor"],
            "total_trades": best["total_trades"],
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info("Saved best params to %s", path)
        return path

    @staticmethod
    def load_optimized(
        strategy_name: str,
        asset: str,
        timeframe: str,
    ) -> dict[str, Any] | None:
        """Load previously saved optimized params from disk.

        Args:
            strategy_name: Strategy name.
            asset: Asset symbol.
            timeframe: Candle interval.

        Returns:
            The optimized parameter dict, or ``None`` if no saved file exists.
        """
        safe_asset = asset.replace("/", "_")
        filename = f"{strategy_name}_{safe_asset}_{timeframe}.json"
        path = _OPTIMIZED_DIR / filename

        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("params")
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to load optimized params from %s", path)
            return None
