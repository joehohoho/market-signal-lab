"""Market screener with configurable scoring weights and relative strength."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.config import load_config
from data.storage.parquet_store import ParquetStore
from indicators.core import volatility as calc_volatility
from strategies import STRATEGY_REGISTRY, Strategy
from strategies.base import Signal, SignalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default stock universe
# ---------------------------------------------------------------------------

DEFAULT_STOCK_UNIVERSE: list[str] = [
    # US large-cap / mega-cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "V", "UNH", "JNJ", "WMT", "PG", "MA", "HD",
    # US mid-cap / growth
    "SQ", "PLTR", "SNAP", "ROKU", "SOFI",
    # Canada
    "SHOP.TO", "RY.TO", "TD.TO", "ENB.TO", "CNQ.TO",
    # Penny stocks (typically < $5)
    "SNDL", "CLOV", "TELL", "BBIG", "MULN",
]

# ---------------------------------------------------------------------------
# Default composite score weights (overridable via config)
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS: dict[str, float] = {
    "signal_strength": 0.30,
    "liquidity": 0.20,
    "volatility_adj": 0.20,
    "relative_strength": 0.30,
}


# ---------------------------------------------------------------------------
# ScreenerResult
# ---------------------------------------------------------------------------

@dataclass
class ScreenerResult:
    """A single asset's screening output, ready for ranking."""

    asset: str
    price: float
    volume: float
    signals: list[SignalResult] = field(default_factory=list)
    composite_score: float = 0.0
    liquidity_rank: float = 0.0
    volatility_adj_score: float = 0.0
    relative_strength: float = 0.0


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------

class Screener:
    """Scan a universe of assets against strategies and rank them.

    Supports configurable scoring weights and relative strength ranking.
    """

    def __init__(self, parquet_dir: str | None = None) -> None:
        cfg = load_config()
        storage_cfg = cfg.get("storage", {})
        resolved_dir = parquet_dir or storage_cfg.get("parquet_dir", "data/candles")
        self._store = ParquetStore(resolved_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        universe_name: str = "crypto",
        timeframe: str = "1d",
        strategies: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[ScreenerResult]:
        """Scan every asset in *universe_name* and return ranked results.

        Parameters
        ----------
        universe_name : str
            Universe name from config or built-in.
        timeframe : str
            Candle interval to evaluate.
        strategies : list[str] or None
            Strategy names to run (None = all).
        weights : dict or None
            Override composite score weights. Keys: ``signal_strength``,
            ``liquidity``, ``volatility_adj``, ``relative_strength``.
        """
        assets = self._resolve_universe(universe_name)
        active_strategies = self._resolve_strategies(strategies)

        if not active_strategies:
            logger.warning("No active strategies. Returning empty scan.")
            return []

        # Load weights from config or use defaults
        w = self._resolve_weights(weights)

        cfg = load_config()
        strategy_params: dict[str, dict[str, Any]] = cfg.get("strategies", {})

        # Phase 1: Gather data and compute signals + returns for all assets
        asset_data: list[dict[str, Any]] = []

        for asset in assets:
            df = self._store.load(asset, timeframe)
            if df.empty or len(df) < 30:
                logger.debug("Skipping %s: insufficient data (%d rows)", asset, len(df))
                continue

            signals = self._run_strategies(
                df, asset, timeframe, active_strategies, strategy_params,
            )

            buy_signals = [s for s in signals if s.signal == Signal.BUY]
            if not buy_signals:
                continue

            # Compute return for relative strength ranking
            close = df["close"]
            n_bars = min(60, len(close) - 1)  # ~3 months for daily
            ret = (close.iloc[-1] / close.iloc[-n_bars] - 1.0) if n_bars > 0 else 0.0

            asset_data.append({
                "asset": asset,
                "df": df,
                "signals": signals,
                "buy_signals": buy_signals,
                "return_60": float(ret),
            })

        if not asset_data:
            return []

        # Phase 2: Compute relative strength rank (percentile within universe)
        returns = [d["return_60"] for d in asset_data]
        sorted_returns = sorted(returns)
        n = len(sorted_returns)

        for d in asset_data:
            rank_pos = sorted_returns.index(d["return_60"])
            d["relative_strength_pct"] = rank_pos / max(n - 1, 1)  # 0 to 1

        # Phase 3: Build scored results
        results: list[ScreenerResult] = []
        for d in asset_data:
            result = self._build_result(
                d["df"], d["asset"], d["signals"], d["buy_signals"],
                d["relative_strength_pct"], w,
            )
            results.append(result)

        results.sort(key=lambda r: r.composite_score, reverse=True)

        logger.info(
            "Screener scan: %d/%d assets passed (universe=%s, tf=%s)",
            len(results), len(assets), universe_name, timeframe,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_universe(self, universe_name: str) -> list[str]:
        cfg = load_config()
        screener_cfg = cfg.get("screener", {})
        universes_cfg: dict[str, Any] = screener_cfg.get("universes", {})

        if universe_name == "stocks_daily":
            return list(DEFAULT_STOCK_UNIVERSE)

        universe = universes_cfg.get(universe_name, {})
        asset_list: list[str] = universe.get("assets", [])
        if not asset_list:
            logger.warning("Universe '%s' has no assets.", universe_name)
        return asset_list

    @staticmethod
    def _resolve_strategies(names: list[str] | None) -> list[Strategy]:
        if names is None:
            strategy_classes = list(STRATEGY_REGISTRY.values())
        else:
            strategy_classes = []
            for name in names:
                cls = STRATEGY_REGISTRY.get(name)
                if cls is None:
                    logger.warning("Strategy '%s' not found, skipping.", name)
                else:
                    strategy_classes.append(cls)
        return [cls() for cls in strategy_classes]

    @staticmethod
    def _resolve_weights(overrides: dict[str, float] | None) -> dict[str, float]:
        """Merge user/config weights with defaults."""
        cfg = load_config()
        config_weights = cfg.get("screener", {}).get("weights", {})
        w = {**_DEFAULT_WEIGHTS}
        if config_weights:
            w.update(config_weights)
        if overrides:
            w.update(overrides)

        # Normalise so weights sum to 1.0
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
        return w

    @staticmethod
    def _run_strategies(
        df: pd.DataFrame,
        asset: str,
        timeframe: str,
        active_strategies: list[Strategy],
        strategy_params: dict[str, dict[str, Any]],
    ) -> list[SignalResult]:
        signals: list[SignalResult] = []

        prepared = df
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                prepared = df.set_index("timestamp")

        for strat in active_strategies:
            params = strategy_params.get(strat.name, {}).get(timeframe, {})
            try:
                result = strat.latest_signal(prepared, asset, timeframe, params)
                signals.append(result)
            except Exception:
                logger.exception(
                    "Strategy '%s' failed on %s/%s", strat.name, asset, timeframe,
                )
        return signals

    @staticmethod
    def _build_result(
        df: pd.DataFrame,
        asset: str,
        all_signals: list[SignalResult],
        buy_signals: list[SignalResult],
        relative_strength_pct: float,
        weights: dict[str, float],
    ) -> ScreenerResult:
        latest = df.iloc[-1]
        price: float = float(latest["close"])
        volume: float = float(latest["volume"])

        # Signal strength
        avg_strength = float(np.mean([s.strength for s in buy_signals]))

        # Liquidity proxy
        avg_volume = float(df["volume"].tail(20).mean())
        liquidity_proxy = volume / avg_volume if avg_volume > 0 else 0.0

        # Volatility-adjusted score
        vol_series = calc_volatility(df["close"], period=20)
        current_vol = float(vol_series.iloc[-1]) if not np.isnan(vol_series.iloc[-1]) else 1.0
        vol_adj_score = avg_strength / current_vol if current_vol > 0 else avg_strength
        vol_adj_score = min(vol_adj_score, 10.0)

        # Composite score with configurable weights
        composite = (
            weights.get("signal_strength", 0.3) * avg_strength
            + weights.get("liquidity", 0.2) * min(liquidity_proxy, 5.0)
            + weights.get("volatility_adj", 0.2) * vol_adj_score
            + weights.get("relative_strength", 0.3) * relative_strength_pct
        )

        return ScreenerResult(
            asset=asset,
            price=price,
            volume=volume,
            signals=all_signals,
            composite_score=composite,
            liquidity_rank=liquidity_proxy,
            volatility_adj_score=vol_adj_score,
            relative_strength=relative_strength_pct,
        )
