"""Market screener that scans a universe of assets and ranks them by signal quality."""

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
# Default stock universe -- popular US + Canada tickers including penny stocks
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
# Composite score weights
# ---------------------------------------------------------------------------
_WEIGHT_SIGNAL_STRENGTH: float = 0.4
_WEIGHT_LIQUIDITY: float = 0.3
_WEIGHT_VOLATILITY_ADJ: float = 0.3


# ---------------------------------------------------------------------------
# ScreenerResult
# ---------------------------------------------------------------------------

@dataclass
class ScreenerResult:
    """A single asset's screening output, ready for ranking.

    Attributes:
        asset: Ticker / symbol.
        price: Most recent close price.
        volume: Most recent bar volume.
        signals: List of :class:`SignalResult` from all evaluated strategies.
        composite_score: Combined ranking score (higher is better).
        liquidity_rank: Normalised volume vs. average volume ratio.
        volatility_adj_score: Signal strength divided by rolling volatility.
    """

    asset: str
    price: float
    volume: float
    signals: list[SignalResult] = field(default_factory=list)
    composite_score: float = 0.0
    liquidity_rank: float = 0.0
    volatility_adj_score: float = 0.0


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------

class Screener:
    """Scan a universe of assets against one or more strategies and rank them.

    Parameters
    ----------
    parquet_dir : str or None
        Override the default parquet directory.  When ``None`` the value is
        read from the application config.
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
    ) -> list[ScreenerResult]:
        """Scan every asset in *universe_name* and return ranked results.

        Parameters
        ----------
        universe_name : str
            Name of the universe defined in the ``screener.universes`` config
            section.  Built-in names: ``"crypto"``, ``"stocks_daily"``.
        timeframe : str
            Candle interval to evaluate (e.g. ``"1d"``).
        strategies : list[str] or None
            Strategy names to run.  When ``None`` every strategy in
            :data:`STRATEGY_REGISTRY` is used.

        Returns
        -------
        list[ScreenerResult]
            Assets that have at least one BUY signal, sorted by
            ``composite_score`` descending.
        """
        assets = self._resolve_universe(universe_name)
        active_strategies = self._resolve_strategies(strategies)

        if not active_strategies:
            logger.warning(
                "No active strategies found (registry has %d entries). "
                "Returning empty scan.",
                len(STRATEGY_REGISTRY),
            )
            return []

        cfg = load_config()
        strategy_params: dict[str, dict[str, Any]] = cfg.get("strategies", {})

        results: list[ScreenerResult] = []

        for asset in assets:
            df = self._store.load(asset, timeframe)
            if df.empty or len(df) < 30:
                logger.debug("Skipping %s: insufficient data (%d rows)", asset, len(df))
                continue

            signals = self._run_strategies(
                df, asset, timeframe, active_strategies, strategy_params,
            )

            # Only keep assets with at least one BUY signal
            buy_signals = [s for s in signals if s.signal == Signal.BUY]
            if not buy_signals:
                continue

            result = self._build_result(df, asset, signals, buy_signals)
            results.append(result)

        # Sort descending by composite score
        results.sort(key=lambda r: r.composite_score, reverse=True)

        logger.info(
            "Screener scan complete: %d/%d assets passed (universe=%s, tf=%s)",
            len(results),
            len(assets),
            universe_name,
            timeframe,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_universe(self, universe_name: str) -> list[str]:
        """Return the list of asset symbols for the named universe."""
        cfg = load_config()
        screener_cfg = cfg.get("screener", {})
        universes_cfg: dict[str, Any] = screener_cfg.get("universes", {})

        if universe_name == "stocks_daily":
            # Use the hardcoded default list (no full ticker API available)
            return list(DEFAULT_STOCK_UNIVERSE)

        universe = universes_cfg.get(universe_name, {})
        asset_list: list[str] = universe.get("assets", [])

        if not asset_list:
            logger.warning(
                "Universe '%s' has no assets configured. "
                "Check screener.universes in config.",
                universe_name,
            )
        return asset_list

    @staticmethod
    def _resolve_strategies(
        names: list[str] | None,
    ) -> list[Strategy]:
        """Instantiate strategy objects from names (or use all registered)."""
        if names is None:
            strategy_classes = list(STRATEGY_REGISTRY.values())
        else:
            strategy_classes = []
            for name in names:
                cls = STRATEGY_REGISTRY.get(name)
                if cls is None:
                    logger.warning("Strategy '%s' not found in registry, skipping.", name)
                else:
                    strategy_classes.append(cls)

        return [cls() for cls in strategy_classes]

    @staticmethod
    def _run_strategies(
        df: pd.DataFrame,
        asset: str,
        timeframe: str,
        active_strategies: list[Strategy],
        strategy_params: dict[str, dict[str, Any]],
    ) -> list[SignalResult]:
        """Evaluate every strategy on the latest bar of *df*."""
        signals: list[SignalResult] = []

        # Ensure DatetimeIndex for strategy compute()
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
    ) -> ScreenerResult:
        """Compute composite score and build a :class:`ScreenerResult`."""
        latest = df.iloc[-1]
        price: float = float(latest["close"])
        volume: float = float(latest["volume"])

        # Average signal strength across BUY signals
        avg_strength = float(np.mean([s.strength for s in buy_signals]))

        # Liquidity proxy: current volume / 20-bar average volume
        avg_volume = float(df["volume"].tail(20).mean())
        liquidity_proxy = volume / avg_volume if avg_volume > 0 else 0.0

        # Volatility (rolling std of log returns, last 20 bars)
        vol_series = calc_volatility(df["close"], period=20)
        current_vol = float(vol_series.iloc[-1]) if not np.isnan(vol_series.iloc[-1]) else 1.0
        vol_adj_score = avg_strength / current_vol if current_vol > 0 else avg_strength

        # Cap the volatility-adjusted score to prevent extreme outliers
        vol_adj_score = min(vol_adj_score, 10.0)

        composite = (
            _WEIGHT_SIGNAL_STRENGTH * avg_strength
            + _WEIGHT_LIQUIDITY * min(liquidity_proxy, 5.0)  # cap at 5x avg
            + _WEIGHT_VOLATILITY_ADJ * vol_adj_score
        )

        return ScreenerResult(
            asset=asset,
            price=price,
            volume=volume,
            signals=all_signals,
            composite_score=composite,
            liquidity_rank=liquidity_proxy,
            volatility_adj_score=vol_adj_score,
        )
