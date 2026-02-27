"""Paper trading simulator: historical replay with virtual portfolio.

The simulator steps through candles one at a time, maintaining a virtual cash
balance, open positions, and a completed-trade ledger.  It reuses the same
fill model (fees, slippage, spread) as :class:`~backtest.engine.BacktestEngine`
so that paper results are directly comparable to backtests.

State can be saved to / loaded from a JSON file in the ``data/`` directory for
persistence across sessions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import get_config
from strategies.base import Signal, SignalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root for default state storage
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_STATE_DIR = _PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Fee helpers (mirrored from backtest.engine to avoid circular import)
# ---------------------------------------------------------------------------

_DEFAULT_FEE_PRESETS: dict[str, dict[str, float]] = {
    "crypto_major": {
        "fee_pct": 0.001,
        "slippage_pct": 0.0005,
        "spread_pct": 0.0002,
    },
    "liquid_stock": {
        "fee_pct": 0.0,
        "slippage_pct": 0.0003,
        "spread_pct": 0.0001,
    },
    "penny_stock": {
        "fee_pct": 0.0,
        "slippage_pct": 0.005,
        "spread_pct": 0.003,
    },
}


def _load_fee_preset(preset_name: str) -> dict[str, float]:
    """Resolve a fee preset from config, falling back to built-in defaults."""
    try:
        cfg = get_config()
        presets = cfg.get("fee_presets", {})
        if preset_name in presets:
            return {
                "fee_pct": float(presets[preset_name].get("fee_pct", 0.0)),
                "slippage_pct": float(presets[preset_name].get("slippage_pct", 0.0)),
                "spread_pct": float(presets[preset_name].get("spread_pct", 0.0)),
            }
    except Exception:
        pass

    if preset_name in _DEFAULT_FEE_PRESETS:
        return _DEFAULT_FEE_PRESETS[preset_name].copy()

    logger.warning(
        "Unknown fee preset '%s', falling back to liquid_stock defaults",
        preset_name,
    )
    return _DEFAULT_FEE_PRESETS["liquid_stock"].copy()


def _effective_buy_price(base_price: float, fees: dict[str, float]) -> float:
    """Price actually paid per share when buying (higher than market)."""
    return base_price * (
        1.0 + fees["slippage_pct"] + fees["spread_pct"] / 2.0 + fees["fee_pct"]
    )


def _effective_sell_price(base_price: float, fees: dict[str, float]) -> float:
    """Price actually received per share when selling (lower than market)."""
    return base_price * (
        1.0 - fees["slippage_pct"] - fees["spread_pct"] / 2.0 - fees["fee_pct"]
    )


# ---------------------------------------------------------------------------
# Position helper
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    """Internal representation of an open position."""

    shares: float
    entry_price: float
    entry_time: str  # ISO-8601 string for JSON serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _Position:
        return cls(
            shares=float(d["shares"]),
            entry_price=float(d["entry_price"]),
            entry_time=str(d["entry_time"]),
        )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class PaperTradingSimulator:
    """Historical-replay paper trading simulator.

    Parameters:
        initial_capital: Starting cash balance.
        fee_preset: Fee/slippage/spread preset name.
        position_size_pct: Fraction of current equity to allocate per trade.
        state_file: Optional path for JSON persistence.  Defaults to
                    ``data/paper_state.json`` inside the project root.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fee_preset: str = "liquid_stock",
        position_size_pct: float = 0.10,
        state_file: str | Path | None = None,
    ) -> None:
        self._initial_capital = initial_capital
        self._fee_preset = fee_preset
        self._fees = _load_fee_preset(fee_preset)
        self._position_size_pct = position_size_pct

        if state_file is None:
            self._state_path = _DEFAULT_STATE_DIR / "paper_state.json"
        else:
            self._state_path = Path(state_file)

        # Portfolio state
        self._cash: float = initial_capital
        self._positions: dict[str, _Position] = {}  # asset -> Position
        self._trade_ledger: list[dict[str, Any]] = []
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, initial_capital: float | None = None) -> None:
        """Reset the simulator to a clean state.

        Args:
            initial_capital: New starting capital.  When *None*, reuses the
                             value from ``__init__``.
        """
        if initial_capital is not None:
            self._initial_capital = initial_capital
        self._cash = self._initial_capital
        self._positions.clear()
        self._trade_ledger.clear()
        self._step_count = 0
        logger.info("Paper simulator reset with capital=%.2f", self._initial_capital)

    def step(
        self,
        candle: dict[str, Any] | pd.Series,
        signals: dict[str, SignalResult],
    ) -> list[dict[str, Any]]:
        """Advance the simulator by one candle and process signals.

        Signals are filled at the candle's ``open`` price (simulating a
        market order placed at the previous bar's close, filled at the next
        bar's open).

        Args:
            candle: A single OHLCV bar as a dict or pandas Series.  Must
                    contain keys ``open``, ``high``, ``low``, ``close``,
                    ``volume``, and ``timestamp``.
            signals: Mapping of ``{asset: SignalResult}`` with the signals
                     to execute on this step.

        Returns:
            List of trade dicts that were executed during this step (can be
            empty).
        """
        self._step_count += 1
        executed: list[dict[str, Any]] = []

        price_open = float(candle["open"])
        timestamp = str(candle["timestamp"])

        for asset, signal_result in signals.items():
            sig = signal_result.signal

            if sig == Signal.BUY and asset not in self._positions:
                # Open a long position
                buy_price = _effective_buy_price(price_open, self._fees)
                if buy_price <= 0:
                    continue

                # Use position_size_pct of current total equity estimate
                # (cash only, since we use open before position exists)
                allocation = self._cash * self._position_size_pct
                if allocation <= 0:
                    continue

                shares = allocation / buy_price
                cost = shares * buy_price
                self._cash -= cost

                self._positions[asset] = _Position(
                    shares=shares,
                    entry_price=buy_price,
                    entry_time=timestamp,
                )

                trade = {
                    "asset": asset,
                    "action": "BUY",
                    "timestamp": timestamp,
                    "price": buy_price,
                    "shares": shares,
                    "cost": cost,
                    "step": self._step_count,
                }
                executed.append(trade)
                logger.debug("PAPER BUY %s: %.4f shares @ %.4f", asset, shares, buy_price)

            elif sig == Signal.SELL and asset in self._positions:
                # Close the position
                pos = self._positions.pop(asset)
                sell_price = _effective_sell_price(price_open, self._fees)
                proceeds = pos.shares * sell_price
                pnl = proceeds - pos.shares * pos.entry_price
                pnl_pct = (sell_price / pos.entry_price - 1.0) if pos.entry_price > 0 else 0.0
                self._cash += proceeds

                trade = {
                    "asset": asset,
                    "action": "SELL",
                    "timestamp": timestamp,
                    "entry_time": pos.entry_time,
                    "entry_price": pos.entry_price,
                    "exit_price": sell_price,
                    "shares": pos.shares,
                    "proceeds": proceeds,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "step": self._step_count,
                    "side": "long",
                }
                self._trade_ledger.append(trade)
                executed.append(trade)
                logger.debug(
                    "PAPER SELL %s: %.4f shares @ %.4f  PnL=%.2f (%.2f%%)",
                    asset,
                    pos.shares,
                    sell_price,
                    pnl,
                    pnl_pct * 100,
                )

        return executed

    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Compute total portfolio value given current market prices.

        Args:
            current_prices: Mapping of ``{asset: price}`` for every held
                            asset.  Assets missing from the dict are valued
                            at their entry price (stale price fallback).

        Returns:
            Total portfolio value (cash + mark-to-market positions).
        """
        positions_value = 0.0
        for asset, pos in self._positions.items():
            price = current_prices.get(asset, pos.entry_price)
            positions_value += pos.shares * price
        return self._cash + positions_value

    def get_positions(self) -> dict[str, dict[str, Any]]:
        """Return a dict of currently open positions.

        Returns:
            ``{asset: {"shares": ..., "entry_price": ..., "entry_time": ...}}``
        """
        return {asset: pos.to_dict() for asset, pos in self._positions.items()}

    def get_trade_ledger(self) -> list[dict[str, Any]]:
        """Return the list of all completed (closed) trades."""
        return list(self._trade_ledger)

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def step_count(self) -> int:
        """Number of candles processed so far."""
        return self._step_count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path | None = None) -> Path:
        """Serialise the current state to a JSON file.

        Args:
            path: Target file path.  When *None*, uses the default
                  ``data/paper_state.json``.

        Returns:
            The :class:`~pathlib.Path` that was written.
        """
        target = Path(path) if path is not None else self._state_path
        target.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "initial_capital": self._initial_capital,
            "cash": self._cash,
            "fee_preset": self._fee_preset,
            "position_size_pct": self._position_size_pct,
            "step_count": self._step_count,
            "positions": {
                asset: pos.to_dict() for asset, pos in self._positions.items()
            },
            "trade_ledger": self._trade_ledger,
        }

        with open(target, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info("Paper state saved to %s", target)
        return target

    def load_state(self, path: str | Path | None = None) -> None:
        """Restore state from a JSON file.

        Args:
            path: Source file path.  When *None*, uses the default
                  ``data/paper_state.json``.

        Raises:
            FileNotFoundError: If the state file does not exist.
        """
        source = Path(path) if path is not None else self._state_path

        if not source.exists():
            raise FileNotFoundError(f"Paper state file not found: {source}")

        with open(source) as f:
            state = json.load(f)

        self._initial_capital = float(state["initial_capital"])
        self._cash = float(state["cash"])
        self._fee_preset = state.get("fee_preset", self._fee_preset)
        self._fees = _load_fee_preset(self._fee_preset)
        self._position_size_pct = float(
            state.get("position_size_pct", self._position_size_pct)
        )
        self._step_count = int(state.get("step_count", 0))

        self._positions = {
            asset: _Position.from_dict(d)
            for asset, d in state.get("positions", {}).items()
        }
        self._trade_ledger = state.get("trade_ledger", [])

        logger.info(
            "Paper state loaded from %s (cash=%.2f, %d positions, %d past trades)",
            source,
            self._cash,
            len(self._positions),
            len(self._trade_ledger),
        )
