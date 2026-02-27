"""Alert manager -- evaluate trigger rules and dispatch via SlackAlerter."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

from app.config import load_config
from alerts.slack import SlackAlerter
from strategies.base import SignalResult

logger = logging.getLogger(__name__)

# Rate-limit window: suppress duplicate alerts for the same
# (asset, timeframe) pair within this many seconds.
_RATE_LIMIT_SECONDS: int = 3600  # 1 hour


@dataclass
class _TriggerRule:
    """Parsed representation of a single alert trigger from config."""

    strategy_pattern: str  # glob pattern, e.g. "*" or "sma_crossover"
    timeframes: list[str]
    signals: list[str]  # signal values, e.g. ["BUY", "SELL"]


class AlertManager:
    """Check signals against configured trigger rules and dispatch alerts.

    Alert configuration is loaded from the ``alerts`` section of the
    application config.  If no config is present or alerts are disabled,
    all methods are safe no-ops.

    Rate limiting prevents more than one alert per (asset, timeframe) pair
    per hour.
    """

    def __init__(self) -> None:
        self._alerter = SlackAlerter()
        self._triggers: list[_TriggerRule] = []
        self._enabled: bool = False

        # Rate-limit tracking: (asset, timeframe) -> last alert epoch
        self._last_alert: dict[tuple[str, str], float] = {}

        self._load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_alert(self, signals: list[SignalResult]) -> int:
        """Evaluate *signals* against trigger rules and send matching alerts.

        Parameters
        ----------
        signals : list[SignalResult]
            Signals to evaluate, typically from a screener scan or strategy
            run.

        Returns
        -------
        int
            Number of alerts successfully sent.
        """
        if not self._enabled:
            logger.debug("Alerts are disabled; skipping %d signals.", len(signals))
            return 0

        sent = 0
        for signal in signals:
            if not self._matches_any_trigger(signal):
                continue

            if self._is_rate_limited(signal):
                logger.debug(
                    "Rate-limited: %s/%s (already alerted within the last hour).",
                    signal.asset,
                    signal.timeframe,
                )
                continue

            success = self._alerter.send_signal_alert(signal)
            if success:
                self._record_alert(signal)
                sent += 1

        logger.info(
            "AlertManager processed %d signals, sent %d alerts.", len(signals), sent,
        )
        return sent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load alert trigger rules and enabled flag from application config."""
        try:
            cfg = load_config()
        except Exception:
            logger.debug("Could not load config; alerts disabled.")
            return

        alerts_cfg: dict[str, Any] = cfg.get("alerts", {})
        slack_cfg: dict[str, Any] = alerts_cfg.get("slack", {})

        self._enabled = bool(slack_cfg.get("enabled", False))
        if not self._enabled:
            logger.debug("Slack alerts disabled in config.")
            return

        raw_triggers: list[dict[str, Any]] = slack_cfg.get("triggers", [])
        for entry in raw_triggers:
            rule = _TriggerRule(
                strategy_pattern=str(entry.get("strategy", "*")),
                timeframes=list(entry.get("timeframes", [])),
                signals=[s.upper() for s in entry.get("signals", [])],
            )
            self._triggers.append(rule)

        logger.info(
            "AlertManager loaded %d trigger rule(s), enabled=%s.",
            len(self._triggers),
            self._enabled,
        )

    def _matches_any_trigger(self, signal: SignalResult) -> bool:
        """Return ``True`` if *signal* satisfies at least one trigger rule."""
        signal_direction = signal.signal.value  # "BUY", "SELL", or "HOLD"

        for rule in self._triggers:
            # Strategy pattern match (supports "*" wildcard)
            if not fnmatch(signal.strategy_name, rule.strategy_pattern):
                continue

            # Timeframe match
            if rule.timeframes and signal.timeframe not in rule.timeframes:
                continue

            # Signal direction match
            if rule.signals and signal_direction not in rule.signals:
                continue

            return True

        return False

    def _is_rate_limited(self, signal: SignalResult) -> bool:
        """Return ``True`` if an alert was already sent for this asset/timeframe recently."""
        key = (signal.asset, signal.timeframe)
        last = self._last_alert.get(key)
        if last is None:
            return False
        return (time.monotonic() - last) < _RATE_LIMIT_SECONDS

    def _record_alert(self, signal: SignalResult) -> None:
        """Record that an alert was just sent for rate-limiting purposes."""
        key = (signal.asset, signal.timeframe)
        self._last_alert[key] = time.monotonic()
