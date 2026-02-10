"""Slack webhook alerter using Block Kit formatting."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import pandas as pd

from app.config import load_config
from strategies.base import Signal, SignalResult

logger = logging.getLogger(__name__)

# Emoji mapping for signal directions (used inside Slack message text)
_SIGNAL_EMOJI: dict[Signal, str] = {
    Signal.BUY: ":chart_with_upwards_trend:",
    Signal.SELL: ":chart_with_downwards_trend:",
    Signal.HOLD: ":pause_button:",
}


class SlackAlerter:
    """Send trading signal alerts to a Slack channel via webhook.

    The webhook URL is resolved in this order:

    1. *webhook_url* constructor argument.
    2. ``alerts.slack.webhook_url`` in the application config.
    3. ``SLACK_WEBHOOK_URL`` environment variable.

    If no URL is available, :meth:`send_signal_alert` is a no-op that
    returns ``False``.
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        self._webhook_url = self._resolve_url(webhook_url)
        if not self._webhook_url:
            logger.info(
                "SlackAlerter initialised without a webhook URL. "
                "Alerts will be silently skipped."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_signal_alert(self, signal_result: SignalResult) -> bool:
        """Format and POST a signal alert to Slack.

        Parameters
        ----------
        signal_result : SignalResult
            The signal to broadcast.

        Returns
        -------
        bool
            ``True`` if the message was sent successfully, ``False``
            otherwise (including when no webhook URL is configured).
            Alert failures never raise -- they are logged and swallowed.
        """
        if not self._webhook_url:
            logger.debug("No Slack webhook URL configured; skipping alert.")
            return False

        payload = self._build_payload(signal_result)

        try:
            response = httpx.post(
                self._webhook_url,
                json=payload,
                timeout=10.0,
            )
            if response.status_code == 200:
                logger.info(
                    "Slack alert sent: %s %s (%s)",
                    signal_result.signal.value,
                    signal_result.asset,
                    signal_result.strategy_name,
                )
                return True

            logger.warning(
                "Slack webhook returned %d: %s",
                response.status_code,
                response.text[:200],
            )
            return False

        except Exception:
            logger.exception("Failed to send Slack alert for %s", signal_result.asset)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_url(explicit: str | None) -> str:
        """Resolve the webhook URL from constructor arg, config, or env."""
        if explicit:
            return explicit

        try:
            cfg = load_config()
            url = cfg.get("alerts", {}).get("slack", {}).get("webhook_url", "")
            if url:
                return url
        except Exception:
            pass

        return os.environ.get("SLACK_WEBHOOK_URL", "")

    @staticmethod
    def _build_payload(sr: SignalResult) -> dict[str, Any]:
        """Build a Slack Block Kit payload for the given signal."""
        emoji = _SIGNAL_EMOJI.get(sr.signal, "")
        direction = sr.signal.value
        header_text = f"{emoji} {direction} Signal: {sr.asset}"

        # Key indicator fields from the explanation dict
        indicator_lines: list[str] = []
        for key, value in sr.explanation.items():
            if isinstance(value, float):
                indicator_lines.append(f"*{key}:* {value:.4f}")
            else:
                indicator_lines.append(f"*{key}:* {value}")
        indicators_text = "\n".join(indicator_lines) if indicator_lines else "_No details_"

        # Timestamp display (handle pd.NaT gracefully)
        ts = sr.timestamp
        if isinstance(ts, pd.Timestamp) and pd.notna(ts):
            ts_display = ts.strftime("%Y-%m-%d %H:%M UTC")
        elif hasattr(ts, "strftime") and ts is not None:
            ts_display = ts.strftime("%Y-%m-%d %H:%M UTC")
        else:
            ts_display = "N/A"

        local_ui_link = f"http://localhost:8000/asset/{sr.asset}"

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{direction} Signal: {sr.asset}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Strategy:*\n{sr.strategy_name}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Timeframe:*\n{sr.timeframe}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Strength:*\n{sr.strength:.2f}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{ts_display}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Key Indicators:*\n{indicators_text}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"<{local_ui_link}|View in Dashboard>",
                },
            },
            {"type": "divider"},
        ]

        return {
            "text": header_text,  # Fallback for notifications
            "blocks": blocks,
        }
