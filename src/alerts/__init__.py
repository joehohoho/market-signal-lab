"""Alerts module -- dispatch trading signal notifications via Slack."""

from alerts.manager import AlertManager
from alerts.slack import SlackAlerter

__all__ = ["AlertManager", "SlackAlerter"]
