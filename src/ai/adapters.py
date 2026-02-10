"""AI adapters for generating plain-english explanations and summaries.

Adapters are **read-only** helpers -- they never generate, modify, or override
trade logic.  Their sole purpose is to explain signals and summarise backtest
results in natural language.

Two concrete adapters are provided:

* :class:`NullAdapter` -- no-op stub returned when AI is not enabled.
* :class:`OllamaAdapter` -- calls a local Ollama server for LLM inference.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AIAdapter(abc.ABC):
    """Abstract base class for AI explanation adapters."""

    @abc.abstractmethod
    def explain_signal(self, signal_result: Any) -> str:
        """Return a plain-english explanation of a trading signal.

        Args:
            signal_result: A :class:`~strategies.base.SignalResult` instance.

        Returns:
            Human-readable explanation string, or empty string on failure.
        """
        ...

    @abc.abstractmethod
    def summarize_backtest(self, backtest_result: Any) -> str:
        """Return a natural-language summary of backtest results.

        Args:
            backtest_result: A backtest result object carrying
                ``equity_curve``, ``trades``, and ``metrics``.

        Returns:
            Human-readable summary string, or empty string on failure.
        """
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check whether the underlying AI service is reachable."""
        ...


# ---------------------------------------------------------------------------
# Null (no-op) adapter
# ---------------------------------------------------------------------------


class NullAdapter(AIAdapter):
    """Stub adapter returned when AI is not configured or available.

    All methods are safe to call but return empty / ``False`` values.
    """

    def explain_signal(self, signal_result: Any) -> str:
        return ""

    def summarize_backtest(self, backtest_result: Any) -> str:
        return "AI not enabled"

    def is_available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Ollama adapter
# ---------------------------------------------------------------------------


class OllamaAdapter(AIAdapter):
    """Adapter that calls a local `Ollama <https://ollama.com>`_ server.

    Requires ``httpx`` (already a project dependency) and a running Ollama
    instance.

    .. important::

        This adapter **never** generates or overrides trade logic.  It only
        produces human-readable explanations and summaries from data that
        has already been computed by the trading system.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_signal(self, signal_result: Any) -> str:
        """Generate a plain-english explanation of a trading signal.

        Args:
            signal_result: A :class:`~strategies.base.SignalResult` instance
                with ``signal``, ``strength``, and ``explanation`` attributes.

        Returns:
            Explanation string, or empty string if the request fails.
        """
        try:
            signal_str = str(signal_result.signal.value)
            strength = signal_result.strength
            explanation = signal_result.explanation
        except AttributeError:
            logger.warning("signal_result does not have expected attributes.")
            return ""

        prompt = (
            "You are a concise trading analyst. Given the following trading signal, "
            "provide a brief plain-english explanation (2-3 sentences) of what the "
            "indicators suggest and why this signal was generated.\n\n"
            f"Signal: {signal_str}\n"
            f"Strength: {strength:.2f}\n"
            f"Indicator values: {explanation}\n\n"
            "Explain what these indicators mean and why they produced this signal."
        )

        return self._generate(prompt)

    def summarize_backtest(self, backtest_result: Any) -> str:
        """Generate a natural-language summary of backtest results.

        Args:
            backtest_result: An object with ``metrics`` (dict), ``trades``
                (list/DataFrame), and ``equity_curve`` attributes.

        Returns:
            Summary string, or empty string if the request fails.
        """
        try:
            metrics = backtest_result.metrics
            n_trades = (
                len(backtest_result.trades)
                if hasattr(backtest_result.trades, "__len__")
                else "unknown"
            )
        except AttributeError:
            logger.warning("backtest_result does not have expected attributes.")
            return ""

        prompt = (
            "You are a concise trading analyst. Summarise the following backtest "
            "results in 3-4 sentences. Focus on risk-adjusted performance, win rate, "
            "and any notable strengths or weaknesses.\n\n"
            f"Total trades: {n_trades}\n"
            f"Metrics: {metrics}\n\n"
            "Provide a brief, actionable summary."
        )

        return self._generate(prompt)

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable.

        Returns:
            ``True`` if the server responds, ``False`` otherwise.
        """
        try:
            import httpx

            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """Send a prompt to the Ollama ``/api/generate`` endpoint.

        Args:
            prompt: The full prompt string.

        Returns:
            The generated text, or empty string on failure.
        """
        try:
            import httpx
        except ImportError:
            logger.warning("httpx is not installed; cannot call Ollama.")
            return ""

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama HTTP error: %s", exc)
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
        except Exception:
            logger.exception("Unexpected error calling Ollama")

        return ""
