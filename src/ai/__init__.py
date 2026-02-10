"""Optional AI module for natural-language explanations and summaries.

Provides a factory function :func:`get_ai_adapter` that returns the best
available :class:`~ai.adapters.AIAdapter` based on configuration and
runtime availability.

If no AI backend is reachable, a :class:`~ai.adapters.NullAdapter` is
returned so that callers can always safely call adapter methods without
checking for availability first.
"""

from __future__ import annotations

import logging
from typing import Any

from ai.adapters import AIAdapter, NullAdapter, OllamaAdapter

logger = logging.getLogger(__name__)


def get_ai_adapter(config: dict[str, Any] | None = None) -> AIAdapter:
    """Return the best available AI adapter based on configuration.

    Resolution order:

    1. If *config* contains an ``ai.ollama`` section, try to build an
       :class:`OllamaAdapter` and verify connectivity.
    2. Fall back to application config (``app.config.get_config``).
    3. If nothing is configured or reachable, return a :class:`NullAdapter`.

    Args:
        config: Optional configuration dict.  When ``None``, the application
            config is loaded automatically.

    Returns:
        An :class:`AIAdapter` instance ready for use.
    """
    if config is None:
        try:
            from app.config import get_config
            config = get_config()
        except Exception:
            config = {}

    # Try Ollama.
    ollama_cfg = config.get("ai", {}).get("ollama", {})
    if ollama_cfg:
        model = ollama_cfg.get("model", "qwen2.5:7b-instruct")
        base_url = ollama_cfg.get("base_url", "http://localhost:11434")
        adapter = OllamaAdapter(model=model, base_url=base_url)

        if adapter.is_available():
            logger.info("Using Ollama AI adapter (model=%s)", model)
            return adapter
        else:
            logger.info("Ollama not reachable at %s; falling back to NullAdapter.", base_url)

    return NullAdapter()


__all__ = [
    "AIAdapter",
    "NullAdapter",
    "OllamaAdapter",
    "get_ai_adapter",
]
