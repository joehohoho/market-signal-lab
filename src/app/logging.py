"""Structured logging setup with rich console output."""

from __future__ import annotations

import logging
from typing import Final

from rich.logging import RichHandler

_DEFAULT_LEVEL: Final[str] = "INFO"
_FORMAT: Final[str] = "%(message)s"
_DATE_FORMAT: Final[str] = "[%X]"

_configured: bool = False


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger with a Rich console handler.

    Safe to call multiple times; only the first call takes effect unless
    the root logger has no handlers (e.g. after a reset).

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Falls back to the MSL_LOG_LEVEL env var, then to INFO.
    """
    global _configured
    if _configured:
        return

    if level is None:
        # Deferred import to avoid circular dependency with config module.
        import os
        level = os.environ.get("MSL_LOG_LEVEL", _DEFAULT_LEVEL)

    resolved_level = getattr(logging, level.upper(), logging.INFO)

    handler = RichHandler(
        level=resolved_level,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging is configured.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    setup_logging()
    return logging.getLogger(name)
