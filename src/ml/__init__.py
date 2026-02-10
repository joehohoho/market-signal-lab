"""Optional ML module for model training, scoring, and signal filtering.

This module requires the ``ml`` optional dependency group::

    pip install market-signal-lab[ml]

If scikit-learn is not installed, stub objects are exported that raise a
helpful :class:`ImportError` when used.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from ml.models import MLModel, MLResult, train_and_validate
    from ml.scorer import MLScorer

    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False
    logger.info(
        "scikit-learn not installed; ML features disabled. "
        "Install with: pip install market-signal-lab[ml]"
    )

    class MLModel:  # type: ignore[no-redef]
        """Stub: scikit-learn is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "scikit-learn is required for MLModel. "
                "Install with: pip install market-signal-lab[ml]"
            )

    class MLScorer:  # type: ignore[no-redef]
        """Stub: scikit-learn is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "scikit-learn is required for MLScorer. "
                "Install with: pip install market-signal-lab[ml]"
            )

    class MLResult:  # type: ignore[no-redef]
        """Stub: scikit-learn is not installed."""
        pass

    def train_and_validate(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        raise ImportError(
            "scikit-learn is required for train_and_validate. "
            "Install with: pip install market-signal-lab[ml]"
        )


__all__ = [
    "MLModel",
    "MLResult",
    "MLScorer",
    "train_and_validate",
    "_ML_AVAILABLE",
]
