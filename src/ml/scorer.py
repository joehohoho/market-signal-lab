"""ML scoring and integration helpers.

:class:`MLScorer` loads a trained model and produces probability scores for
new OHLCV data.  Integration helpers allow signals and screener scores to
be filtered or boosted based on ML confidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ml.features import build_features

logger = logging.getLogger(__name__)


class MLScorer:
    """Score new market data using a trained ML model.

    Usage::

        scorer = MLScorer()
        if scorer.load_model("BTC-USD", "1d"):
            prob = scorer.score(ohlcv_df)
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model: Any = None  # ml.models.MLModel when loaded
        self._model_path = Path(model_path) if model_path else None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_model(self, asset: str, timeframe: str) -> bool:
        """Attempt to load a persisted model for *asset* / *timeframe*.

        If a ``model_path`` was supplied at construction it is used directly;
        otherwise the method looks in ``models/<asset>_<timeframe>.joblib``
        relative to the current working directory.

        Args:
            asset: Asset identifier (e.g. ``"BTC-USD"``).
            timeframe: Candle interval string (e.g. ``"1d"``).

        Returns:
            ``True`` if the model was loaded successfully, ``False`` otherwise.
        """
        try:
            from ml.models import MLModel
        except ImportError:
            logger.warning("scikit-learn not available; ML scoring disabled.")
            return False

        path = self._model_path
        if path is None:
            # Convention: models/<asset>_<timeframe>.joblib relative to project root
            safe_asset = asset.replace("/", "-").replace("\\", "-")
            project_root = Path(__file__).resolve().parent.parent.parent
            path = project_root / "models" / f"{safe_asset}_{timeframe}.joblib"

        if not path.exists():
            logger.info("No model found at %s", path)
            return False

        try:
            self._model = MLModel().load(path)
            logger.info("Loaded ML model from %s", path)
            return True
        except Exception:
            logger.exception("Failed to load model from %s", path)
            return False

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> float:
        """Return the ML probability score for the latest bar in *df*.

        Args:
            df: OHLCV DataFrame (must contain enough history for feature
                computation).

        Returns:
            Probability of a positive forward return for the most recent bar.

        Raises:
            RuntimeError: If no model has been loaded.
        """
        if self._model is None:
            raise RuntimeError("No ML model loaded. Call load_model() first.")

        feat_df = build_features(df)

        if feat_df.empty:
            logger.warning("Feature DataFrame is empty after build; returning 0.5.")
            return 0.5

        # Use only the feature columns the model was trained on.
        selected = self._model.metadata.get("selected_features", None)
        if selected:
            feature_cols = [c for c in selected if c in feat_df.columns]
        else:
            meta_cols = {"timestamp", "close"}
            feature_cols = [c for c in feat_df.columns if c not in meta_cols]
        X = feat_df[feature_cols].iloc[[-1]]  # last row only

        proba = self._model.predict_proba(X)
        return float(proba[0])


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def filter_signal(
    signal_result: Any,
    df: pd.DataFrame,
    scorer: MLScorer | None = None,
    threshold: float = 0.55,
) -> Any | None:
    """Optionally filter a signal through the ML model.

    If a scorer is provided and loaded, the signal is only passed through
    when the ML probability meets or exceeds *threshold*.  If the scorer is
    ``None`` or has no model loaded, the signal is returned unchanged (ML
    is opt-in, never blocks by default).

    Args:
        signal_result: A :class:`~strategies.base.SignalResult` instance.
        df: OHLCV DataFrame used to compute the ML score.
        scorer: An :class:`MLScorer` instance (may be ``None``).
        threshold: Minimum ML probability to keep the signal.

    Returns:
        The original *signal_result* if it passes the filter, or ``None``
        if the ML model filtered it out.
    """
    if scorer is None or scorer._model is None:
        return signal_result

    try:
        ml_score = scorer.score(df)
    except Exception:
        logger.exception("ML scoring failed; passing signal through.")
        return signal_result

    if ml_score >= threshold:
        logger.debug("ML filter passed (score=%.4f >= threshold=%.4f)", ml_score, threshold)
        return signal_result

    logger.info(
        "ML filter rejected signal (score=%.4f < threshold=%.4f)",
        ml_score, threshold,
    )
    return None


def boost_screener_score(
    base_score: float,
    ml_score: float,
    weight: float = 0.2,
) -> float:
    """Blend a base screener score with an ML probability.

    The formula is a weighted combination::

        final = (1 - weight) * base_score + weight * ml_score

    Args:
        base_score: Original screener score (0-1 range expected).
        ml_score: ML model probability (0-1 range).
        weight: How much influence the ML score has (default 0.2).

    Returns:
        Blended score in the [0, 1] range, clamped.
    """
    blended = (1.0 - weight) * base_score + weight * ml_score
    return max(0.0, min(1.0, blended))
