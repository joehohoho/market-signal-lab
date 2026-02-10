"""ML model training, walk-forward validation, and prediction.

Supports logistic regression (baseline) and gradient-boosting (main) models
through :class:`MLModel`.  Training is performed via temporal walk-forward
validation with :class:`WalkForwardValidator` and orchestrated by
:func:`train_and_validate`.

All scikit-learn imports are deferred so that the module can be imported
even when sklearn is not installed -- callers will get a clear error at
runtime if they attempt to train without the ``ml`` extra.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.features import build_features, build_target

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk-forward temporal splitter
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Expanding-window temporal cross-validator.

    Each successive window uses a *longer* training set (all data up to the
    split point) and the same-sized test set immediately following it.
    """

    @staticmethod
    def split(
        X: pd.DataFrame | np.ndarray,
        n_windows: int = 5,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate expanding-window train/test index pairs.

        Args:
            X: Feature matrix (only its length is used).
            n_windows: Number of validation windows.

        Returns:
            A list of ``(train_indices, test_indices)`` tuples.

        Raises:
            ValueError: If *n_windows* is less than 2 or the dataset is too
                small to create the requested number of windows.
        """
        n = len(X)
        if n_windows < 2:
            raise ValueError("n_windows must be >= 2")

        # Reserve at least 20% of data for the first training set.
        min_train = max(int(n * 0.2), 1)
        remaining = n - min_train
        if remaining < n_windows:
            raise ValueError(
                f"Not enough data ({n} rows) for {n_windows} windows with "
                f"minimum training size {min_train}."
            )

        step = remaining // n_windows
        splits: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(n_windows):
            split_point = min_train + i * step
            test_end = min(split_point + step, n)
            train_idx = np.arange(0, split_point)
            test_idx = np.arange(split_point, test_end)
            if len(test_idx) == 0:
                continue
            splits.append((train_idx, test_idx))

        return splits


# ---------------------------------------------------------------------------
# MLModel wrapper
# ---------------------------------------------------------------------------


class MLModel:
    """Thin wrapper around scikit-learn classifiers.

    Supports ``"logistic"`` (LogisticRegression) and
    ``"gradient_boosting"`` (HistGradientBoostingClassifier) model types.
    """

    def __init__(self, model_type: str = "gradient_boosting") -> None:
        if model_type not in ("logistic", "gradient_boosting"):
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                "Choose 'logistic' or 'gradient_boosting'."
            )
        self.model_type = model_type
        self._model: Any = None
        self.feature_names: list[str] = []

    # -- training ----------------------------------------------------------

    def train(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray) -> "MLModel":
        """Fit the model on training data.

        Args:
            X_train: Feature matrix.
            y_train: Binary target vector.

        Returns:
            ``self`` for method chaining.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for ML training. "
                "Install with: pip install market-signal-lab[ml]"
            ) from exc

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        if self.model_type == "logistic":
            self._model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ])
        else:
            self._model = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
            )

        self._model.fit(X_train, y_train)
        return self

    # -- inference ---------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return probability of the positive class for each row.

        Args:
            X: Feature matrix with the same columns used during training.

        Returns:
            1-D array of probabilities for the positive (class=1) label.
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained yet.")
        proba = self._model.predict_proba(X)
        return proba[:, 1]

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the model to disk using joblib.

        Args:
            path: Destination file path (e.g. ``"models/gb_btc_1d.joblib"``).
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "joblib is required to save models. "
                "Install with: pip install market-signal-lab[ml]"
            ) from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "feature_names": self.feature_names, "model_type": self.model_type},
            path,
        )
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> "MLModel":
        """Load a previously saved model from disk.

        Args:
            path: Path to the saved joblib file.

        Returns:
            ``self`` for method chaining.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "joblib is required to load models. "
                "Install with: pip install market-signal-lab[ml]"
            ) from exc

        data = joblib.load(path)
        self._model = data["model"]
        self.feature_names = data["feature_names"]
        self.model_type = data["model_type"]
        logger.info("Model loaded from %s", path)
        return self


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MLResult:
    """Result of walk-forward model training and validation.

    Attributes:
        model: The trained :class:`MLModel` (main model from the last window).
        metrics_per_window: List of dicts containing ``'auc'``, ``'window'``
            and other per-window metrics.
        is_stable: Whether the main model passed the stability guardrail.
        feature_importances: Dict mapping feature name to importance score
            (available for gradient-boosting models).
        baseline_aucs: Per-window AUC scores for the logistic baseline.
        main_aucs: Per-window AUC scores for the main model.
    """

    model: MLModel
    metrics_per_window: list[dict[str, Any]] = field(default_factory=list)
    is_stable: bool = True
    feature_importances: dict[str, float] = field(default_factory=dict)
    baseline_aucs: list[float] = field(default_factory=list)
    main_aucs: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def train_and_validate(
    df: pd.DataFrame,
    asset: str,
    timeframe: str = "1d",
    config: dict[str, Any] | None = None,
) -> MLResult:
    """Build features, train models, and run walk-forward validation.

    Two models are trained in each validation window:

    * **Logistic regression** -- a simple baseline.
    * **Gradient boosting** -- the main model.

    A stability guardrail flags the result as unstable if the main model's
    AUC falls below the baseline AUC minus 0.02 in more than 50 % of
    windows.

    Args:
        df: OHLCV DataFrame for a single asset.
        asset: Asset identifier (used for logging / labelling).
        timeframe: Candle interval string.
        config: Optional configuration dict.  Recognised keys:

            * ``n_windows`` (int): Number of walk-forward windows (default 5).
            * ``target_horizon`` (int): Forward-return horizon (default 5).

    Returns:
        An :class:`MLResult` containing the trained model, per-window
        metrics, stability flag, and feature importances.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ML training. "
            "Install with: pip install market-signal-lab[ml]"
        ) from exc

    config = config or {}
    n_windows: int = config.get("n_windows", 5)
    target_horizon: int = config.get("target_horizon", 5)

    logger.info(
        "Training ML models for %s (%s) -- %d windows, horizon=%d",
        asset, timeframe, n_windows, target_horizon,
    )

    # Feature engineering.
    feat_df = build_features(df, timeframe=timeframe)
    target = build_target(feat_df, horizon=target_horizon)

    # Align features and target: drop rows where target is NaN.
    valid_mask = target.notna()
    feat_df = feat_df.loc[valid_mask].reset_index(drop=True)
    target = target.loc[valid_mask].reset_index(drop=True)

    # Separate meta columns from feature columns.
    meta_cols = {"timestamp", "close"}
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    X = feat_df[feature_cols]
    y = target

    # Walk-forward splits.
    splits = WalkForwardValidator.split(X, n_windows=n_windows)

    baseline_aucs: list[float] = []
    main_aucs: list[float] = []
    metrics_per_window: list[dict[str, Any]] = []
    final_main_model: MLModel | None = None

    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Skip degenerate windows.
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            logger.warning("Window %d skipped: degenerate target distribution.", i)
            continue

        # Baseline: logistic regression.
        baseline = MLModel(model_type="logistic").train(X_train, y_train)
        baseline_proba = baseline.predict_proba(X_test)
        baseline_auc = float(roc_auc_score(y_test, baseline_proba))
        baseline_aucs.append(baseline_auc)

        # Main: gradient boosting.
        main = MLModel(model_type="gradient_boosting").train(X_train, y_train)
        main_proba = main.predict_proba(X_test)
        main_auc = float(roc_auc_score(y_test, main_proba))
        main_aucs.append(main_auc)

        final_main_model = main

        window_metrics = {
            "window": i,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "baseline_auc": baseline_auc,
            "main_auc": main_auc,
        }
        metrics_per_window.append(window_metrics)
        logger.info(
            "Window %d: baseline AUC=%.4f  main AUC=%.4f",
            i, baseline_auc, main_auc,
        )

    if final_main_model is None:
        raise RuntimeError("All validation windows were degenerate; unable to train.")

    # Stability guardrail: main model should not underperform baseline
    # by more than 0.02 in the majority of windows.
    underperform_count = sum(
        1 for b_auc, m_auc in zip(baseline_aucs, main_aucs)
        if m_auc < b_auc - 0.02
    )
    total_windows = len(baseline_aucs)
    is_stable = underperform_count <= total_windows * 0.5

    if not is_stable:
        logger.warning(
            "Model flagged as UNSTABLE: main model underperformed baseline in "
            "%d / %d windows.",
            underperform_count, total_windows,
        )

    # Feature importances (gradient-boosting only).
    feature_importances: dict[str, float] = {}
    if final_main_model.model_type == "gradient_boosting" and final_main_model._model is not None:
        try:
            importances = final_main_model._model.feature_importances_
            feature_importances = dict(zip(feature_cols, importances.tolist()))
        except AttributeError:
            pass

    return MLResult(
        model=final_main_model,
        metrics_per_window=metrics_per_window,
        is_stable=is_stable,
        feature_importances=feature_importances,
        baseline_aucs=baseline_aucs,
        main_aucs=main_aucs,
    )
