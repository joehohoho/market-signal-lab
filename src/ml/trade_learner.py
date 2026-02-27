"""Train ML model from actual backtest trade outcomes.

Instead of predicting generic forward returns, this module trains on whether
the strategy's BUY signals led to profitable trades.  This makes the model
strategy-specific and directly useful for filtering bad signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from ml.features import build_features
from ml.models import MLModel
from strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class TradeLearnResult:
    """Result of trade-based ML training."""

    model: MLModel
    total_trades: int
    winning_trades: int
    losing_trades: int
    train_accuracy: float
    val_accuracy: float
    val_auc: float
    threshold: float = 0.50
    selected_features: list[str] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)


def train_from_trades(
    df: pd.DataFrame,
    strategy: Strategy,
    params: dict[str, Any],
    asset: str,
    timeframe: str,
    fee_preset: str = "liquid_stock",
    initial_capital: float = 10_000.0,
) -> TradeLearnResult:
    """Train ML model from backtest trade outcomes.

    Steps:
        1. Run backtest to get trades with P&L.
        2. Compute strategy signals to find BUY signal bars.
        3. Build features for the entire dataset.
        4. Label each executed BUY signal: ``1`` (profitable) or ``0`` (losing).
        5. Train gradient boosting with a temporal train/validation split.

    Args:
        df: OHLCV DataFrame.
        strategy: Strategy instance to evaluate.
        params: Strategy parameters.
        asset: Asset identifier.
        timeframe: Candle timeframe string.
        fee_preset: Fee preset name for the backtest.
        initial_capital: Starting capital.

    Returns:
        A :class:`TradeLearnResult` with the trained model and statistics.

    Raises:
        ValueError: If there are too few trades to train from.
    """
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ML training. "
            "Install with: pip install market-signal-lab[ml]"
        ) from exc

    # 1. Run baseline backtest
    engine = BacktestEngine(initial_capital=initial_capital, fee_preset=fee_preset)
    baseline = engine.run(df, strategy, params, asset=asset, timeframe=timeframe)

    if len(baseline.trades) < 10:
        raise ValueError(
            f"Need at least 10 trades to learn from. Got {len(baseline.trades)}. "
            "Try a longer date range or different strategy."
        )

    # 2. Compute signals to identify BUY bars
    prepared = df.set_index("timestamp") if "timestamp" in df.columns else df.copy()
    if not isinstance(prepared.index, pd.DatetimeIndex):
        prepared.index = pd.to_datetime(prepared.index)
    signals = strategy.compute(prepared, asset, timeframe, params)

    # 3. Build features
    feat_df = build_features(df, timeframe)
    if feat_df.empty:
        raise ValueError("Could not compute features from the data.")

    # Build timestamp -> feature index mapping
    feat_timestamps = pd.to_datetime(feat_df["timestamp"])
    ts_to_feat_idx: dict[Any, int] = {}
    for fidx in range(len(feat_df)):
        ts_to_feat_idx[feat_timestamps.iloc[fidx]] = fidx

    # 4. Map trades to BUY signal bars and label them.
    # Trade's entry_bar is where the order EXECUTED (at open).
    # The BUY signal was at entry_bar - 1 (pending order model).
    trade_outcomes: dict[int, float] = {}
    for trade in baseline.trades:
        signal_bar = trade["entry_bar"] - 1
        if signal_bar >= 0:
            trade_outcomes[signal_bar] = 1.0 if trade["pnl"] > 0 else 0.0

    # 5. Build training dataset
    meta_cols = {"timestamp", "close"}
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]

    df_timestamps = pd.to_datetime(df["timestamp"])

    X_rows: list[pd.Series] = []
    y_labels: list[float] = []

    for signal_bar, outcome in sorted(trade_outcomes.items()):
        if signal_bar >= len(df):
            continue
        bar_ts = df_timestamps.iloc[signal_bar]
        feat_idx = ts_to_feat_idx.get(bar_ts)
        if feat_idx is not None:
            X_rows.append(feat_df[feature_cols].iloc[feat_idx])
            y_labels.append(outcome)

    if len(X_rows) < 10:
        raise ValueError(
            f"Only {len(X_rows)} BUY signals could be matched to features. "
            "Need at least 10."
        )

    X = pd.DataFrame(X_rows).reset_index(drop=True)
    y = pd.Series(y_labels, name="target")

    n_winning = int(y.sum())
    n_losing = len(y) - n_winning

    if n_winning < 2 or n_losing < 2:
        raise ValueError(
            f"Need at least 2 winning and 2 losing trades. "
            f"Got {n_winning} wins, {n_losing} losses."
        )

    # 6. Feature selection — reduce to top features to avoid overfitting
    #    on small datasets.  Use mutual information to rank features.
    from sklearn.feature_selection import mutual_info_classif

    max_features = min(8, len(feature_cols))
    mi_scores = mutual_info_classif(X.fillna(0), y, random_state=42)
    top_indices = np.argsort(mi_scores)[::-1][:max_features]
    selected_cols = [feature_cols[i] for i in top_indices]
    X = X[selected_cols]
    feature_cols = selected_cols

    logger.info(
        "Feature selection: kept %d of %d features: %s",
        len(selected_cols), len(mi_scores),
        ", ".join(selected_cols[:5]),
    )

    # 7. Model selection — logistic regression for small datasets,
    #    gradient boosting only when there's enough data.
    model_type = "logistic" if len(X) < 80 else "gradient_boosting"

    # 8. Train with temporal split (70/30)
    split_point = int(len(X) * 0.7)
    if split_point < 4:
        split_point = max(4, len(X) // 2)

    X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]

    # Ensure both classes in train and val sets
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        split_point = int(len(X) * 0.6)
        X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]

    if y_train.nunique() < 2 or y_val.nunique() < 2:
        # Last resort: use all data for training, no validation
        X_train, X_val = X, X
        y_train, y_val = y, y

    model = MLModel(model_type=model_type).train(X_train, y_train)

    # Evaluate
    train_proba = model.predict_proba(X_train)
    train_preds = (train_proba >= 0.5).astype(int)
    train_acc = float(accuracy_score(y_train, train_preds))

    val_proba = model.predict_proba(X_val)
    val_preds = (val_proba >= 0.5).astype(int)
    val_acc = float(accuracy_score(y_val, val_preds))

    try:
        val_auc = float(roc_auc_score(y_val, val_proba))
    except ValueError:
        val_auc = 0.5

    # Find optimal threshold that maximises precision on training data
    # (we want to block losers, not block winners)
    best_threshold = 0.50
    best_score = 0.0
    for t in np.arange(0.40, 0.65, 0.01):
        preds_t = (train_proba >= t).astype(int)
        # Score = trades kept that were winners / total trades kept
        kept = preds_t.sum()
        if kept < 3:  # need at least a few trades
            continue
        wins_kept = ((preds_t == 1) & (y_train == 1)).sum()
        precision = wins_kept / kept
        # Penalise if we block too many trades (keep at least 40%)
        keep_ratio = kept / len(preds_t)
        score = precision * min(1.0, keep_ratio / 0.4)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    logger.info("Optimal ML threshold: %.2f (precision score: %.3f)", best_threshold, best_score)

    # Feature importances
    feature_importances: dict[str, float] = {}
    if model._model is not None:
        try:
            importances = model._model.feature_importances_
            feature_importances = dict(zip(feature_cols, importances.tolist()))
        except AttributeError:
            # Logistic regression: use coefficient magnitudes
            try:
                clf = model._model.named_steps["clf"]
                importances = np.abs(clf.coef_[0])
                feature_importances = dict(zip(feature_cols, importances.tolist()))
            except (AttributeError, KeyError):
                pass

    logger.info(
        "Trade learner: %d trades (%d win, %d loss) | "
        "train_acc=%.3f val_acc=%.3f val_auc=%.3f",
        len(y), n_winning, n_losing, train_acc, val_acc, val_auc,
    )

    return TradeLearnResult(
        model=model,
        total_trades=len(y),
        winning_trades=n_winning,
        losing_trades=n_losing,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        val_auc=val_auc,
        threshold=best_threshold,
        selected_features=selected_cols,
        feature_importances=feature_importances,
    )
