"""Model training module for Phase-1 XGBoost model."""

from typing import Tuple, Dict
import joblib
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from phase1.config import ARTIFACTS_DIR, XGBOOST_PARAMS
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="train_model.log")


def train_xgboost(X_train, y_train, X_val, y_val) -> Tuple[XGBClassifier, Dict[str, float]]:
    """
    Train an XGBoost classifier with class imbalance handling
    and return model + evaluation metrics.
    """

    # -----------------------------
    # 1. Check class distribution
    # -----------------------------
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)

    logger.info(f"Training class distribution -> Negative: {n_neg}, Positive: {n_pos}")

    # -----------------------------
    # 2. Handle imbalance using SMOTE
    # -----------------------------
    try:
        logger.info("Applying SMOTE to balance training data")

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        logger.info(
            f"After SMOTE -> Negative: {np.sum(y_train_resampled == 0)}, "
            f"Positive: {np.sum(y_train_resampled == 1)}"
        )

    except Exception as e:
        logger.warning(f"SMOTE failed: {str(e)}. Proceeding with original training data.")
        X_train_resampled = X_train
        y_train_resampled = y_train

    # -----------------------------
    # 3. Compute scale_pos_weight
    # -----------------------------
    n_pos_resampled = np.sum(y_train_resampled == 1)
    n_neg_resampled = np.sum(y_train_resampled == 0)

    if n_pos_resampled > 0 and n_neg_resampled > 0:
        scale_pos_weight = n_neg_resampled / n_pos_resampled
    else:
        scale_pos_weight = 1.0

    logger.info(f"Using scale_pos_weight = {scale_pos_weight:.2f}")

    # Copy config so original dict isn't modified
    xgb_params = dict(XGBOOST_PARAMS)
    xgb_params["scale_pos_weight"] = scale_pos_weight

    # -----------------------------
    # 4. Initialize model
    # -----------------------------
    logger.info("Initializing XGBoost model")
    model = XGBClassifier(**xgb_params)

    # -----------------------------
    # 5. Train model
    # -----------------------------
    logger.info("Training XGBoost model")

    model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # -----------------------------
    # 6. Predict on validation set
    # -----------------------------
    y_proba = model.predict_proba(X_val)[:, 1]

    # Lower threshold from default 0.5 to improve recall
    threshold = 0.10
    y_pred = (y_proba >= threshold).astype(int)

    # -----------------------------
    # 7. Evaluate metrics
    # -----------------------------
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        recall_score,
        precision_score,
        f1_score,
        confusion_matrix
    )

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "auc": float(roc_auc_score(y_val, y_proba)),
        "recall": float(recall_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_val, y_pred, zero_division=0))
    }

    cm = confusion_matrix(y_val, y_pred)

    logger.info(f"Validation metrics: {metrics}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # -----------------------------
    # 8. Save model
    # -----------------------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "xgboost_model.joblib"
    joblib.dump(model, model_path)

    logger.info(f"Saved model to {model_path}")

    return model, metrics