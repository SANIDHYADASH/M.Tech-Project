from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    f1_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

from phase2.lstm_model import build_lstm_model
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="train_lstm.log")


def train_lstm(X, y):
    """
    Train LSTM model for loan delinquency prediction.
    """

    logger.info("Splitting sequence data into train and validation sets")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Ensure labels are integer type
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")

    # -----------------------------------------
    # Compute class weights for imbalance
    # -----------------------------------------
    logger.info("Computing class weights")

    num_negative = int(np.sum(y_train == 0))
    num_positive = int(np.sum(y_train == 1))

    logger.info(f"Negative samples: {num_negative}")
    logger.info(f"Positive samples: {num_positive}")

    if num_positive == 0:
        logger.warning("No positive samples found in y_train. Using default class weights.")
        class_weights = {
            0: 1.0,
            1: 1.0
        }
    else:
        class_weights = {
            0: 1.0,
            1: float(num_negative / num_positive)
        }
    # logger.info("Computing class weights")

    # classes = np.unique(y_train)

    # weights = compute_class_weight(
    #     class_weight="balanced",
    #     classes=classes,
    #     y=y_train
    # )

    # class_weights = {
    #     0: weights[0],
    #     1: weights[1]
    # }
    # class_weights = {
    #     0: float(weights[0]),
    #     1: float(weights[1])
    # }

    logger.info(f"Using class weights: {class_weights}")

    # -----------------------------------------
    # Build model
    # -----------------------------------------
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    logger.info("Starting LSTM training")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        class_weight=class_weights,
        verbose=1
    )

    logger.info("Generating validation predictions")

    y_proba = model.predict(X_val).flatten()

    # Inspect probability distribution across thresholds
    for t in [0.50, 0.51, 0.52, 0.53, 0.54]:
        temp_pred = (y_proba >= t).astype(int)

        logger.info(
            f"Threshold={t} | "
            f"Predicted Positives={np.sum(temp_pred == 1)}"
        )

    # Lower threshold for minority class detection
    # threshold = 0.20
    threshold = 0.52
    y_pred = (y_proba >= threshold).astype(int)

    logger.info(f"Using prediction threshold = {threshold}")
    logger.info(
        f"Prediction probability stats -> "
        f"Min: {y_proba.min():.4f}, "
        f"Max: {y_proba.max():.4f}, "
        f"Mean: {y_proba.mean():.4f}"
    )

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_val, y_proba)),
        "f1_score": float(f1_score(y_val, y_pred, zero_division=0))
    }

    cm = confusion_matrix(y_val, y_pred)

    logger.info(f"LSTM Validation Metrics: {metrics}")
    logger.info(f"Confusion Matrix:\n{cm}")

    logger.info(f"Predicted positives: {np.sum(y_pred == 1)}")
    logger.info(f"Actual positives: {np.sum(y_val == 1)}")

    # -----------------------------------------
    # Save model
    # -----------------------------------------
    os.makedirs("models", exist_ok=True)

    model_path = "models/lstm_model.keras"
    model.save(model_path)

    logger.info(f"Saved LSTM model to {model_path}")

    return model, history