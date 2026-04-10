from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    f1_score,
    confusion_matrix
)
import numpy as np
import os

from tensorflow.keras.callbacks import EarlyStopping

from phase2.lstm_model import build_lstm_model
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="train_lstm.log")


def train_lstm(X, y):
    """
    Train improved LSTM model for loan delinquency prediction.
    """

    logger.info("Splitting sequence data into train and validation sets")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")

    # -----------------------------------------
    # Stronger class weights for imbalance
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
            1: 4.0
        }
    else:
        imbalance_ratio = num_negative / num_positive

        class_weights = {
            0: 1.0,
            1: min(float(imbalance_ratio), 4.0)
        }

    logger.info(f"Using class weights: {class_weights}")

    # -----------------------------------------
    # Build model
    # -----------------------------------------
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    # -----------------------------------------
    # Early stopping
    # -----------------------------------------
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    logger.info("Starting LSTM training")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    logger.info("Generating validation predictions")

    y_proba = model.predict(X_val).flatten()

    # -----------------------------------------
    # Threshold testing
    # -----------------------------------------
    # for t in [0.378, 0.380, 0.382, 0.384, 0.386]:
    for t in [0.386, 0.388, 0.390, 0.392, 0.394]:
        temp_pred = (y_proba >= t).astype(int)

        logger.info(
            f"Threshold={t} | "
            f"Predicted Positives={np.sum(temp_pred == 1)}"
        )

    # threshold = 0.38
    threshold = 0.389
    y_pred = (y_proba >= threshold).astype(int)

    logger.info(f"Using prediction threshold = {threshold}")

    logger.info(
        f"Prediction probability stats -> "
        f"Min: {y_proba.min():.4f}, "
        f"Max: {y_proba.max():.4f}, "
        f"Mean: {y_proba.mean():.4f}"
    )

    # -----------------------------------------
    # Metrics
    # -----------------------------------------
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