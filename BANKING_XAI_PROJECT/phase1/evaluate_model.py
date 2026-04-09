
"""Evaluation helpers for the banking XAI project."""

from typing import Dict
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="evaluation.log")

def detailed_evaluation(y_true, y_pred) -> Dict[str, float]:
    """Log detailed evaluation and return confusion matrix stats."""
    logger.info("Classification report:\n" + classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")
    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }
