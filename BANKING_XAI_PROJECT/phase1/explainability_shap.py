
"""Explainability layer using SHAP for the XGBoost model."""

from typing import Any
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from phase1.config import ARTIFACTS_DIR, N_SHAP_SAMPLES, N_LOCAL_EXPLANATIONS
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="explainability_shap.log")

def compute_shap_values(model: Any, X_test, feature_names=None):
    """Compute SHAP values for a sample of X_test and save plots."""
    logger.info("Initializing SHAP TreeExplainer")
    explainer = shap.TreeExplainer(model)

    logger.info("Sampling test data for SHAP")
    if N_SHAP_SAMPLES and X_test.shape[0] > N_SHAP_SAMPLES:
        idx = np.random.choice(X_test.shape[0], N_SHAP_SAMPLES, replace=False)
        X_sample = X_test[idx]
    else:
        X_sample = X_test

    logger.info(f"Computing SHAP values for {X_sample.shape[0]} samples")
    shap_values = explainer.shap_values(X_sample)

    # Global summary plot
    summary_path = ARTIFACTS_DIR / "shap_summary.png"
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary plot to {summary_path}")

    # Local explanations (first few samples)
    for i in range(min(N_LOCAL_EXPLANATIONS, X_sample.shape[0])):
        force_path = ARTIFACTS_DIR / f"shap_force_{i}.png"
        shap.plots.force(
            explainer.expected_value,
            shap_values[i],
            matplotlib=True,
            show=False
        )
        plt.savefig(force_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP force plot for sample {i} to {force_path}")

    # Save explainer & shap values (optional, lightweight)
    explainer_path = ARTIFACTS_DIR / "shap_explainer.joblib"
    joblib.dump(explainer, explainer_path)
    logger.info(f"Saved SHAP explainer to {explainer_path}")
