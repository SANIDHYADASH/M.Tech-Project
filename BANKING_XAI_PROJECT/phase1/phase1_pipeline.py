"""Main pipeline script for Phase-1 Banking XAI project."""

import numpy as np
from phase1.config import ARTIFACTS_DIR
from phase1.data_ingestion import load_raw_data
from phase1.feature_engineering import add_domain_features
from phase1.preprocessing import preprocess_and_split
from phase1.train_model import train_xgboost
from phase1.evaluate_model import detailed_evaluation
from phase1.explainability_shap import compute_shap_values
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="main_pipeline.log")


def run_phase1():
    logger.info("==== Starting Phase-1 Banking XAI Pipeline ====")

    # 1. Load data
    df = load_raw_data()

    # 2. Domain feature engineering
    df_fe = add_domain_features(df)

    # 3. Preprocess & split
    X_train, X_test, y_train, y_test, preprocessor, num_feats, cat_feats = preprocess_and_split(df_fe)

    # 4. Train model
    model, metrics = train_xgboost(X_train, y_train, X_test, y_test)

    # 5. Evaluate detailed metrics
    # Use probability threshold instead of default predict()
    y_proba = model.predict_proba(X_test)[:, 1]

    threshold = 0.05
    y_pred = (y_proba >= threshold).astype(int)

    logger.info(f"Using evaluation threshold = {threshold}")

    detailed_stats = detailed_evaluation(y_test, y_pred)

    logger.info(f"Final validation metrics: {metrics}")
    logger.info(f"Confusion matrix stats: {detailed_stats}")

    # 6. SHAP explainability
    if hasattr(X_test, "toarray"):
        X_test_for_shap = X_test.toarray()
    else:
        X_test_for_shap = X_test

    feature_names = None
    compute_shap_values(model, X_test_for_shap, feature_names=feature_names)

    logger.info("==== Phase-1 Pipeline completed successfully ====")

    # Return objects for Phase-2 integration
    return {
        "df": df_fe,
        "features": [col for col in df_fe.columns if col != "loan_status"],
        "target": "loan_status",
        "model": model,
        "metrics": metrics,
        "preprocessor": preprocessor,
        "num_features": num_feats,
        "cat_features": cat_feats
}


def main():
    run_phase1()


if __name__ == "__main__":
    main()