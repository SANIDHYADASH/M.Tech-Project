"""Configuration for the Banking XAI Phase-1 project."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "raw"
ARTIFACTS_DIR = BASE_DIR.parent / "models"
LOG_DIR = BASE_DIR.parent / "logs"

# Data config
RAW_DATA_FILE = DATA_DIR / "loan_data.csv"   # your 10k-row CSV here
TARGET_COLUMN = "loan_status"
BAD_LOAN_LABELS = [
    "Charged Off",
    "Default",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "In Grace Period"
]

# Train / validation split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model config
XGBOOST_PARAMS = {
    # "n_estimators": 300,  # more trees for early stopping
    # "max_depth": 4,      # moderate depth
    # "learning_rate": 0.05,  # slightly faster learning
    # "subsample": 0.8,    # more data per tree
    # "colsample_bytree": 0.8,
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 1,      # less L1 regularization
    "reg_lambda": 2,     # less L2 regularization
    "min_child_weight": 4, # allow more splits
    # "eval_metric": "auc",
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
}

# SHAP config
N_SHAP_SAMPLES = 1000
N_LOCAL_EXPLANATIONS = 5
