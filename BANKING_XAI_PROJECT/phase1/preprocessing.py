"""Preprocessing pipeline: cleaning, target handling, and train/test split."""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from phase1.config import BAD_LOAN_LABELS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="preprocessing.log")


def encode_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert textual loan_status to binary delinquency label.

    0 = non-delinquent / good
    1 = delinquent / bad
    """
    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in data.")

    # Skip re-encoding if already numeric
    if pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        logger.info("loan_status already encoded. Skipping re-encoding.")
        logger.info(df[TARGET_COLUMN].value_counts(dropna=False).to_string())
        return df

    status = df[TARGET_COLUMN].astype(str).str.strip()

    bad_statuses = {
        "Charged Off",
        "Default",
        "Late (16-30 days)",
        "Late (31-120 days)",
        "In Grace Period"
    }

    logger.info("Loan status value counts BEFORE encoding:")
    logger.info(status.value_counts(dropna=False).to_string())

    df[TARGET_COLUMN] = status.apply(
        lambda x: 1 if str(x).strip() in bad_statuses else 0
    )

    logger.info("Loan status value counts AFTER encoding (0=good, 1=bad):")
    logger.info(df[TARGET_COLUMN].value_counts(dropna=False).to_string())

    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning operations.

    - Encode target column
    - Remove leakage columns
    - Remove high-cardinality columns
    - Convert data types
    - Drop rows with missing target
    """
    logger.info("Starting basic cleaning")

    df = df.copy()

    # -----------------------------------------
    # Replace string placeholders like 'NA'
    # -----------------------------------------
    df = df.replace(["NA", "", "null", "None"], np.nan)

    # -----------------------------------------
    # Encode target safely
    # -----------------------------------------
    if TARGET_COLUMN in df.columns:
        df = encode_loan_status(df)

    # -----------------------------------------
    # Drop only true leakage columns
    # -----------------------------------------
    leakage_cols = [
        "balance",
        "paid_total",
        "paid_principal",
        "paid_interest",
        "paid_late_fees"
    ]

    cols_to_drop = [col for col in leakage_cols if col in df.columns]

    if cols_to_drop:
        logger.info(f"Dropping leakage columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # -----------------------------------------
    # Drop high-cardinality columns
    # -----------------------------------------
    high_cardinality_cols = [
        "emp_title"
    ]

    high_cardinality_to_drop = [
        col for col in high_cardinality_cols if col in df.columns
    ]

    if high_cardinality_to_drop:
        logger.info(f"Dropping high-cardinality columns: {high_cardinality_to_drop}")
        df = df.drop(columns=high_cardinality_to_drop)

    # -----------------------------------------
    # Convert known numeric columns
    # -----------------------------------------
    numeric_conversion_cols = [
        "emp_length",
        "annual_income",
        "debt_to_income",
        "annual_income_joint",
        "debt_to_income_joint",
        "delinq_2y",
        "months_since_last_delinq",
        "earliest_credit_line",
        "inquiries_last_12m",
        "total_credit_lines",
        "open_credit_lines",
        "total_credit_limit",
        "total_credit_utilized",
        "num_collections_last_12m",
        "num_historical_failed_to_pay",
        "months_since_90d_late",
        "current_accounts_delinq",
        "total_collection_amount_ever",
        "current_installment_accounts",
        "accounts_opened_24m",
        "months_since_last_credit_inquiry",
        "num_satisfactory_accounts",
        "num_accounts_120d_past_due",
        "num_accounts_30d_past_due",
        "num_active_debit_accounts",
        "total_debit_limit",
        "num_total_cc_accounts",
        "num_open_cc_accounts",
        "num_cc_carrying_balance",
        "num_mort_accounts",
        "account_never_delinq_percent",
        "tax_liens",
        "public_record_bankrupt",
        "loan_amount",
        "term",
        "interest_rate",
        "installment",
        "emi_to_income_ratio",
        "total_credit_utilization_ratio",
        "avg_debit_limit_per_active_account",
        "credit_history_length_years",
        "open_credit_ratio",
        "cc_carrying_balance_ratio",
        "installment_burden_ratio",
        "delinquency_score",
        "bankruptcy_or_tax_issue",
        "recent_inquiry_risk",
        "issue_year",
        "issue_month_num"
    ]

    for col in numeric_conversion_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------------------
    # Convert issue_month into useful features
    # -----------------------------------------
    if "issue_month" in df.columns:
        logger.info("Converting issue_month into date features")

        if not pd.api.types.is_datetime64_any_dtype(df["issue_month"]):
            df["issue_month"] = pd.to_datetime(
                df["issue_month"],
                format="%b-%Y",
                errors="coerce"
            )

        df["issue_year"] = df["issue_month"].dt.year
        df["issue_month_num"] = df["issue_month"].dt.month

    # -----------------------------------------
    # Drop rows with missing target
    # -----------------------------------------
    if TARGET_COLUMN in df.columns:
        before = df.shape[0]

        df = df.dropna(subset=[TARGET_COLUMN])

        after = df.shape[0]

        logger.info(f"Dropped {before - after} rows with missing target.")

    logger.info(f"Final cleaned dataframe shape: {df.shape}")

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer for numeric and categorical features.
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    logger.info(
        f"Numeric features: {len(numeric_features)}, "
        f"Categorical features: {len(categorical_features)}"
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features


def preprocess_and_split(df: pd.DataFrame):
    """
    Full preprocessing: clean, split, and build ColumnTransformer.
    """
    df = df.replace({pd.NA: np.nan})

    df_clean = basic_cleaning(df)

    X, y = split_features_target(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    preprocessor, num_feats, cat_feats = build_preprocessor(X_train)

    logger.info("Fitting preprocessor on training data")

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    logger.info(
        f"Transformed Train shape: {X_train_trans.shape}, "
        f"Test shape: {X_test_trans.shape}"
    )

    return (
        X_train_trans,
        X_test_trans,
        y_train,
        y_test,
        preprocessor,
        num_feats,
        cat_feats
    )