"""Domain-specific feature engineering for banking risk."""

import numpy as np
import pandas as pd
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="feature_engineering.log")


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add banking/credit risk features tailored to the loan delinquency dataset.

    This function only creates features if the required source columns exist.
    """
    df = df.copy()

    logger.info("Adding domain-specific features")

    # -----------------------------------------
    # Replace placeholder values
    # -----------------------------------------
    df = df.replace(["NA", "", "null", "None"], np.nan)

    # -----------------------------------------
    # 1. EMI-to-income ratio
    # -----------------------------------------
    if {"installment", "annual_income"}.issubset(df.columns):
        logger.info("Creating feature: emi_to_income_ratio")

        monthly_income = pd.to_numeric(df["annual_income"], errors="coerce") / 12.0
        installment = pd.to_numeric(df["installment"], errors="coerce")

        df["emi_to_income_ratio"] = (
            installment / monthly_income.replace(0, np.nan)
        )

    # -----------------------------------------
    # 2. Total credit utilization ratio
    # -----------------------------------------
    if {"total_credit_utilized", "total_credit_limit"}.issubset(df.columns):
        logger.info("Creating feature: total_credit_utilization_ratio")

        utilized = pd.to_numeric(df["total_credit_utilized"], errors="coerce")
        limit = pd.to_numeric(df["total_credit_limit"], errors="coerce")

        df["total_credit_utilization_ratio"] = (
            utilized / limit.replace(0, np.nan)
        )

    # -----------------------------------------
    # 3. Average debit limit per active account
    # -----------------------------------------
    if {"total_debit_limit", "num_active_debit_accounts"}.issubset(df.columns):
        logger.info("Creating feature: avg_debit_limit_per_active_account")

        total_debit_limit = pd.to_numeric(df["total_debit_limit"], errors="coerce")
        active_debit_accounts = pd.to_numeric(df["num_active_debit_accounts"], errors="coerce")

        df["avg_debit_limit_per_active_account"] = (
            total_debit_limit / active_debit_accounts.replace(0, np.nan)
        )

    # -----------------------------------------
    # 4. Credit history length in years
    # -----------------------------------------
    if {"earliest_credit_line", "issue_month"}.issubset(df.columns):
        logger.info("Creating feature: credit_history_length_years")

        earliest_year = pd.to_numeric(df["earliest_credit_line"], errors="coerce")

        issue_year = (
            df["issue_month"]
            .astype(str)
            .str.extract(r"(\d{4})", expand=False)
        )

        issue_year = pd.to_numeric(issue_year, errors="coerce")

        df["credit_history_length_years"] = issue_year - earliest_year

    # -----------------------------------------
    # 5. High debt-to-income flag
    # -----------------------------------------
    if "debt_to_income" in df.columns:
        logger.info("Creating feature: high_dti_flag")

        dti = pd.to_numeric(df["debt_to_income"], errors="coerce")

        df["high_dti_flag"] = (dti > 35).astype(int)

    # -----------------------------------------
    # 6. Open credit ratio
    # -----------------------------------------
    if {"open_credit_lines", "total_credit_lines"}.issubset(df.columns):
        logger.info("Creating feature: open_credit_ratio")

        open_credit = pd.to_numeric(df["open_credit_lines"], errors="coerce")
        total_credit = pd.to_numeric(df["total_credit_lines"], errors="coerce")

        df["open_credit_ratio"] = (
            open_credit / total_credit.replace(0, np.nan)
        )

    # -----------------------------------------
    # 7. Credit card carrying balance ratio
    # -----------------------------------------
    if {"num_cc_carrying_balance", "num_total_cc_accounts"}.issubset(df.columns):
        logger.info("Creating feature: cc_carrying_balance_ratio")

        cc_balance = pd.to_numeric(df["num_cc_carrying_balance"], errors="coerce")
        total_cc = pd.to_numeric(df["num_total_cc_accounts"], errors="coerce")

        df["cc_carrying_balance_ratio"] = (
            cc_balance / total_cc.replace(0, np.nan)
        )

    # -----------------------------------------
    # 8. Installment burden ratio
    # -----------------------------------------
    if {"installment", "annual_income"}.issubset(df.columns):
        logger.info("Creating feature: installment_burden_ratio")

        installment = pd.to_numeric(df["installment"], errors="coerce")
        monthly_income = pd.to_numeric(df["annual_income"], errors="coerce") / 12.0

        df["installment_burden_ratio"] = (
            installment / monthly_income.replace(0, np.nan)
        )

    # -----------------------------------------
    # 9. Delinquency score
    # -----------------------------------------
    delinquency_cols = {
        "delinq_2y",
        "num_historical_failed_to_pay",
        "current_accounts_delinq"
    }

    if delinquency_cols.issubset(df.columns):
        logger.info("Creating feature: delinquency_score")

        delinq_2y = pd.to_numeric(df["delinq_2y"], errors="coerce").fillna(0)
        historical_failed = pd.to_numeric(df["num_historical_failed_to_pay"], errors="coerce").fillna(0)
        current_delinq = pd.to_numeric(df["current_accounts_delinq"], errors="coerce").fillna(0)

        df["delinquency_score"] = (
            delinq_2y + historical_failed + current_delinq
        )

    # -----------------------------------------
    # 10. Bankruptcy or tax issue flag
    # -----------------------------------------
    if {"public_record_bankrupt", "tax_liens"}.issubset(df.columns):
        logger.info("Creating feature: bankruptcy_or_tax_issue")

        bankrupt = pd.to_numeric(df["public_record_bankrupt"], errors="coerce").fillna(0)
        tax_liens = pd.to_numeric(df["tax_liens"], errors="coerce").fillna(0)

        df["bankruptcy_or_tax_issue"] = (
            ((bankrupt > 0) | (tax_liens > 0))
        ).astype(int)

    # -----------------------------------------
    # 11. Recent inquiry risk
    # -----------------------------------------
    if {"inquiries_last_12m", "months_since_last_credit_inquiry"}.issubset(df.columns):
        logger.info("Creating feature: recent_inquiry_risk")

        inquiries = pd.to_numeric(df["inquiries_last_12m"], errors="coerce")
        months_since_inquiry = pd.to_numeric(
            df["months_since_last_credit_inquiry"],
            errors="coerce"
        )

        df["recent_inquiry_risk"] = (
            inquiries / (months_since_inquiry.fillna(0) + 1)
        )

    logger.info(f"Feature engineering completed. Final shape: {df.shape}")

    return df