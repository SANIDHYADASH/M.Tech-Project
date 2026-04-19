SEQ_LENGTH = 6
BATCH_SIZE = 32
EPOCHS = 20
LSTM_UNITS = 128
DROPOUT = 0.3

SEQUENCE_FEATURES = [
    "annual_income",
    "debt_to_income",
    "loan_amount",
    "interest_rate",
    "installment",
    "term",
    "total_credit_utilization_ratio",
    "open_credit_ratio",
    "cc_carrying_balance_ratio",
    "installment_burden_ratio",
    "delinq_2y",
    "months_since_last_delinq",
    "num_historical_failed_to_pay",
    "months_since_90d_late",
    "current_accounts_delinq",
    "account_never_delinq_percent",
    "recent_inquiry_risk",
    "bankruptcy_or_tax_issue",
    "delinquency_score"
]