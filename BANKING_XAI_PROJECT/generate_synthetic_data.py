import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

NUM_RECORDS = 5000
OUTPUT_PATH = "data/raw/loan_data.csv"

states = ["CA", "TX", "NY", "FL", "NJ", "PA", "OH", "IL", "AZ", "NC"]
homeownership_types = ["RENT", "MORTGAGE", "OWN"]
verified_income_types = ["Verified", "Not Verified", "Source Verified"]
loan_purposes = [
    "debt_consolidation", "credit_card", "home_improvement",
    "medical", "vacation", "moving", "small_business", "other"
]
application_types = ["individual", "joint"]
grades = ["A", "B", "C", "D", "E"]
sub_grades = ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2", "E1", "E2"]
listing_status = ["whole", "fractional"]
disbursement_methods = ["Cash", "DirectPay"]
loan_statuses = ["Current", "Fully Paid", "Late (16-30 days)", "Late (31-120 days)", "Charged Off"]

rows = []

for i in range(NUM_RECORDS):

    annual_income = np.random.randint(25000, 180000)
    loan_amount = np.random.randint(2000, 50000)
    interest_rate = round(np.random.uniform(5.0, 24.0), 2)
    debt_to_income = round(np.random.uniform(5, 45), 2)
    total_credit_limit = np.random.randint(10000, 300000)
    total_credit_utilized = np.random.randint(1000, total_credit_limit)
    total_debit_limit = np.random.randint(2000, 50000)
    num_active_debit_accounts = np.random.randint(1, 10)
    installment = round(loan_amount / np.random.randint(12, 60), 2)
    delinq_2y = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
    public_record_bankrupt = np.random.choice([0, 1], p=[0.95, 0.05])
    tax_liens = np.random.choice([0, 1], p=[0.97, 0.03])
    inquiries_last_12m = np.random.randint(0, 10)
    months_since_last_delinq = np.random.choice([np.nan, np.random.randint(1, 60)], p=[0.6, 0.4])
    months_since_90d_late = np.random.choice([np.nan, np.random.randint(1, 90)], p=[0.85, 0.15])
    account_never_delinq_percent = round(np.random.uniform(60, 100), 1)

    risk_score = 0

    if debt_to_income > 35:
        risk_score += 2
    if interest_rate > 18:
        risk_score += 2
    if delinq_2y > 0:
        risk_score += 2
    if public_record_bankrupt == 1:
        risk_score += 3
    if tax_liens == 1:
        risk_score += 2
    if inquiries_last_12m > 5:
        risk_score += 1
    if account_never_delinq_percent < 80:
        risk_score += 2
    if total_credit_utilized / total_credit_limit > 0.8:
        risk_score += 2
    if annual_income < 40000:
        risk_score += 1

    if risk_score >= 8:
        loan_status = np.random.choice(
            ["Late (31-120 days)", "Charged Off"],
            p=[0.7, 0.3]
        )
    elif risk_score >= 5:
        loan_status = np.random.choice(
            ["Late (16-30 days)", "Current"],
            p=[0.4, 0.6]
        )
    else:
        loan_status = np.random.choice(
            ["Current", "Fully Paid"],
            p=[0.85, 0.15]
        )

    row = {
        "emp_title": fake.job(),
        "emp_length": np.random.randint(0, 11),
        "state": random.choice(states),
        "homeownership": random.choice(homeownership_types),
        "annual_income": annual_income,
        "verified_income": random.choice(verified_income_types),
        "debt_to_income": debt_to_income,
        "annual_income_joint": np.random.choice([np.nan, np.random.randint(50000, 250000)]),
        "verification_income_joint": random.choice(verified_income_types),
        "debt_to_income_joint": np.random.choice([np.nan, round(np.random.uniform(5, 40), 2)]),
        "delinq_2y": delinq_2y,
        "months_since_last_delinq": months_since_last_delinq,
        "earliest_credit_line": np.random.randint(1985, 2018),
        "inquiries_last_12m": inquiries_last_12m,
        "total_credit_lines": np.random.randint(5, 50),
        "open_credit_lines": np.random.randint(1, 25),
        "total_credit_limit": total_credit_limit,
        "total_credit_utilized": total_credit_utilized,
        "num_collections_last_12m": np.random.randint(0, 3),
        "num_historical_failed_to_pay": np.random.randint(0, 5),
        "months_since_90d_late": months_since_90d_late,
        "current_accounts_delinq": np.random.randint(0, 3),
        "total_collection_amount_ever": np.random.randint(0, 5000),
        "current_installment_accounts": np.random.randint(0, 8),
        "accounts_opened_24m": np.random.randint(0, 15),
        "months_since_last_credit_inquiry": np.random.randint(0, 24),
        "num_satisfactory_accounts": np.random.randint(1, 20),
        "num_accounts_120d_past_due": np.random.randint(0, 2),
        "num_accounts_30d_past_due": np.random.randint(0, 3),
        "num_active_debit_accounts": num_active_debit_accounts,
        "total_debit_limit": total_debit_limit,
        "num_total_cc_accounts": np.random.randint(1, 25),
        "num_open_cc_accounts": np.random.randint(1, 15),
        "num_cc_carrying_balance": np.random.randint(0, 10),
        "num_mort_accounts": np.random.randint(0, 5),
        "account_never_delinq_percent": account_never_delinq_percent,
        "tax_liens": tax_liens,
        "public_record_bankrupt": public_record_bankrupt,
        "loan_purpose": random.choice(loan_purposes),
        "application_type": random.choice(application_types),
        "loan_amount": loan_amount,
        "term": random.choice([36, 60]),
        "interest_rate": interest_rate,
        "installment": installment,
        "grade": random.choice(grades),
        "sub_grade": random.choice(sub_grades),
        "issue_month": fake.date_between(start_date="-3y", end_date="today").strftime("%b-%Y"),
        "loan_status": loan_status,
        "initial_listing_status": random.choice(listing_status),
        "disbursement_method": random.choice(disbursement_methods),
        "balance": round(np.random.uniform(0, loan_amount), 2),
        "paid_total": round(np.random.uniform(0, loan_amount), 2),
        "paid_principal": round(np.random.uniform(0, loan_amount * 0.8), 2),
        "paid_interest": round(np.random.uniform(0, loan_amount * 0.2), 2),
        "paid_late_fees": round(np.random.uniform(0, 500), 2)
    }

    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Create directory if not exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Save CSV
df.to_csv(OUTPUT_PATH, index=False)

print(f"Synthetic dataset created successfully with shape: {df.shape}")
print(f"Saved to: {OUTPUT_PATH}")
print(df.head())