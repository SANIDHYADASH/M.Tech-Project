import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from phase1.feature_engineering import add_domain_features
from phase1.preprocessing import basic_cleaning
import joblib

model = joblib.load('artifacts/xgboost_model.joblib')

st.title('Banking Early Warning Dashboard')

uploaded = st.file_uploader('Upload Loan Dataset')

if uploaded:
    df = pd.read_csv(uploaded)

    # Load preprocessor
    preprocessor = joblib.load("artifacts/preprocessor.joblib")

    # Apply same feature engineering used during training
    df = add_domain_features(df)
    df = basic_cleaning(df)

    # Remove target column if present
    if "loan_status" in df.columns:
        df = df.drop(columns=["loan_status"])

    # Transform using saved preprocessor
    df_transformed = preprocessor.transform(df)

    # Predict
    preds = model.predict_proba(df_transformed)[:, 1]
    df['Risk Score'] = preds

    st.dataframe(df)
    st.bar_chart(df['Risk Score'])