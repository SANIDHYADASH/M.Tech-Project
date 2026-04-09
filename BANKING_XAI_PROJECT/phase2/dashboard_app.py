import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

from phase1.feature_engineering import add_domain_features
from phase1.preprocessing import basic_cleaning
from phase2.sequence_builder import build_sequences

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Banking Early Warning Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    xgb_model = joblib.load("artifacts/xgboost_model.joblib")
    preprocessor = joblib.load("artifacts/preprocessor.joblib")

    lstm_model = None
    if os.path.exists("models/lstm_model.keras"):
        lstm_model = load_model("models/lstm_model.keras")

    return xgb_model, preprocessor, lstm_model


xgb_model, preprocessor, lstm_model = load_artifacts()

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("🏦 Banking Early Warning Dashboard")
st.markdown(
    "Upload a banking loan dataset to generate delinquency risk predictions, behavioral analysis, and portfolio-level insights."
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("Dashboard Controls")
risk_threshold = st.sidebar.slider(
    "High Risk Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.70,
    step=0.05
)

uploaded = st.file_uploader("Upload Loan Dataset CSV", type=["csv"])

if uploaded:
    raw_df = pd.read_csv(uploaded)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(raw_df.head())

    # ---------------------------------------------------------
    # Feature Engineering + Cleaning
    # ---------------------------------------------------------
    df = add_domain_features(raw_df.copy())
    df = basic_cleaning(df)

    target_present = "loan_status" in df.columns

    # Preserve original cleaned dataframe for display
    prediction_df = df.copy()

    if "loan_status" in df.columns:
        df_model = df.drop(columns=["loan_status"])
    else:
        df_model = df.copy()

    # ---------------------------------------------------------
    # Phase-1 XGBoost Predictions
    # ---------------------------------------------------------
    transformed_data = preprocessor.transform(df_model)
    xgb_preds = xgb_model.predict_proba(transformed_data)[:, 1]

    prediction_df["XGBoost Risk Score"] = xgb_preds

    # ---------------------------------------------------------
    # Phase-2 LSTM Predictions
    # ---------------------------------------------------------
    if lstm_model is not None:
        feature_cols = [col for col in df.columns if col != "loan_status"]

        X_seq, y_seq = build_sequences(
            df,
            feature_cols=feature_cols,
            target_col="loan_status" if "loan_status" in df.columns else feature_cols[0]
        )

        lstm_preds = lstm_model.predict(X_seq).flatten()

        padded_lstm_preds = np.concatenate([
            np.full(6, np.nan),
            lstm_preds
        ])

        prediction_df["LSTM Risk Score"] = padded_lstm_preds[:len(prediction_df)]

        prediction_df["Final Hybrid Risk Score"] = (
            0.7 * prediction_df["XGBoost Risk Score"] +
            0.3 * prediction_df["LSTM Risk Score"].fillna(prediction_df["XGBoost Risk Score"])
        )
    else:
        prediction_df["Final Hybrid Risk Score"] = prediction_df["XGBoost Risk Score"]

    # ---------------------------------------------------------
    # Risk Categorization
    # ---------------------------------------------------------
    def assign_risk(score):
        if score >= risk_threshold:
            return "High Risk"
        elif score >= 0.4:
            return "Medium Risk"
        return "Low Risk"

    prediction_df["Risk Category"] = prediction_df["Final Hybrid Risk Score"].apply(assign_risk)

    # ---------------------------------------------------------
    # KPI Metrics
    # ---------------------------------------------------------
    st.subheader("Portfolio Risk Summary")

    total_accounts = len(prediction_df)
    high_risk = (prediction_df["Risk Category"] == "High Risk").sum()
    medium_risk = (prediction_df["Risk Category"] == "Medium Risk").sum()
    low_risk = (prediction_df["Risk Category"] == "Low Risk").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Accounts", total_accounts)
    col2.metric("High Risk Accounts", high_risk)
    col3.metric("Medium Risk Accounts", medium_risk)
    col4.metric("Low Risk Accounts", low_risk)

    # ---------------------------------------------------------
    # Charts
    # ---------------------------------------------------------
    st.subheader("Risk Distribution")

    risk_counts = prediction_df["Risk Category"].value_counts().reset_index()
    risk_counts.columns = ["Risk Category", "Count"]

    fig_pie = px.pie(
        risk_counts,
        names="Risk Category",
        values="Count",
        title="Portfolio Risk Composition"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_hist = px.histogram(
        prediction_df,
        x="Final Hybrid Risk Score",
        nbins=30,
        title="Final Risk Score Distribution"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---------------------------------------------------------
    # High Risk Accounts Table
    # ---------------------------------------------------------
    st.subheader("Top High-Risk Accounts")

    high_risk_df = prediction_df.sort_values(
        by="Final Hybrid Risk Score",
        ascending=False
    ).head(20)

    display_cols = [
        col for col in [
            "annual_income",
            "loan_amount",
            "interest_rate",
            "debt_to_income",
            "delinq_2y",
            "public_record_bankrupt",
            "XGBoost Risk Score",
            "LSTM Risk Score",
            "Final Hybrid Risk Score",
            "Risk Category"
        ] if col in high_risk_df.columns
    ]

    st.dataframe(high_risk_df[display_cols])

    # ---------------------------------------------------------
    # Segment Analysis
    # ---------------------------------------------------------
    st.subheader("Risk by Loan Purpose")

    if "loan_purpose" in prediction_df.columns:
        purpose_risk = (
            prediction_df.groupby("loan_purpose")["Final Hybrid Risk Score"]
            .mean()
            .reset_index()
            .sort_values(by="Final Hybrid Risk Score", ascending=False)
        )

        fig_purpose = px.bar(
            purpose_risk,
            x="loan_purpose",
            y="Final Hybrid Risk Score",
            title="Average Risk by Loan Purpose"
        )
        st.plotly_chart(fig_purpose, use_container_width=True)

    # ---------------------------------------------------------
    # Recommendations
    # ---------------------------------------------------------
    st.subheader("Recommended Actions")

    if high_risk > total_accounts * 0.30:
        st.warning(
            "High proportion of risky borrowers detected. Consider stricter credit review, manual underwriting, and early intervention strategies."
        )

    if "debt_to_income" in prediction_df.columns:
        avg_dti = prediction_df["debt_to_income"].mean()
        st.info(f"Average Debt-to-Income Ratio: {avg_dti:.2f}")

        if avg_dti > 30:
            st.warning("Portfolio DTI is high. Consider focusing on lower-risk income segments.")

    if "interest_rate" in prediction_df.columns:
        avg_interest = prediction_df["interest_rate"].mean()
        st.info(f"Average Interest Rate: {avg_interest:.2f}%")

    # ---------------------------------------------------------
    # Download Results
    # ---------------------------------------------------------
    st.subheader("Download Prediction Results")

    csv = prediction_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Full Risk Report CSV",
        data=csv,
        file_name="loan_risk_predictions.csv",
        mime="text/csv"
    )

