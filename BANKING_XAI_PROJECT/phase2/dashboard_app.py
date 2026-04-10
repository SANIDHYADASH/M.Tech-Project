import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shap
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
    xgb_model = joblib.load("models/xgboost_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")

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
    # Risk Distribution
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
    st.plotly_chart(fig_pie, use_container_width=True, key="risk_distribution_pie")

    fig_hist = px.histogram(
        prediction_df,
        x="Final Hybrid Risk Score",
        nbins=30,
        color="Risk Category",
        title="Final Risk Score Distribution"
    )
    st.plotly_chart(fig_hist, use_container_width=True, key="risk_distribution_hist")

    # ---------------------------------------------------------
    # Customer Segmentation Module
    # ---------------------------------------------------------
    st.subheader("Customer Segmentation Analysis")

    cluster_features = [
        col for col in [
            "annual_income",
            "loan_amount",
            "debt_to_income",
            "total_credit_utilization_ratio",
            "delinq_2y"
        ] if col in prediction_df.columns
    ]

    if len(cluster_features) >= 3:
        cluster_df = prediction_df[cluster_features].fillna(0)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)

        kmeans = KMeans(n_clusters=4, random_state=42)
        prediction_df["Customer Segment"] = kmeans.fit_predict(scaled_data)

        fig_cluster = px.scatter(
            prediction_df,
            x="annual_income",
            y="loan_amount",
            color=prediction_df["Customer Segment"].astype(str),
            size=np.clip(prediction_df["Final Hybrid Risk Score"], 0.01, None),
            title="Customer Segmentation"
        )
        st.plotly_chart(fig_cluster, use_container_width=True, key="customer_segmentation_chart")

    # ---------------------------------------------------------
    # Geographic Risk Analysis
    # ---------------------------------------------------------
    if "state" in prediction_df.columns:
        st.subheader("Geographic Risk Analysis")

        state_risk = (
            prediction_df.groupby("state")["Final Hybrid Risk Score"]
            .mean()
            .reset_index()
            .sort_values(by="Final Hybrid Risk Score", ascending=False)
        )

        fig_state = px.bar(
            state_risk,
            x="state",
            y="Final Hybrid Risk Score",
            title="Average Risk Score by State"
        )
        st.plotly_chart(fig_state, use_container_width=True, key="state_risk_chart")

    # ---------------------------------------------------------
    # Grade-wise Risk Analysis
    # ---------------------------------------------------------
    if "grade" in prediction_df.columns:
        st.subheader("Loan Grade Risk Analysis")

        grade_risk = (
            prediction_df.groupby("grade")["Final Hybrid Risk Score"]
            .mean()
            .reset_index()
        )

        fig_grade = px.bar(
            grade_risk,
            x="grade",
            y="Final Hybrid Risk Score",
            color="Final Hybrid Risk Score",
            title="Average Risk Score by Loan Grade"
        )
        st.plotly_chart(fig_grade, use_container_width=True, key="grade_risk_chart")
    
    # ---------------------------------------------------------
    # Income vs Risk Analysis
    # ---------------------------------------------------------
    st.subheader("Income vs Risk Analysis")

    fig_income = px.scatter(
        prediction_df,
        x="annual_income",
        y="Final Hybrid Risk Score",
        color="Risk Category",
        title="Income vs Final Risk Score"
    )

    st.plotly_chart(fig_income, use_container_width=True, key="income_risk_chart")

    # ---------------------------------------------------------
    # Monthly Trend Analysis
    # ---------------------------------------------------------

    if "issue_year" in prediction_df.columns and "issue_month_num" in prediction_df.columns:
        st.subheader("Monthly Risk Trend")

        monthly_risk = (
            prediction_df.groupby(["issue_year", "issue_month_num"])["Final Hybrid Risk Score"]
            .mean()
            .reset_index()
        )

        monthly_risk["period"] = (
            monthly_risk["issue_year"].astype(str) + "-" +
            monthly_risk["issue_month_num"].astype(str)
        )

        fig_month = px.line(
            monthly_risk,
            x="period",
            y="Final Hybrid Risk Score",
            title="Monthly Average Risk Trend"
        )

        st.plotly_chart(fig_month, use_container_width=True, key="monthly_risk_chart")

    # ---------------------------------------------------------
    # What-If Scenario Simulator
    # ---------------------------------------------------------
    st.subheader("What-If Risk Simulation")

    sim_income = st.slider("Annual Income", 20000, 200000, 50000)
    sim_loan = st.slider("Loan Amount", 1000, 50000, 10000)
    sim_dti = st.slider("Debt To Income", 1, 60, 20)
    sim_interest = st.slider("Interest Rate", 5, 30, 12)

    simulated_score = (
        (sim_loan / 50000) * 0.3 +
        (sim_dti / 60) * 0.3 +
        (sim_interest / 30) * 0.2 +
        (1 - sim_income / 200000) * 0.2
    )

    st.metric("Simulated Risk Score", round(simulated_score, 3))

    # ---------------------------------------------------------
    # Early Warning Triggers
    # ---------------------------------------------------------
    st.subheader("Early Warning Trigger Rules")

    warning_df = prediction_df[
        (prediction_df["Final Hybrid Risk Score"] > 0.7) |
        (prediction_df["debt_to_income"] > 35) |
        (prediction_df["delinq_2y"] > 1)
    ]

    st.write(f"Accounts Requiring Manual Review: {len(warning_df)}")
    # st.dataframe(warning_df.head(20))
    st.dataframe(warning_df)

    # ---------------------------------------------------------
    # Portfolio Health Score
    # ---------------------------------------------------------
    st.subheader("Portfolio Health Index")

    high_risk_percentage = (high_risk / total_accounts) * 100
    avg_dti = prediction_df["debt_to_income"].mean()
    avg_util = prediction_df["total_credit_utilization_ratio"].mean() * 100

    delinquency_rate = (
        len(prediction_df[prediction_df["Risk Category"] == "High Risk"]) /
        total_accounts
    ) * 100

    portfolio_health = 100 - (
        high_risk_percentage * 0.5 +
        avg_dti * 0.2 +
        avg_util * 0.2 +
        delinquency_rate * 0.1
    )

    portfolio_health = max(0, min(100, portfolio_health))

    st.metric("Portfolio Health Score", round(portfolio_health, 2))

    if portfolio_health > 75:
        st.success("Healthy Portfolio")
    elif portfolio_health > 50:
        st.warning("Moderate Risk Portfolio")
    else:
        st.error("High Stress Portfolio")

    # ---------------------------------------------------------
    # SHAP Explainability in Dashboard
    # ---------------------------------------------------------
    st.subheader("Model Explainability")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(transformed_data[:100])

    st.write("Top Features Driving Delinquency Predictions")

    fig, ax = plt.subplots(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        transformed_data[:100],
        show=False
    )

    st.pyplot(fig)
    plt.close(fig)

    # ---------------------------------------------------------
    # Fairness Analysis
    # ---------------------------------------------------------
    if "homeownership" in prediction_df.columns:
        st.subheader("Fairness Analysis")

        fairness_df = (
            prediction_df.groupby("homeownership")["Final Hybrid Risk Score"]
            .mean()
            .reset_index()
        )

        fig_fairness = px.bar(
            fairness_df,
            x="homeownership",
            y="Final Hybrid Risk Score",
            title="Average Risk Score by Homeownership"
        )

        st.plotly_chart(fig_fairness, use_container_width=True, key="fairness_chart")

    # ---------------------------------------------------------
    # Charts
    # ---------------------------------------------------------
    st.subheader("Risk Distribution")

    risk_counts = prediction_df["Risk Category"].value_counts().reset_index()
    risk_counts.columns = ["Risk Category", "Count"]

    # fig_pie = px.pie(
    #     risk_counts,
    #     names="Risk Category",
    #     values="Count",
    #     title="Portfolio Risk Composition"
    # )
    # st.plotly_chart(fig_pie, use_container_width=True, key="risk_distribution_pie")

    # fig_hist = px.histogram(
    #     prediction_df,
    #     x="Final Hybrid Risk Score",
    #     nbins=30,
    #     title="Final Risk Score Distribution"
    # )
    # st.plotly_chart(fig_hist, use_container_width=True, key="risk_distribution_hist")

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
        st.plotly_chart(fig_purpose, use_container_width=True, key="loan_purpose_chart")

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

