import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/xgboost_model.joblib')

st.title('Banking Early Warning Dashboard')

uploaded = st.file_uploader('Upload Loan Dataset')

if uploaded:
    df = pd.read_csv(uploaded)

    preds = model.predict_proba(df)[:, 1]

    df['Risk Score'] = preds

    st.dataframe(df)
    st.bar_chart(df['Risk Score'])