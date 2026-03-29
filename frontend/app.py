import streamlit as st
import pandas as pd
import numpy as np
import requests

# =========================
# CONFIG
# =========================
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Driving Risk Predictor", layout="wide")

st.title("🚗 Driving Risk Predictor")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # =========================
    # ORIGINAL GRAPH LOGIC (UNCHANGED)
    # =========================
    st.subheader("📈 Sensor Data")

    st.line_chart(df[['X_Acc', 'Y_Acc', 'Z_Acc']])
    st.line_chart(df[['X_Gyro', 'Y_Gyro', 'Z_Gyro']])

    # =========================
    # PREDICT BUTTON
    # =========================
    if st.button("🚀 Predict"):

        try:
            # SAME DATA → backend
            data = df[['X_Acc','Y_Acc','Z_Acc','X_Gyro','Y_Gyro','Z_Gyro']].values.tolist()

            response = requests.post(API_URL, json={"data": data})
            result = response.json()

            if "score" in result:

                score = result["score"]
                status = result["status"]

                st.subheader("🎯 Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Score", f"{score}/100")

                with col2:
                    st.metric("Status", status)

            else:
                st.error(result)

        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("Upload CSV to start")