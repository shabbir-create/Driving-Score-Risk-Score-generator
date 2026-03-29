import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

from src.config import SENSOR_COLS, WINDOW_SIZE, STEP_SIZE

# -----------------------
# Load model + scaler
# -----------------------
@st.cache_resource
def load_all():
    model = load_model("models/lstm_model.h5")
    with open("results/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_all()

# -----------------------
# Functions
# -----------------------
def create_windows(data):
    windows = []
    for i in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
        windows.append(data[i:i + WINDOW_SIZE])
    return np.array(windows)

def calculate_score(predictions):
    avg_probs = np.mean(predictions, axis=0)
    weights = np.array([20, 40, 60, 80, 100])
    return round(np.dot(avg_probs, weights), 2)

def get_status(score):
    if score >= 90:
        return "🟢 SMOOTH"
    elif score >= 80:
        return "🟢 SAFEST"
    elif score >= 60:
        return "🟡 SAFE"
    elif score >= 40:
        return "🟠 RISKY"
    else:
        return "🔴 DANGEROUS"

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Driving Behavior Detection", layout="wide")

st.title("🚗 Driving Behavior Detection System")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

# -----------------------
# Main Logic
# -----------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # -----------------------
    # Preview
    # -----------------------
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # -----------------------
    # Prediction
    # -----------------------
    st.subheader("🤖 Model Prediction")

    data = df[SENSOR_COLS].values

    windows = create_windows(data)

    X = scaler.transform(
        windows.reshape(-1, windows.shape[-1])
    ).reshape(windows.shape)

    preds = model.predict(X, verbose=0)

    score = calculate_score(preds)
    status = get_status(score)

    # -----------------------
    # Result
    # -----------------------
    st.subheader("🎯 Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Driving Score", f"{score}/100")

    with col2:
        st.success(f"Status: {status}")

    # -----------------------
    # Graphs (MOVED TO BOTTOM ✅)
    # -----------------------
    st.subheader("📊 Sensor Visualization")

    df['Acc_Mag'] = np.sqrt(
        df['X_Acc']**2 + df['Y_Acc']**2 + df['Z_Acc']**2
    )

    df['Gyro_Mag'] = np.sqrt(
        df['X_Gyro']**2 + df['Y_Gyro']**2 + df['Z_Gyro']**2
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Accelerometer Magnitude")
        st.line_chart(df['Acc_Mag'])

    with col2:
        st.write("### Gyroscope Magnitude")
        st.line_chart(df['Gyro_Mag'])

    st.write("### Raw Sensor Data")
    st.line_chart(df[['X_Acc', 'Y_Acc', 'Z_Acc']])
    st.line_chart(df[['X_Gyro', 'Y_Gyro', 'Z_Gyro']])

else:
    st.info("👆 Please upload a CSV file to start analysis")