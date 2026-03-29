import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras.models import load_model

from src.config import SENSOR_COLS, WINDOW_SIZE, STEP_SIZE

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Driving AI",
    layout="wide",
    page_icon="🚗"
)

# -----------------------
# CUSTOM CSS (Premium UI 🔥)
# -----------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stMetric {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        background-color: #00c6ff;
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_all():
    model = load_model("models/lstm_model.h5", compile=False)
    with open("results/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_all()

# -----------------------
# FUNCTIONS
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

@st.cache_data
def prepare_data(df):
    df['Acc_Mag'] = np.sqrt(df['X_Acc']**2 + df['Y_Acc']**2 + df['Z_Acc']**2)
    df['Gyro_Mag'] = np.sqrt(df['X_Gyro']**2 + df['Y_Gyro']**2 + df['Z_Gyro']**2)
    return df

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("⚙️ Settings")

show_graphs = st.sidebar.checkbox("Show Graphs", value=True)
max_points = st.sidebar.slider("Graph Resolution", 100, 2000, 500)

# -----------------------
# MAIN UI
# -----------------------
st.title("🚗 Driving Behavior AI Dashboard")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # Tabs for clean UI
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Graphs", "📄 Data"])

    # -----------------------
    # TAB 1 → RESULT
    # -----------------------
    with tab1:

        data = df[SENSOR_COLS].values
        windows = create_windows(data)

        X = scaler.transform(
            windows.reshape(-1, windows.shape[-1])
        ).reshape(windows.shape)

        with st.spinner("🤖 Analyzing driving behavior..."):
            preds = model.predict(X, verbose=0)

        score = calculate_score(preds)
        status = get_status(score)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("🚗 Driving Score", f"{score}/100")

        with col2:
            st.metric("📌 Status", status)

    # -----------------------
    # TAB 2 → GRAPHS
    # -----------------------
    with tab2:

        if show_graphs:

            df = prepare_data(df)

            if len(df) > max_points:
                df_plot = df.iloc[::len(df)//max_points]
            else:
                df_plot = df

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Accelerometer")
                st.line_chart(df_plot['Acc_Mag'])

            with col2:
                st.subheader("Gyroscope")
                st.line_chart(df_plot['Gyro_Mag'])

            st.subheader("Raw Sensor Data")
            st.line_chart(df_plot[['X_Acc', 'Y_Acc', 'Z_Acc']])
            st.line_chart(df_plot[['X_Gyro', 'Y_Gyro', 'Z_Gyro']])

    # -----------------------
    # TAB 3 → DATA
    # -----------------------
    with tab3:
        st.dataframe(df.head(100))

else:
    st.info("👆 Upload a CSV file to start")
