import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

from src.config import SENSOR_COLS, WINDOW_SIZE, STEP_SIZE

# -----------------------------
# Load model + scaler
# -----------------------------
model = load_model("models/lstm_model.h5")

with open("results/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# -----------------------------
# Create windows
# -----------------------------
def create_windows(data):
    windows = []
    for i in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
        windows.append(data[i:i + WINDOW_SIZE])
    return np.array(windows)


# -----------------------------
# Score calculation (same as Colab)
# -----------------------------
def calculate_driving_score(predictions):
    avg_probs = np.mean(predictions, axis=0)
    weights = np.array([20, 40, 60, 80, 100])
    return round(np.dot(avg_probs, weights), 2)


# -----------------------------
# Main test function
# -----------------------------
def run_final_test(csv_path):
    df = pd.read_csv(csv_path)
    data = df[SENSOR_COLS].values

    # windows
    test_windows = create_windows(data)

    # scale
    X = scaler.transform(
        test_windows.reshape(-1, test_windows.shape[-1])
    ).reshape(test_windows.shape)

    # predict
    preds = model.predict(X, verbose=0)

    score = calculate_driving_score(preds)

    print(f"\nRide: {os.path.basename(csv_path)}")
    print(f"Driving Score: {score}/100")

    # status logic
    if score >= 90:
        status = "SMOOTH"
    elif score >= 80:
        status = "SAFEST"
    elif score >= 60:
        status = "SAFE"
    elif score >= 40:
        status = "RISKY"
    else:
        status = "DANGEROUS"

    print(f"Status: {status}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    file = "dataset/testing/shabbir_backpack_rating5_smooth.csv"   # change file here
    run_final_test(file)