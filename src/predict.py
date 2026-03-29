import numpy as np
import pandas as pd
import os
from src.config import SENSOR_COLS, WINDOW_SIZE, STEP_SIZE

def calculate_score(predictions):
    weights = np.array([20, 40, 60, 80, 100])
    return round(np.dot(np.mean(predictions, axis=0), weights), 2)

def run_test(model, scaler, file):
    df = pd.read_csv(file)
    data = df[SENSOR_COLS].values

    windows = [data[i:i+WINDOW_SIZE] for i in range(0, len(data)-WINDOW_SIZE+1, STEP_SIZE)]
    X = scaler.transform(np.array(windows).reshape(-1, 6)).reshape(len(windows), WINDOW_SIZE, 6)

    preds = model.predict(X)
    score = calculate_score(preds)

    print(f"{file} → Score: {score}")