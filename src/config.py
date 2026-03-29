import os

DATASET_ROOT = "dataset"
WINDOW_SIZE = 120
STEP_SIZE = 20

SENSOR_COLS = ['X_Acc', 'Y_Acc', 'Z_Acc', 'X_Gyro', 'Y_Gyro', 'Z_Gyro']

RESULTS_DIR = "results"
MODEL_SAVE = "models/lstm_model.h5"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)