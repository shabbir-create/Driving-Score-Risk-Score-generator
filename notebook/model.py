# --- Cell
import os
# Set dataset path
DATASET_ROOT = "LSTM MODEL/dataset"

# Debug
print(os.listdir(DATASET_ROOT))

# --- Cell ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Settings
WINDOW_SIZE = 120
STEP_SIZE = 20
SENSOR_COLS = ['X_Acc', 'Y_Acc', 'Z_Acc', 'X_Gyro', 'Y_Gyro', 'Z_Gyro']
RESULTS_DIR = "/content/drive/MyDrive/DrivingRiskProject/results"
MODEL_SAVE = "/content/drive/MyDrive/DrivingRiskProject/lstm_driving_model.h5"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_dataset(root):
    all_dfs = []

    for star in range(1, 6):
        folder = Path(root) / f"{star} star"
        csv_files = list(folder.glob("*.csv"))
        print(f"  Processing {star}★ — {len(csv_files)} files")

        for fpath in csv_files:
            df = pd.read_csv(fpath)

            if all(col in df.columns for col in SENSOR_COLS):
                df['star_label'] = star
                df['source_file'] = fpath.name
                all_dfs.append(df[SENSOR_COLS + ['star_label', 'source_file']])

    return pd.concat(all_dfs, ignore_index=True)
def make_windows(df):
    X_list, y_list = [], []

    for fname, group in df.groupby('source_file'):
        label = int(group['star_label'].iloc[0]) - 1
        values = group[SENSOR_COLS].values.astype(np.float32)

        for start in range(0, len(values) - WINDOW_SIZE + 1, STEP_SIZE):
            X_list.append(values[start : start + WINDOW_SIZE])
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    # Print dataset shape
    print(f"Final dataset shape: {X.shape}")

    return X, y

def build_model():
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(WINDOW_SIZE, 6)),
        Dropout(0.3),
        BatchNormalization(),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# --- Cell ---
print("[1] Loading Data...")
df_raw = load_dataset(DATASET_ROOT)

X, y = make_windows(df_raw)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Print shapes
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Scale
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, 6))

X_train_n = scaler.transform(X_train.reshape(-1, 6)).reshape(len(X_train), WINDOW_SIZE, 6)
X_test_n = scaler.transform(X_test.reshape(-1, 6)).reshape(len(X_test), WINDOW_SIZE, 6)
# Train
model = build_model()
callbacks = [EarlyStopping(patience=8, restore_best_weights=True),
             ModelCheckpoint(MODEL_SAVE, save_best_only=True)]

print("[2] Training Starting...")
history = model.fit(X_train_n, to_categorical(y_train, 5),
                    validation_split=0.15, epochs=50, batch_size=64, callbacks=callbacks)

# Save Scaler
with open(f"{RESULTS_DIR}/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print(" Training Complete!")

# --- Cell ---
from sklearn.metrics import classification_report, confusion_matrix

def show_detailed_metrics(model, X_test_n, y_test):
    # 1. Get Predictions
    y_probs = model.predict(X_test_n, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # 2. Print Classification Report (Precision, Recall, F1, Support)
    target_names = ['1★ Dangerous', '2★ Poor', '3★ Average', '4★ Good', '5★ Safe']
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("="*60)

    # 3. Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix: Actual vs Predicted Classes')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()



# --- EXECUTION ---
# 1. Show the metrics table (Precision/Recall/F1)
show_detailed_metrics(model, X_test_n, y_test)



# --- Cell ---
def calculate_driving_score(predictions):
    avg_probs = np.mean(predictions, axis=0)
    weights = np.array([20, 40, 60, 80, 100]) # 1* is 20 points, 5* is 100 points
    return round(np.dot(avg_probs, weights), 2)

def run_final_test(csv_path):
    df = pd.read_csv(csv_path)
    data = df[SENSOR_COLS].values
    # Prep windows
    test_windows = [data[i:i+WINDOW_SIZE] for i in range(0, len(data)-WINDOW_SIZE+1, STEP_SIZE)]
    X_val = scaler.transform(np.array(test_windows).reshape(-1, 6)).reshape(len(test_windows), WINDOW_SIZE, 6)

    # Predict
    preds = model.predict(X_val, verbose=0)
    score = calculate_driving_score(preds)

    print(f"\nRide: {os.path.basename(csv_path)}")
    print(f"DRIVING SCORE: {score}/100")
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
    print(f"STATUS: {status}")

# --- Cell ---
# Run a test on a 1-star file
test_file = f"{DATASET_ROOT}/testing/vikram_5.1.csv"
if os.path.exists(test_file):
    run_final_test(test_file)

# --- Cell ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualize_csv(file_path):
    df = pd.read_csv(file_path)

    # Add magnitude features
    df['Acc_Mag'] = np.sqrt(df['X_Acc']**2 + df['Y_Acc']**2 + df['Z_Acc']**2)
    df['Gyro_Mag'] = np.sqrt(df['X_Gyro']**2 + df['Y_Gyro']**2 + df['Z_Gyro']**2)

    plt.figure(figsize=(14, 8))

    # Accelerometer
    plt.subplot(2, 1, 1)
    plt.plot(df['X_Acc'], label='X Acc')
    plt.plot(df['Y_Acc'], label='Y Acc')
    plt.plot(df['Z_Acc'], label='Z Acc')
    plt.plot(df['Acc_Mag'], label='Acc Mag', linewidth=2)
    plt.title("Accelerometer")
    plt.legend()
    plt.grid()

    # Gyroscope
    plt.subplot(2, 1, 2)
    plt.plot(df['X_Gyro'], label='X Gyro')
    plt.plot(df['Y_Gyro'], label='Y Gyro')
    plt.plot(df['Z_Gyro'], label='Z Gyro')
    plt.plot(df['Gyro_Mag'], label='Gyro Mag', linewidth=2)
    plt.title("Gyroscope")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
test_file = f"{DATASET_ROOT}/testing/vikram_5.1.csv"

if os.path.exists(test_file):

    # 🔹 Step 1: Visualize
    visualize_csv(test_file)


