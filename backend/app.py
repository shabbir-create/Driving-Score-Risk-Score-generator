from fastapi import FastAPI
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = FastAPI()

# =========================
# LOAD MODEL + SCALER
# =========================
model = load_model("models/model.keras",compile=False)

with open("results/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

WINDOW_SIZE = 120


# =========================
# SCORE (HIGH = SAFE)
# =========================
def calculate_score(preds):
    preds = np.array(preds)

    if len(preds.shape) == 2:
        preds = preds[0]

    weights = np.array([20, 40, 60, 80, 100])  # safety weights

    score = np.dot(preds, weights)

    return round(float(score), 2)


# =========================
# STATUS (5 CLASSES)
# =========================
def get_status(score):
    if score >= 80:
        return "🟢 Very Safe"
    elif score >= 60:
        return "🟢 Safe"
    elif score >= 40:
        return "🟡 Moderate"
    elif score >= 20:
        return "🟠 Risky"
    else:
        return "🔴 Dangerous"


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "API running 🚀"}


@app.post("/predict")
def predict(input_data: dict):
    try:
        data = input_data.get("data")
        arr = np.array(data)

        # ensure 2D
        if len(arr.shape) != 2:
            return {"error": "Input must be 2D"}

        timesteps, features = arr.shape

        # =========================
        # PADDING / TRIMMING
        # =========================
        if timesteps < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - timesteps, features))
            arr = np.vstack([arr, pad])
        elif timesteps > WINDOW_SIZE:
            arr = arr[-WINDOW_SIZE:]

        # =========================
        # SCALE
        # =========================
        arr = scaler.transform(arr)

        # =========================
        # RESHAPE
        # =========================
        arr = arr.reshape(1, WINDOW_SIZE, features)

        # =========================
        # PREDICT
        # =========================
        preds = model.predict(arr, verbose=0)

        # =========================
        # SCORE + STATUS
        # =========================
        score = calculate_score(preds)
        status = get_status(score)

        return {
            "score": score,
            "status": status
        }

    except Exception as e:
        return {"error": str(e)}