import joblib
from tensorflow.keras.models import load_model
model = load_model("models/lstm_model.h5")
joblib.dump(model, "model.pkl")
print("Model converted and saved as model.pkl")