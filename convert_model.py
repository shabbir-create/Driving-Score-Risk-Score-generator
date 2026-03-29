from tensorflow.keras.models import load_model
model=load_model("backend/models/lstm_model.h5", compile=False)
model.save("backend/models/model.keras")
print("suceessfully converted to keras format")