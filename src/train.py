import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from src.config import MODEL_SAVE, RESULTS_DIR
from src.model import build_model

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, 6))

    X_train = scaler.transform(X_train.reshape(-1, 6)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, 6)).reshape(X_test.shape)

    model = build_model()

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE, save_best_only=True)
    ]

    model.fit(X_train, to_categorical(y_train, 5),
              validation_split=0.15,
              epochs=50,
              batch_size=64,
              callbacks=callbacks)

    with open(f"{RESULTS_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return model, X_test, y_test