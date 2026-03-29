from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from src.config import WINDOW_SIZE

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
    return model