import numpy as np
from src.config import WINDOW_SIZE, STEP_SIZE, SENSOR_COLS

def make_windows(df):
    X_list, y_list = [], []

    for fname, group in df.groupby('source_file'):
        label = int(group['star_label'].iloc[0]) - 1
        values = group[SENSOR_COLS].values.astype(np.float32)

        for start in range(0, len(values) - WINDOW_SIZE + 1, STEP_SIZE):
            X_list.append(values[start : start + WINDOW_SIZE])
            y_list.append(label)

    return np.array(X_list), np.array(y_list)