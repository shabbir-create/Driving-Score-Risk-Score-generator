import pandas as pd
from pathlib import Path
from src.config import SENSOR_COLS

def load_dataset(root):
    all_dfs = []

    for star in range(1, 6):
        folder = Path(root) / f"{star} star"
        csv_files = list(folder.glob("*.csv"))

        for fpath in csv_files:
            df = pd.read_csv(fpath)

            if all(col in df.columns for col in SENSOR_COLS):
                df['star_label'] = star
                df['source_file'] = fpath.name
                all_dfs.append(df[SENSOR_COLS + ['star_label', 'source_file']])

    return pd.concat(all_dfs, ignore_index=True)