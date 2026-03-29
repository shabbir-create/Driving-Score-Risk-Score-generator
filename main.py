from src.config import DATASET_ROOT
from src.data_loader import load_dataset
from src.preprocess import make_windows
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    df = load_dataset(DATASET_ROOT)
    X, y = make_windows(df)

    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()