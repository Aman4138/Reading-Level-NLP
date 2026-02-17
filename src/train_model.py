import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = os.path.join("data", "clear_corpus.csv")
MODEL_PATH = os.path.join("models", "readability_model.joblib")

TEXT_COLUMN = "Excerpt"
TARGET_COLUMN = "Flesch-Reading-Ease"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    print("Columns in CSV:", df.columns.tolist())

    df = df[[TEXT_COLUMN, TARGET_COLUMN]].dropna()

    if len(df) > 5000:
        df = df.sample(5000, random_state=42)
        print("Sampled 5000 rows for initial experiments.")
    return df


def build_pipeline() -> Pipeline:
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=5,
    )
    model = Ridge(alpha=1.0, random_state=42)
    pipe = Pipeline([
        ("tfidf", tfidf),
        ("regressor", model),
    ])
    return pipe


def main():
    os.makedirs("models", exist_ok=True)

    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Dataset size: {len(df)}")

    X = df[TEXT_COLUMN].astype(str)
    y = df[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Building pipeline...")
    pipe = build_pipeline()

    print("Training model...")
    pipe.fit(X_train, y_train)

    print("Evaluating on validation set...")
    y_pred = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)  
    rmse = np.sqrt(mse)                      
    r2 = r2_score(y_test, y_pred)

    print(f"Validation RMSE: {rmse:.3f}")
    print(f"Validation R²:   {r2:.3f}")

    print(f"Saving model to {MODEL_PATH} ...")
    joblib.dump(pipe, MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
