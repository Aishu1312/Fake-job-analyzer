"""
Train (or retrain) the fake-job detection model.

Run from the fake-job-analyzer directory:
    python -m model.train_model
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
VEC_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")


def train():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    required_cols = {"description", "fraudulent"}
    missing = required_cols - set(df.columns)
    if missing:
        available = list(df.columns)
        raise ValueError(
            f"Dataset is missing columns: {missing}. "
            f"Available columns: {available}"
        )

    df = df[["description", "fraudulent"]].dropna()
    df["description"] = df["description"].astype(str)

    X, y = df["description"], df["fraudulent"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10_000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    print("\nEvaluation on test set:")
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VEC_PATH}")
    return model, vectorizer


if __name__ == "__main__":
    train()
