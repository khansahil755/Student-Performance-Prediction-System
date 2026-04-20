"""Train and save a simple Logistic Regression model + scaler."""

import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import FEATURE_COLUMNS, get_models_path, load_dataset, save_model


def load_and_prepare_data(csv_path=None):
    """Load the cleaned dataset and return features, labels, and the full dataframe."""
    X, y, df = load_dataset(csv_path)
    if X is None or y is None or df is None or df.empty:
        raise FileNotFoundError("student_data.csv not found or contains no valid rows.")
    return X, y, df


def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset while preserving class balance across train and test sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_scaler_and_transform(X_train, X_test=None):
    """Fit a StandardScaler on training features and optionally transform test features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is None:
        return scaler, X_train_scaled

    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def evaluate_model(model, X_test, y_test):
    """Compute and print only accuracy."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", round(accuracy, 4))
    return accuracy


def train_all(csv_path=None, save=True):
    """Load data, scale features, train Logistic Regression, and save artifacts."""
    X, y, cleaned_df = load_and_prepare_data(csv_path)
    X = X[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler, X_train_scaled, X_test_scaled = get_scaler_and_transform(X_train, X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)

    accuracy = evaluate_model(model, X_test_scaled, y_test)

    models_dir = get_models_path()
    if save:
        save_model(model, os.path.join(models_dir, "trained_model.pkl"))
        save_model(scaler, os.path.join(models_dir, "scaler.pkl"))

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy,
        "rows_used": len(cleaned_df),
    }


if __name__ == "__main__":
    print("Training Student Performance Prediction models...")
    results = train_all()
    print("\nModels saved in models/")
    print("Rows used:", results["rows_used"])
    print("Accuracy:", round(results["accuracy"], 4))
