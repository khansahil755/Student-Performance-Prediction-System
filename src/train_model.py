"""
Training pipeline for the Student Performance Prediction System.
Uses a tuned Logistic Regression model as the main classifier and a balanced Decision Tree for comparison.
"""

import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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


def train_logistic_regression(X_train, y_train):
    """Tune and train the main Logistic Regression model."""
    parameter_grid = {
        "C": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        "solver": ["lbfgs"],
        "class_weight": [None, "balanced"],
    }
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        LogisticRegression(max_iter=3000, random_state=42),
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=None,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def train_decision_tree(X_train, y_train):
    """Tune and train a Decision Tree for comparison and ensembling."""
    parameter_grid = {
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 4, 8],
        "class_weight": [None, "balanced"],
    }
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=None,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Compute model metrics and return them in a structured format."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    labels = ["Average", "Fail", "Pass"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    report = classification_report(y_test, predictions, labels=labels, zero_division=0)

    print(f"\n--- {model_name} ---")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))
    print("Confusion Matrix:\n", matrix)
    print("Classification Report:\n", report)

    return accuracy, macro_f1, balanced_acc, matrix, report


def _build_ensemble_weights(logistic_f1, tree_f1):
    """Convert model quality scores into stable probability-blend weights."""
    total = float(logistic_f1 + tree_f1)
    if total <= 0:
        return {"primary": 0.7, "aux": 0.3}

    primary = max(0.55, logistic_f1 / total)
    aux = 1.0 - primary
    return {"primary": round(primary, 4), "aux": round(aux, 4)}


def train_all(csv_path=None, save=True):
    """
    Full training flow: load data, split, scale, tune the main model, evaluate, and save.
    """
    X, y, cleaned_df = load_and_prepare_data(csv_path)
    X = X[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler, X_train_scaled, X_test_scaled = get_scaler_and_transform(X_train, X_test)

    logistic_model, best_params, cv_score = train_logistic_regression(X_train_scaled, y_train)
    logistic_accuracy, logistic_f1, logistic_bal_acc, logistic_cm, logistic_report = evaluate_model(
        logistic_model,
        X_test_scaled,
        y_test,
        "Logistic Regression",
    )

    tree_model, dt_best_params, dt_cv_score = train_decision_tree(X_train, y_train)
    tree_accuracy, tree_f1, tree_bal_acc, tree_cm, tree_report = evaluate_model(
        tree_model,
        X_test,
        y_test,
        "Decision Tree",
    )
    ensemble_weights = _build_ensemble_weights(logistic_f1, tree_f1)

    models_dir = get_models_path()
    if save:
        # Bundle metadata so prediction can combine strong models reliably.
        model_bundle = {
            "version": 2,
            "primary_model": logistic_model,
            "aux_model": tree_model,
            "feature_columns": FEATURE_COLUMNS,
            "ensemble_weights": ensemble_weights,
        }
        save_model(model_bundle, os.path.join(models_dir, "trained_model.pkl"))
        save_model(tree_model, os.path.join(models_dir, "decision_tree_model.pkl"))
        save_model(scaler, os.path.join(models_dir, "scaler.pkl"))

    return {
        "lr_model": logistic_model,
        "dt_model": tree_model,
        "scaler": scaler,
        "lr_accuracy": logistic_accuracy,
        "lr_f1_macro": logistic_f1,
        "lr_balanced_accuracy": logistic_bal_acc,
        "dt_accuracy": tree_accuracy,
        "dt_f1_macro": tree_f1,
        "dt_balanced_accuracy": tree_bal_acc,
        "lr_cm": logistic_cm,
        "dt_cm": tree_cm,
        "lr_report": logistic_report,
        "dt_report": tree_report,
        "lr_best_params": best_params,
        "lr_cv_score": cv_score,
        "dt_best_params": dt_best_params,
        "dt_cv_score": dt_cv_score,
        "ensemble_weights": ensemble_weights,
        "rows_used": len(cleaned_df),
    }


if __name__ == "__main__":
    print("Training Student Performance Prediction models...")
    results = train_all()
    print("\nModels saved in models/")
    print("Rows used:", results["rows_used"])
    print("Best Logistic Regression params:", results["lr_best_params"])
    print("Logistic Regression CV F1 Macro:", round(results["lr_cv_score"], 4))
    print("Logistic Regression Accuracy:", round(results["lr_accuracy"], 4))
    print("Logistic Regression Test Macro F1:", round(results["lr_f1_macro"], 4))
    print("Best Decision Tree params:", results["dt_best_params"])
    print("Decision Tree CV F1 Macro:", round(results["dt_cv_score"], 4))
    print("Decision Tree Accuracy:", round(results["dt_accuracy"], 4))
    print("Decision Tree Test Macro F1:", round(results["dt_f1_macro"], 4))
    print("Ensemble Weights:", results["ensemble_weights"])
