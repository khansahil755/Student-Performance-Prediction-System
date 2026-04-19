"""
Utility helpers for the Student Performance Prediction System.
Includes path helpers, dataset cleaning, validation, and model persistence.
"""

import os
import pickle

import pandas as pd


FEATURE_COLUMNS = ["attendance", "internal", "assignment", "previous"]
TARGET_COLUMN = "result"
FEATURE_LIMITS = {
    "attendance": (0, 100),
    "internal": (0, 30),
    "assignment": (0, 20),
    "previous": (0, 100),
}
TARGET_LABELS = {"pass": "Pass", "average": "Average", "fail": "Fail"}


def get_project_root():
    """Return absolute path to the project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path():
    """Return path to the main student dataset."""
    base = get_project_root()
    default_path = os.path.join(base, "data", "student_data.csv")
    fallback_path = os.path.join(base, "student_data.csv")
    if os.path.exists(default_path):
        return default_path
    return fallback_path


def get_models_path():
    """Return path to the models directory."""
    return os.path.join(get_project_root(), "models")


def validate_required_columns(df):
    """Return missing required dataset columns, if any."""
    expected_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    return [column for column in expected_columns if column not in df.columns]


def clean_dataset(df):
    """
    Standardize the dataset before training.
    Drops invalid rows, normalizes labels, and clips numeric features to valid ranges.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN])

    cleaned = df.copy()
    missing_columns = validate_required_columns(cleaned)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")

    cleaned = cleaned[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()

    for feature in FEATURE_COLUMNS:
        cleaned[feature] = pd.to_numeric(cleaned[feature], errors="coerce")

    cleaned[TARGET_COLUMN] = (
        cleaned[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(TARGET_LABELS)
    )

    cleaned = cleaned.dropna()

    for feature, (minimum, maximum) in FEATURE_LIMITS.items():
        cleaned = cleaned[cleaned[feature].between(minimum, maximum)]

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def load_dataset(csv_path=None):
    """
    Load and clean the dataset.
    Returns: (X DataFrame, y Series, cleaned_df) or (None, None, None) if file not found.
    """
    path = csv_path or get_data_path()
    if not os.path.exists(path):
        return None, None, None

    df = pd.read_csv(path)
    cleaned_df = clean_dataset(df)
    if cleaned_df.empty:
        return None, None, cleaned_df

    X = cleaned_df[FEATURE_COLUMNS]
    y = cleaned_df[TARGET_COLUMN]
    return X, y, cleaned_df


def build_feature_frame(attendance, internal, assignment, previous):
    """Create a single-row feature DataFrame with the expected training columns."""
    return pd.DataFrame(
        [[float(attendance), float(internal), float(assignment), float(previous)]],
        columns=FEATURE_COLUMNS,
    )


def validate_input(attendance, internal, assignment, previous):
    """
    Validate user inputs for prediction.
    Returns: (is_valid: bool, error_message: str)
    """
    values = {
        "Attendance": attendance,
        "Internal marks": internal,
        "Assignment marks": assignment,
        "Previous semester marks": previous,
    }

    numeric_values = {}
    for label, value in values.items():
        try:
            numeric_values[label] = float(value)
        except (TypeError, ValueError):
            return False, "All values must be numbers."

    limit_map = {
        "Attendance": FEATURE_LIMITS["attendance"],
        "Internal marks": FEATURE_LIMITS["internal"],
        "Assignment marks": FEATURE_LIMITS["assignment"],
        "Previous semester marks": FEATURE_LIMITS["previous"],
    }

    for label, value in numeric_values.items():
        minimum, maximum = limit_map[label]
        if not minimum <= value <= maximum:
            return False, f"{label} must be between {minimum} and {maximum}."

    return True, ""


def save_model(model, filepath):
    """Save a trained object using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as file:
        pickle.dump(model, file)


def load_model(filepath):
    """Load a pickled model object from disk."""
    with open(filepath, "rb") as file:
        return pickle.load(file)
