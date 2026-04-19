"""
Prediction helpers for the Student Performance Prediction System.
Loads trained models and returns class predictions, confidence, and explanation text.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import build_feature_frame, get_models_path, load_model, validate_input


def get_model_and_scaler():
    """Load the saved model and scaler from disk."""
    models_dir = get_models_path()
    model_path = os.path.join(models_dir, "trained_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    return load_model(model_path), load_model(scaler_path)


def predict_result(attendance, internal, assignment, previous, model=None, scaler=None):
    """
    Predict student performance and return (prediction, confidence, probabilities).
    """
    valid, error_message = validate_input(attendance, internal, assignment, previous)
    if not valid:
        raise ValueError(error_message)

    if model is None or scaler is None:
        model, scaler = get_model_and_scaler()

    if model is None or scaler is None:
        return "Unknown", 0.0, {}

    feature_frame = build_feature_frame(attendance, internal, assignment, previous)
    transformed = scaler.transform(feature_frame)

    prediction = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]
    class_labels = model.classes_

    probability_map = {
        label: round(float(probability), 4)
        for label, probability in sorted(
            zip(class_labels, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    confidence = max(probability_map.values()) if probability_map else 0.0
    return prediction, confidence, probability_map


def explain_prediction(attendance, internal, assignment, previous, prediction, model=None):
    """
    Generate a short viva-friendly explanation using the strongest visible factors.
    """
    scores = {
        "attendance": float(attendance),
        "internal": float(internal),
        "assignment": float(assignment),
        "previous": float(previous),
    }

    strengths = []
    risks = []

    if scores["attendance"] >= 85:
        strengths.append("strong attendance")
    elif scores["attendance"] < 60:
        risks.append("low attendance")

    if scores["internal"] >= 20:
        strengths.append("good internal marks")
    elif scores["internal"] < 10:
        risks.append("weak internal marks")

    if scores["assignment"] >= 14:
        strengths.append("good assignment performance")
    elif scores["assignment"] < 8:
        risks.append("low assignment marks")

    if scores["previous"] >= 65:
        strengths.append("solid previous semester marks")
    elif scores["previous"] < 40:
        risks.append("poor previous semester marks")

    if prediction == "Pass":
        if strengths:
            return f"The model predicts Pass mainly because of {', '.join(strengths[:2])}."
        return "The model predicts Pass because the overall academic profile is above the risk zone."

    if prediction == "Fail":
        if risks:
            return f"The model predicts Fail mainly because of {', '.join(risks[:2])}."
        return "The model predicts Fail because the combined academic indicators are weak."

    if strengths and risks:
        return f"The model predicts Average because the student shows {strengths[0]} but also has {risks[0]}."
    if strengths:
        return f"The model predicts Average because the student has some positives such as {strengths[0]}, but not enough for Pass."
    if risks:
        return f"The model predicts Average because the profile is moderate overall, with some concern around {risks[0]}."
    return f"The model predicts {prediction} based on the overall balance of the four academic inputs."
