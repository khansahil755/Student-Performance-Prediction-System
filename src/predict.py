"""
Prediction helpers for the Student Performance Prediction System.
Loads trained models and returns class predictions, confidence, and explanation text.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import build_feature_frame, get_models_path, load_model, validate_input


def get_model_and_scaler():
    """Load the saved model bundle (or legacy model) and scaler from disk."""
    models_dir = get_models_path()
    model_path = os.path.join(models_dir, "trained_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    return load_model(model_path), load_model(scaler_path)


def _extract_models(model_artifact):
    """Handle both legacy single-model artifacts and new bundled artifacts."""
    if isinstance(model_artifact, dict) and "primary_model" in model_artifact:
        primary_model = model_artifact["primary_model"]
        aux_model = model_artifact.get("aux_model")
        weights = model_artifact.get("ensemble_weights", {"primary": 0.7, "aux": 0.3})
        return primary_model, aux_model, weights

    return model_artifact, None, {"primary": 1.0, "aux": 0.0}


def _predict_with_probabilities(model, features):
    """Return (predicted_label, class_labels, probabilities) for one model."""
    prediction = model.predict(features)[0]
    if not hasattr(model, "predict_proba"):
        return prediction, [prediction], [1.0]

    probabilities = model.predict_proba(features)[0]
    class_labels = list(model.classes_)
    return prediction, class_labels, probabilities


def _blend_probabilities(primary_labels, primary_probs, aux_labels, aux_probs, weights):
    """Blend two probability distributions into a unified class map."""
    probability_map = {}

    for label, prob in zip(primary_labels, primary_probs):
        probability_map[label] = probability_map.get(label, 0.0) + float(prob) * float(weights["primary"])

    for label, prob in zip(aux_labels, aux_probs):
        probability_map[label] = probability_map.get(label, 0.0) + float(prob) * float(weights["aux"])

    total = sum(probability_map.values())
    if total > 0:
        for label in list(probability_map.keys()):
            probability_map[label] = probability_map[label] / total

    return probability_map


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
        raise FileNotFoundError(
            "Saved model artifacts are missing. Retrain the model from 'Model Info & Retrain' first."
        )

    feature_frame = build_feature_frame(attendance, internal, assignment, previous)
    transformed = scaler.transform(feature_frame) if scaler is not None else feature_frame

    primary_model, aux_model, weights = _extract_models(model)
    _, primary_labels, primary_probs = _predict_with_probabilities(primary_model, transformed)

    if aux_model is not None:
        _, aux_labels, aux_probs = _predict_with_probabilities(aux_model, feature_frame)
        merged = _blend_probabilities(primary_labels, primary_probs, aux_labels, aux_probs, weights)
    else:
        merged = {label: float(prob) for label, prob in zip(primary_labels, primary_probs)}

    sorted_items = sorted(merged.items(), key=lambda item: item[1], reverse=True)
    prediction = sorted_items[0][0]
    confidence = sorted_items[0][1]

    probability_map = {label: round(float(prob), 4) for label, prob in sorted_items}
    return prediction, confidence, probability_map


def explain_prediction(attendance, internal, assignment, previous, prediction, model=None):
    """
    Generate a short viva-friendly explanation using the strongest visible factors.
    """
    if prediction not in {"Pass", "Average", "Fail"}:
        return "Prediction is unavailable until a trained model is loaded."
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

    academic_strength = np.mean([scores["attendance"], scores["previous"]]) >= 72

    if prediction == "Pass":
        if strengths:
            return f"The model predicts Pass mainly because of {', '.join(strengths[:2])}."
        if academic_strength:
            return "The model predicts Pass due to a consistently strong overall academic profile."
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
