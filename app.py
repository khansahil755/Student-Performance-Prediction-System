import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import explain_prediction, predict_result  # noqa: E402
from src.train_model import train_all  # noqa: E402
from src.utils import get_models_path, validate_input  # noqa: E402


st.set_page_config(page_title="Student Performance Prediction", layout="centered")


def render_result_card(prediction: str, confidence: float) -> None:
    class_map = {"Pass": "result-pass", "Average": "result-average", "Fail": "result-fail"}
    style_class = class_map.get(prediction, "result-average")
    st.markdown(
        f"""
        <div class="result-card {style_class}">
            <div style="opacity:0.85;">Prediction</div>
            <div style="font-size:1.6rem; font-weight:700; margin:0.15rem 0;">{prediction}</div>
            <div>Confidence: {confidence * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
      .result-card { border-radius: 12px; padding: 1rem; border-left: 5px solid; }
      .result-pass { background: #edf8f0; border-color: #2e7d32; }
      .result-average { background: #fff8e8; border-color: #d89b00; }
      .result-fail { background: #fff0f0; border-color: #c43d3d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Student Performance Prediction")
st.caption("Inputs → (optional) Train → Predict")

models_dir = get_models_path()
model_path = os.path.join(models_dir, "trained_model.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")
models_ready = os.path.exists(model_path) and os.path.exists(scaler_path)

if not models_ready:
    st.warning("Model files not found. Click Train once before using Predict.")

with st.expander("Train model", expanded=not models_ready):
    st.write("Train once to create `models/trained_model.pkl` and `models/scaler.pkl`.")
    if st.button("Train", type="primary", width="stretch"):
        with st.spinner("Training..."):
            result = train_all(save=True)
        st.success(f"Done. Accuracy: {result['accuracy'] * 100:.2f}%")
        st.rerun()

st.subheader("Inputs")
attendance = st.slider("Attendance (%)", 0, 100, 75)
internal = st.slider("Internal marks (out of 30)", 0, 30, 20)
assignment = st.slider("Assignment marks (out of 20)", 0, 20, 14)
previous = st.slider("Previous semester marks (out of 100)", 0, 100, 65)

valid, err = validate_input(attendance, internal, assignment, previous)
if not valid:
    st.error(err)

st.subheader("Prediction")
if st.button("Predict", type="primary", width="stretch", disabled=not valid or not models_ready):
    pred, confidence, prob = predict_result(attendance, internal, assignment, previous)
    render_result_card(pred, confidence)
    st.info(explain_prediction(attendance, internal, assignment, previous, pred))
    st.write("Probabilities:", prob)
