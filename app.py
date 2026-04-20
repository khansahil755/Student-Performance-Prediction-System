"""
Student Performance Prediction System - Streamlit Web Application.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import explain_prediction, predict_result
from src.train_model import train_all
from src.utils import get_data_path, get_models_path, validate_input


st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(25, 118, 210, 0.10), transparent 28%),
                linear-gradient(180deg, #f4f7fb 0%, #eef3f8 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: linear-gradient(135deg, #17324d 0%, #234d73 100%);
            color: #ffffff;
            padding: 1.75rem;
            border-radius: 18px;
            box-shadow: 0 16px 34px rgba(23, 50, 77, 0.14);
            margin-bottom: 1.25rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.88);
            margin-bottom: 1rem;
            max-width: 760px;
        }
        .panel-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(20, 45, 70, 0.08);
            border-radius: 16px;
            padding: 1.15rem;
            box-shadow: 0 10px 22px rgba(29, 53, 87, 0.06);
            margin-bottom: 1rem;
        }
        .mini-card {
            background: #ffffff;
            border: 1px solid rgba(20, 45, 70, 0.08);
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 8px 18px rgba(29, 53, 87, 0.05);
        }
        .mini-label {
            font-size: 0.85rem;
            color: #58708c;
            margin-bottom: 0.25rem;
        }
        .mini-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #17324d;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #17324d;
            margin-bottom: 0.75rem;
        }
        .stMarkdown, .stCaption, .stText, .stWrite, p, label, small {
            color: #1f3448 !important;
        }
        .stSlider label, .stSlider [data-testid="stTickBar"], .stSlider div[role="slider"] {
            color: #17324d !important;
        }
        .stSlider [data-testid="stThumbValue"] {
            color: #17324d !important;
            font-weight: 600 !important;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p {
            color: #e8eef5 !important;
        }
        .result-card {
            border-radius: 16px;
            padding: 1.15rem;
            margin-top: 0.75rem;
            border-left: 6px solid;
            box-shadow: 0 10px 22px rgba(29, 53, 87, 0.07);
        }
        .result-pass {
            background: #edf8f0;
            border-color: #2e7d32;
            color: #1f5f28;
        }
        .result-average {
            background: #fff8e8;
            border-color: #d89b00;
            color: #7a5a00;
        }
        .result-fail {
            background: #fff0f0;
            border-color: #c43d3d;
            color: #842727;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Student Performance Prediction System</div>
            <div class="hero-subtitle">
                A machine learning dashboard for predicting student outcomes from attendance
                and academic indicators.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label, value):
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(prediction, confidence):
    class_map = {
        "Pass": "result-pass",
        "Average": "result-average",
        "Fail": "result-fail",
    }
    style_class = class_map.get(prediction, "result-average")
    st.markdown(
        f"""
        <div class="result-card {style_class}">
            <div style="font-size:0.9rem; opacity:0.85;">Predicted Outcome</div>
            <div style="font-size:1.8rem; font-weight:700; margin:0.15rem 0;">{prediction}</div>
            <div style="font-size:1rem;">Confidence Score: {confidence * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_input_chart(attendance, internal, assignment, previous):
    labels = ["Attendance", "Internal", "Assignment", "Previous"]
    values = [
        float(attendance),
        (float(internal) / 30.0) * 100.0,
        (float(assignment) / 20.0) * 100.0,
        float(previous),
    ]
    colors = ["#2f6ea5", "#4d9a79", "#d99b43", "#9c5b7b"]

    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent of maximum")
    ax.set_title("Input strength (each bar is 0-100% of its range)", fontsize=11, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    raw_labels = [attendance, internal, assignment, previous]
    for bar, raw in zip(bars, raw_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(raw),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#17324d",
        )

    return fig


def build_distribution_chart(series, title):
    order = ["Pass", "Average", "Fail"]
    counts = pd.Series({label: int(series.eq(label).sum()) for label in order})
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    colors = ["#2e7d32", "#d89b00", "#c43d3d"]
    counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


render_hero()

st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Select a section",
    ["Predict", "Dataset Preview", "Model Info & Retrain", "Prediction History"],
    index=0,
)


if page == "Predict":
    models_dir = get_models_path()
    model_path = os.path.join(models_dir, "trained_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    models_ready = os.path.exists(model_path) and os.path.exists(scaler_path)

    if not models_ready:
        st.warning("Trained model files were not found. Train the model once before predicting.")
        if st.button("Train Model", type="primary", width="stretch"):
            with st.spinner("Training model..."):
                try:
                    result = train_all(save=True)
                    st.success(f"Model trained successfully (Accuracy: {result['accuracy'] * 100:.2f}%).")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Student Academic Inputs</div>', unsafe_allow_html=True)
        st.caption("Adjust values to estimate whether the student is likely to pass, average, or fail.")

        col1, col2 = st.columns(2)
        with col1:
            attendance = st.slider("Attendance (%)", 0, 100, 75)
            internal = st.slider("Internal Marks (out of 30)", 0, 30, 20)
        with col2:
            assignment = st.slider("Assignment Marks (out of 20)", 0, 20, 14)
            previous = st.slider("Previous Semester Marks (out of 100)", 0, 100, 65)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Input Visualization</div>', unsafe_allow_html=True)
        chart = build_input_chart(attendance, internal, assignment, previous)
        st.pyplot(chart, width="stretch")
        plt.close(chart)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Quick Evaluation</div>', unsafe_allow_html=True)

        valid, err = validate_input(attendance, internal, assignment, previous)
        if not valid:
            st.error(err)

        if st.button("Predict Result", type="primary", width="stretch", disabled=not valid or not models_ready):
            try:
                pred, confidence, prob_dict = predict_result(attendance, internal, assignment, previous)
                render_result_card(pred, confidence)
                st.info(explain_prediction(attendance, internal, assignment, previous, pred))

                st.session_state.prediction_history.append(
                    {
                        "attendance": attendance,
                        "internal": internal,
                        "assignment": assignment,
                        "previous": previous,
                        "result": pred,
                        "confidence": round(confidence * 100, 2),
                    }
                )

                st.write("Probability breakdown:")
                for cls, prob in prob_dict.items():
                    st.write(f"- {cls}: {prob * 100:.1f}%")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Dataset Preview":
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)

    data_path = get_data_path()
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        top_col1, top_col2, top_col3 = st.columns(3)
        with top_col1:
            render_kpi_card("Rows", len(df))
        with top_col2:
            render_kpi_card("Columns", len(df.columns))
        with top_col3:
            render_kpi_card("Target Column", "result")

        st.dataframe(df.head(100), width="stretch", height=360)

        if "result" in df.columns:
            chart = build_distribution_chart(df["result"], "Result Distribution in Dataset")
            st.pyplot(chart, width="stretch")
            plt.close(chart)
    else:
        st.warning("student_data.csv was not found.")

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Info & Retrain":
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Information</div>', unsafe_allow_html=True)
    st.write("Model: Logistic Regression")
    st.write("Pipeline: Load data -> Scale features -> Train -> Save")

    models_dir = get_models_path()
    model_path = os.path.join(models_dir, "trained_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        st.success("Saved model and scaler are available.")
    else:
        st.warning("Saved model files were not found. Retraining is recommended.")

    if st.button("Retrain Model", type="primary", width="stretch"):
        with st.spinner("Training model..."):
            try:
                result = train_all(save=True)
                st.success(f"Training completed. Accuracy: {result['accuracy'] * 100:.2f}%")
            except Exception as exc:
                st.error(str(exc))
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Prediction History":
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)

    if not st.session_state.prediction_history:
        st.info("No predictions yet. Run a prediction from the Predict page.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, width="stretch", height=360)

        if "result" in history_df.columns:
            chart = build_distribution_chart(history_df["result"], "Session Prediction Distribution")
            st.pyplot(chart, width="stretch")
            plt.close(chart)

    st.markdown("</div>", unsafe_allow_html=True)
