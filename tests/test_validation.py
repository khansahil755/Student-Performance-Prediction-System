"""
Unit tests for Student Performance Prediction System.
Tests: input validation, prediction output shape, explain_prediction.
"""

import sys
import os

# Add project root so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.utils import validate_input, load_dataset, get_data_path
from src.predict import predict_result, explain_prediction, get_model_and_scaler


class TestValidateInput:
    """Test input validation for prediction."""

    def test_valid_input(self):
        valid, msg = validate_input(80, 20, 15, 70)
        assert valid is True
        assert msg == ""

    def test_attendance_out_of_range_high(self):
        valid, msg = validate_input(101, 20, 15, 70)
        assert valid is False
        assert "Attendance" in msg

    def test_attendance_out_of_range_low(self):
        valid, msg = validate_input(-5, 20, 15, 70)
        assert valid is False

    def test_internal_out_of_range(self):
        valid, msg = validate_input(80, 35, 15, 70)
        assert valid is False
        assert "Internal" in msg

    def test_assignment_out_of_range(self):
        valid, msg = validate_input(80, 20, 25, 70)
        assert valid is False
        assert "Assignment" in msg

    def test_previous_out_of_range(self):
        valid, msg = validate_input(80, 20, 15, 105)
        assert valid is False
        assert "Previous" in msg

    def test_non_numeric(self):
        valid, msg = validate_input("abc", 20, 15, 70)
        assert valid is False
        assert "numbers" in msg.lower()


class TestPredict:
    """Test prediction and explanation (requires trained model)."""

    @pytest.fixture
    def model_and_scaler(self):
        return get_model_and_scaler()

    def test_predict_returns_three_values(self, model_and_scaler):
        model, scaler = model_and_scaler
        if model is None:
            pytest.skip("No trained model found. Run: python -m src.train_model")
        pred, confidence, prob_dict = predict_result(75, 20, 14, 65, model=model, scaler=scaler)
        assert pred in ("Pass", "Average", "Fail")
        assert 0 <= confidence <= 1
        assert isinstance(prob_dict, dict)
        assert any(label in prob_dict for label in ("Pass", "Average", "Fail"))
        assert abs(sum(float(v) for v in prob_dict.values()) - 1.0) < 0.01

    def test_explain_prediction_returns_string(self):
        explanation = explain_prediction(55, 10, 8, 40, "Fail")
        assert isinstance(explanation, str)
        assert len(explanation) > 5
        assert "fail" in explanation.lower() or "predict" in explanation.lower()


class TestLoadDataset:
    """Test dataset loading."""

    def test_load_dataset_if_file_exists(self):
        path = get_data_path()
        if not os.path.exists(path):
            pytest.skip("student_data.csv not found")
        X, y, df = load_dataset()
        assert X is not None
        assert y is not None
        assert df is not None
        assert list(X.columns) == ["attendance", "internal", "assignment", "previous"]
        assert "result" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
