# Student Performance Prediction System

**BCA Final Year Project** – A simple, production-quality web application that predicts student performance (**Pass / Average / Fail**) using Machine Learning.

---

## Project Goal

Predict student result based on:

| Input | Description |
|-------|-------------|
| **Attendance %** | Percentage of classes attended |
| **Internal marks** | Internal exam score (out of 30) |
| **Assignment marks** | Assignment score (out of 20) |
| **Previous semester** | Last semester total (out of 100) |

**Output:** Predicted result (Pass / Average / Fail), confidence score, and a short explanation (viva-friendly).

---

## Tech Stack

- **Python 3.14.3
- **Pandas** – Data handling
- **NumPy** – Numerical operations
- **Scikit-learn** – Logistic Regression (main), Decision Tree (comparison)
- **Streamlit** – Web UI
- **Matplotlib** – Charts
- **CSV** – No database; dataset is a single CSV file

---

## Folder Structure

```
student-performance-project/
│
├── data/
│   ├── student_data.csv      # Dataset (attendance, internal, assignment, previous, result)
│   └── generate_dataset.py   # Script to regenerate synthetic data
│
├── models/
│   ├── trained_model.pkl     # Logistic Regression model (main)
│   ├── decision_tree_model.pkl
│   └── scaler.pkl            # StandardScaler for features
│
├── src/
│   ├── __init__.py
│   ├── train_model.py        # Training pipeline (LR + DT)
│   ├── predict.py            # Prediction + explain_prediction()
│   └── utils.py              # Load data, validate input, save/load model
│
├── tests/
│   └── test_validation.py    # Unit tests (validation, prediction)
│
├── app.py                    # Streamlit web app
├── requirements.txt
└── README.md
```

---

## How to Install

1. **Clone or download** the project and go to the project folder:
   ```bash
   cd student-performance-project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   # source venv/bin/activate   # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate dataset (if `data/student_data.csv` is missing):**
   ```bash
   cd data
   python generate_dataset.py
   cd ..
   ```

5. **Train the model (creates `models/trained_model.pkl` and `scaler.pkl`):**
   ```bash
   python -m src.train_model
   ```

---

## How to Run

**Start the web app:**

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## ML Steps (for Viva)

1. **Load data** – Read `student_data.csv` (columns: attendance, internal, assignment, previous, result).
2. **Clean data** – Drop rows with missing values.
3. **Train/Test split** – 80% train, 20% test, fixed random state.
4. **Scale features** – StandardScaler (mean=0, std=1) for Logistic Regression.
5. **Train model** – Logistic Regression (main), Decision Tree (comparison).
6. **Evaluate** – Accuracy, confusion matrix, classification report.
7. **Save model** – Pickle the model and scaler to `models/`.

---

## File Explanation

| File | Purpose |
|------|--------|
| **data/student_data.csv** | Sample dataset; 100+ rows of synthetic data. |
| **data/generate_dataset.py** | Generates CSV with rules: high → Pass, medium → Average, low → Fail. |
| **src/utils.py** | Paths, `load_dataset()`, `validate_input()`, `save_model()`, `load_model()`. |
| **src/train_model.py** | Full pipeline: load → split → scale → train LR & DT → evaluate → save. |
| **src/predict.py** | `predict_result()`, `explain_prediction()` for viva. |
| **app.py** | Streamlit UI: Predict, Dataset preview, Model info, Retrain, History, Download CSV. |

---

## App Features

- **Predict** – Sliders for 4 inputs → Predict → Result, confidence, explanation.
- **Bar chart** – Visual summary of entered marks.
- **Dataset preview** – First 100 rows and result distribution.
- **Model info & retrain** – View steps, upload CSV, retrain and see accuracy.
- **Prediction history** – List of predictions in session; download as CSV.
- **Download prediction** – Single prediction as CSV.

---

## Viva-Friendly: `explain_prediction()`

The function in `src/predict.py` returns a short reason for the prediction, for example:

- *"Low attendance caused predicted Fail."*
- *"Good internal marks supported predicted Pass."*

It uses simple rules on the four inputs so you can explain it clearly in the viva.

---

## Sample Predictions

| Attendance | Internal | Assignment | Previous | Predicted | Confidence |
|------------|----------|------------|----------|-----------|------------|
| 85         | 22       | 16         | 72       | Pass      | ~90%+      |
| 65         | 12       | 10         | 48       | Average   | ~70%+      |
| 55         | 8        | 6          | 35       | Fail      | ~80%+      |

*(Exact values depend on the trained model.)*

---

## Testing

**Input validation:** All inputs are checked (numeric, within allowed ranges).

**Run unit tests:**

```bash
python -m pytest tests/ -v
```

Or run the test file directly:

```bash
python tests/test_validation.py
```

---

## Optional Features (in App)

- **Upload CSV** – In “Model Info & Retrain”, upload your own dataset and retrain.
- **Retrain model** – Button to retrain and see new accuracy and confusion matrix.
- **Save prediction history** – Predictions are stored in session; you can download as CSV.

---

## Screenshots (What to Show)

1. **Home / Predict** – Sliders, bar chart, Predict button, result (Pass/Average/Fail), confidence, explanation.
2. **Dataset Preview** – Table and result distribution bar chart.
3. **Model Info & Retrain** – Accuracy after retrain, confusion matrix, classification report.
4. **Prediction History** – Table and “Download history as CSV”.

---

## University Requirements Met

- **Simple and explainable** – No deep learning; Logistic Regression and Decision Tree only.
- **No heavy frameworks** – Streamlit + CSV only.
- **Clear comments** – e.g. `# Step 1: Load dataset`, `# Step 2: Train model`.
- **Viva-ready** – `explain_prediction()` and clear ML steps in README.

---


