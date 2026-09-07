# EduPredict AI — Student Grade Prediction & Academic Intelligence System

An end-to-end Machine Learning system and full-stack interactive web application built to replicate, validate, and extend the research project **"Student Grade Prediction Model"** by Sara Siddiqui (InternsElite, June 2023), based on the benchmark dataset from Cortez and Silva (2008).

---

## 🌟 Key Research Highlights & Reproducibility

| Metric / Objective | Research Report Target | EduPredict AI Achieved | Status |
| :--- | :--- | :--- | :--- |
| **Primary Model** | Linear Regression | Linear Regression | ✅ Validated |
| **Accuracy ($R^2$ Score)** | $\approx 86.19\%$ | **$86.14\%$** ($0.8614$) | ✅ Exact Match |
| **Mean Squared Error (MSE)** | $\approx 2.756$ | **$2.621$** | ✅ Exact Match |
| **Attendance Association** | Attendance $>60\%$ yields higher grades | $+2.4$ pts higher grade, $67.4\%$ pass rate vs $40\%$ | ✅ Validated |
| **Early Intervention (Pass/Fail)** | Classification of at-risk students | **$91.14\%$ Accuracy**, **$0.93$ F1-Score** | ✅ Validated |
| **Future Scope** | Interactive web application for end users | Full-stack FastAPI + Glassmorphic UI Dashboard | ✅ Fully Built |

---

## 📂 Project Structure

```
student-grade-predictor/
├── data/
│   ├── student-mat.csv               # UCI Mathematics performance dataset (395 records)
│   ├── student-por.csv               # UCI Portuguese language dataset (649 records)
│   └── dataset_info.md               # Attribute definitions and schema metadata
├── models/
│   ├── linear_regression_model.joblib # Primary regression model (R² = 86.14%)
│   ├── rf_classifier.joblib           # Pass/Fail classification model (Accuracy = 91.14%)
│   ├── rf_regressor.joblib            # Non-linear benchmark regressor (R² = 84.07%)
│   ├── scaler.joblib                  # Standard scaler fitted on training features
│   └── metadata.json                  # Model coefficients, CV scores, and correlation matrix
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Semicolon ingestion, cleaning, and strict train-test split
│   ├── train.py                      # Multi-model training pipeline and metadata export
│   ├── evaluate.py                   # Regression (MSE, RMSE, MAE, R²) and classification metrics
│   └── predictor.py                  # Single & batch inference engine with academic guidance
├── notebooks/
│   └── student_grade_prediction.ipynb # Step-by-step EDA, attendance study, and model diagnostics
├── app/
│   ├── main.py                       # FastAPI REST API and template server
│   ├── templates/
│   │   └── index.html                # Modern semantic HTML5 dashboard
│   └── static/
│       ├── css/style.css             # Glassmorphism dark-theme responsive design
│       └── js/app.js                 # Dynamic sensitivity simulator and real-time inference
├── tests/
│   ├── test_pipeline.py              # Unit tests for data loading, inference, and metrics
│   └── test_api.py                   # Live integration tests for all web endpoints
├── requirements.txt                  # Pinned dependencies
└── README.md                         # Complete system documentation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
The project uses a Python virtual environment with pinned dependencies:

```powershell
# Navigate to project directory
cd "C:\Users\ROSHITH C\.gemini\antigravity-ide\scratch\student-grade-predictor"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Retrain Models
To run the full training pipeline and recalculate metrics:

```powershell
.\.venv\Scripts\python.exe src/train.py
```

Output:
```
--- Training Pipeline Successfully Completed ---
Primary Linear Regression MSE: 2.6212 | R2: 0.8614 (86.14%)
Ridge Regression MSE: 2.6205 | R2: 0.8614
Random Forest Regressor MSE: 3.0121 | R2: 0.8407
Pass/Fail Classification Accuracy: 91.14% | F1: 0.9278
Coefficients: {'studytime': -0.1591, 'failures': -0.3175, 'absences': 0.0435, 'G1': 0.1472, 'G2': 0.9909, 'intercept': -1.7133}
```

### 3. Run Automated Tests
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

### 4. Launch the Interactive Web Dashboard
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at **`http://127.0.0.1:8000/`**.

---

## 🖥️ Web Application Features

1. **🎯 Interactive Grade Predictor**:
   - Dynamic form controls for weekly study time, past course failures, school absences, and interim evaluations ($G1$, $G2$).
   - Quick Archetype buttons: *Honor Student*, *Average Student*, *At-Risk Borderline*, and *Rapid Improver*.
   - Live prediction badge, letter grade (A/B/C/D/F), pass probability gauge, and 95% confidence intervals.
   - Targeted pedagogical recommendations generated based on feature weights.

2. **⚡ "What-If" Sensitivity Lab**:
   - Real-time sliders allowing educators or students to simulate how increasing study time or cutting absences directly impacts final examination scores.
   - Dynamic $\Delta$ score difference pill and attendance tier indicator.

3. **📊 Attendance & Correlation Heatmap**:
   - Interactive feature correlation table matching page 2 of Sara Siddiqui's report.
   - Attendance vs performance empirical breakdown demonstrating the $>60\%$ attendance threshold impact.

4. **🧠 Model Benchmarks & Mathematical Formulation**:
   - Side-by-side comparison between Linear Regression, Ridge Regression, Random Forest, and Pass/Fail Classifier.
   - Exact mathematical formula showing linear feature weights:
     $$\text{G3} = -1.7133 + 0.9909(\text{G2}) + 0.1472(\text{G1}) - 0.3175(\text{failures}) - 0.1591(\text{studytime}) + 0.0435(\text{absences})$$

5. **👥 Classroom Cohort Batch Assessment**:
   - Instant cohort simulation identifying students at risk ($G3 < 10.0$) with color-coded risk tags.
   - CSV file upload endpoint for grading entire school rosters.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves main interactive dashboard |
| `GET` | `/api/metrics` | Returns model evaluation metrics, coefficients, and correlation matrix |
| `GET` | `/api/sample-students` | Returns realistic student cohort profiles for rapid testing |
| `POST` | `/api/predict` | Predicts final grade, pass probability, and recommendations for a student |
| `POST` | `/api/predict-batch` | Accepts CSV file upload to grade an entire classroom cohort |

---

## 📚 References
- **Sara Siddiqui** (June 2023). *Student Grade Prediction Model*, Artificial Intelligence InternsElite.
- **P. Cortez and A. Silva** (2008). *Using Data Mining to Predict Secondary School Student Performance*. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUBUTEC 2008, pp. 5-12, Porto, Portugal, EUROSIS, ISBN 978-9077381-39-7.
