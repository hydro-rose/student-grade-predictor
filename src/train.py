"""Model Training and Experimentation Pipeline.
Trains Linear Regression, Ridge, Random Forest, and Classification models.
Saves serialized models and metadata for production serving.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    load_raw_data,
    add_derived_features,
    CORE_FEATURES,
    TARGET_FEATURE,
)
from src.evaluate import evaluate_regression, evaluate_classification


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_models(
    subject: str = "mat",
    test_size: float = 0.2,
    random_state: int = 28,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute complete training pipeline and persist best models.
    
    Args:
        subject: Subject data to train on ('mat' for Math, 'por' for Portuguese).
        test_size: Train/test split ratio.
        random_state: Seed ensuring reproducibility.
        
    Returns:
        Tuple of (regression_results, classification_results)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingest and prepare data
    df = load_raw_data(subject=subject)
    df = add_derived_features(df)

    X = df[CORE_FEATURES].copy()
    y_reg = df[TARGET_FEATURE].values
    y_clf = df["passed"].values

    # 2. Strict ML featurization ordering: split BEFORE fitting scaler
    (
        X_train,
        X_test,
        y_train_reg,
        y_test_reg,
        y_train_clf,
        y_test_clf,
    ) = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=random_state
    )

    # 3. Fit standard scaler on training data only
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Save scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # 4. Train Primary Model: Linear Regression (Direct feature space as in report)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_reg)
    y_pred_lr = lr_model.predict(X_test)
    lr_eval = evaluate_regression(y_test_reg, y_pred_lr, model_name="Linear Regression")

    # K-Fold Cross Validation for robustness
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    lr_cv_scores = cross_val_score(lr_model, X, y_reg, cv=kfold, scoring="r2")
    lr_eval["cv_r2_mean"] = round(float(np.mean(lr_cv_scores)), 4)
    lr_eval["cv_r2_std"] = round(float(np.std(lr_cv_scores)), 4)

    # Save primary regression model
    joblib.dump(lr_model, MODELS_DIR / "linear_regression_model.joblib")

    # 5. Train Benchmark Regressor: Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train_reg)
    y_pred_ridge = ridge_model.predict(X_test)
    ridge_eval = evaluate_regression(y_test_reg, y_pred_ridge, model_name="Ridge Regression")

    # 6. Train Non-Linear Benchmark: Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=6)
    rf_reg.fit(X_train, y_train_reg)
    y_pred_rf = rf_reg.predict(X_test)
    rf_eval = evaluate_regression(y_test_reg, y_pred_rf, model_name="Random Forest Regressor")
    joblib.dump(rf_reg, MODELS_DIR / "rf_regressor.joblib")

    # 7. Train Early-Intervention Classifier: Random Forest Classifier (Pass/Fail)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=5)
    rf_clf.fit(X_train, y_train_clf)
    y_pred_clf = rf_clf.predict(X_test)
    y_proba_clf = rf_clf.predict_proba(X_test)[:, 1]
    clf_eval = evaluate_classification(y_test_clf, y_pred_clf, y_proba_clf, model_name="Pass/Fail Classifier")
    joblib.dump(rf_clf, MODELS_DIR / "rf_classifier.joblib")

    # 8. Compute Correlation Matrix matching page 2 of research report
    corr_features = ["studytime", "failures", "absences", "G1", "G2"]
    corr_matrix = df[corr_features].corr().round(3).to_dict()

    # 9. Attendance vs Performance breakdown
    # Report finding: Most students with >60% attendance achieve better academic grades
    attendance_stats = {
        "above_60_attendance": {
            "count": int((df["attendance_above_60"] == 1).sum()),
            "avg_grade": round(float(df[df["attendance_above_60"] == 1]["G3"].mean()), 2),
            "pass_rate": round(float((df[df["attendance_above_60"] == 1]["passed"] == 1).mean() * 100), 1),
        },
        "below_60_attendance": {
            "count": int((df["attendance_above_60"] == 0).sum()),
            "avg_grade": round(float(df[df["attendance_above_60"] == 0]["G3"].mean()), 2),
            "pass_rate": round(float((df[df["attendance_above_60"] == 0]["passed"] == 1).mean() * 100), 1),
        },
    }

    # Model coefficients
    coefficients = {feat: round(float(coef), 4) for feat, coef in zip(CORE_FEATURES, lr_model.coef_)}
    coefficients["intercept"] = round(float(lr_model.intercept_), 4)

    # 10. Persist metadata
    metadata = {
        "subject": subject,
        "dataset_size": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": CORE_FEATURES,
        "target": TARGET_FEATURE,
        "linear_regression": lr_eval,
        "ridge_regression": ridge_eval,
        "rf_regressor": rf_eval,
        "classifier": clf_eval,
        "coefficients": coefficients,
        "correlation_matrix": corr_matrix,
        "attendance_insights": attendance_stats,
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("--- Training Pipeline Successfully Completed ---")
    print(f"Primary Linear Regression MSE: {lr_eval['mse']} | R2: {lr_eval['r2_score']} ({lr_eval['r2_percentage']}%)")
    print(f"Ridge Regression MSE: {ridge_eval['mse']} | R2: {ridge_eval['r2_score']}")
    print(f"Random Forest Regressor MSE: {rf_eval['mse']} | R2: {rf_eval['r2_score']}")
    print(f"Pass/Fail Classification Accuracy: {clf_eval['accuracy_pct']}% | F1: {clf_eval['f1_score']}")
    print(f"Coefficients: {coefficients}")

    return metadata, clf_eval


if __name__ == "__main__":
    train_models()
