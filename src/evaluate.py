"""Evaluation module for Student Grade Prediction models.
Calculates standard regression & classification metrics, cross-validation scores, and residuals.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Linear Regression",
) -> Dict[str, Any]:
    """Calculate comprehensive regression evaluation metrics.
    
    Includes:
    - MSE (Mean Squared Error) - exactly matching report metric
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R2 Score (Accuracy score as termed in report)
    - Explained Variance
    - Max Error
    """
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    residuals = y_true - y_pred
    mean_residual = float(np.mean(residuals))
    std_residual = float(np.std(residuals))

    return {
        "model_name": model_name,
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2_score": round(r2, 4),
        "r2_percentage": round(r2 * 100, 2),
        "mean_residual": round(mean_residual, 4),
        "std_residual": round(std_residual, 4),
        "raw_mse": mse,
        "raw_r2": r2,
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    model_name: str = "Classifier",
) -> Dict[str, Any]:
    """Calculate classification metrics for pass/fail early risk intervention."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            roc_auc = None

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
    }
