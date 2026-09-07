"""Evaluation module for Student Grade Prediction models.
Calculates standard regression & classification metrics, cross-validation scores,
bootstrapped confidence intervals, and residual analytics.
"""

from typing import Dict, Any, Optional
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
from sklearn.model_selection import cross_val_score, KFold


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple:
    """Compute empirical bootstrap confidence interval for a given metric."""
    rng = np.random.RandomState(random_state)
    bootstrapped_scores = []
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2 and metric_fn == r2_score:
            continue
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    if not bootstrapped_scores:
        return 0.0, 0.0

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(bootstrapped_scores, alpha * 100))
    upper = float(np.percentile(bootstrapped_scores, (1.0 - alpha) * 100))
    return round(lower, 4), round(upper, 4)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Linear Regression",
    compute_ci: bool = True,
) -> Dict[str, Any]:
    """Calculate comprehensive regression evaluation metrics.
    
    Includes:
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R2 Score (Accuracy score as termed in report)
    - 95% Bootstrap Confidence Intervals
    - Residual statistics
    """
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    residuals = y_true - y_pred
    mean_residual = float(np.mean(residuals))
    std_residual = float(np.std(residuals))

    res = {
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

    if compute_ci and len(y_true) > 10:
        mse_ci = compute_bootstrap_ci(y_true, y_pred, mean_squared_error)
        r2_ci = compute_bootstrap_ci(y_true, y_pred, r2_score)
        res["mse_95_ci"] = list(mse_ci)
        res["r2_95_ci"] = list(r2_ci)

    return res


def evaluate_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Run k-fold cross-validation on regression model."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    neg_mse_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")

    return {
        "cv_r2_mean": round(float(np.mean(r2_scores)), 4),
        "cv_r2_std": round(float(np.std(r2_scores)), 4),
        "cv_mse_mean": round(float(np.mean(-neg_mse_scores)), 4),
        "cv_mse_std": round(float(np.std(-neg_mse_scores)), 4),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
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
            if y_proba.ndim == 2:
                proba_pos = y_proba[:, 1]
            else:
                proba_pos = y_proba
            roc_auc = float(roc_auc_score(y_true, proba_pos))
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
