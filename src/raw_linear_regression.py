"""Raw Python Implementation of Linear Regression & Statistical Analysis.
Zero external ML dependencies (no scikit-learn, no numpy, no pandas).
Pure standard library math, matrix operations, and algorithms from first principles.
Replicating the research by Sara Siddiqui (June 2023) on the Cortez & Silva dataset.
"""

import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# =============================================================================
# 1. PURE PYTHON LINEAR ALGEBRA & MATRIX ENGINE
# =============================================================================

def transpose(matrix: List[List[float]]) -> List[List[float]]:
    """Compute the transpose of a 2D matrix: A^T."""
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiply two 2D matrices: C = A x B."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError(f"Matrix dimension mismatch: ({rows_A}x{cols_A}) and ({rows_B}x{cols_B})")

    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            a_ik = A[i][k]
            for j in range(cols_B):
                C[i][j] += a_ik * B[k][j]
    return C


def matvec_mul(A: List[List[float]], v: List[float]) -> List[float]:
    """Multiply matrix A by vector v: result = A x v."""
    return [sum(row[c] * v[c] for c in range(len(v))) for row in A]


def invert_matrix_gauss_jordan(A: List[List[float]]) -> List[List[float]]:
    """Compute matrix inverse A^-1 using Gauss-Jordan elimination with partial pivoting.
    Pure standard library implementation.
    """
    n = len(A)
    # Check square
    for row in A:
        if len(row) != n:
            raise ValueError("Matrix must be square to invert.")

    # Create augmented matrix [A | I]
    augmented = [
        [float(A[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]

    # Gauss-Jordan forward and backward elimination
    for col in range(n):
        # 1. Pivot selection
        max_row = col
        max_val = abs(augmented[col][col])
        for r in range(col + 1, n):
            if abs(augmented[r][col]) > max_val:
                max_val = abs(augmented[r][col])
                max_row = r

        if max_val < 1e-12:
            raise ValueError(f"Matrix is singular or near-singular at column {col}.")

        # Swap rows if necessary
        if max_row != col:
            augmented[col], augmented[max_row] = augmented[max_row], augmented[col]

        # 2. Normalize pivot row
        pivot = augmented[col][col]
        for c in range(2 * n):
            augmented[col][c] /= pivot

        # 3. Eliminate other rows
        for r in range(n):
            if r != col:
                factor = augmented[r][col]
                if abs(factor) > 1e-15:
                    for c in range(col, 2 * n):
                        augmented[r][c] -= factor * augmented[col][c]

    # Extract right side inverse matrix
    inverse = [[augmented[i][j + n] for j in range(n)] for i in range(n)]
    return inverse


# =============================================================================
# 2. STATISTICAL EVALUATION METRICS (PURE PYTHON)
# =============================================================================

def mean(values: List[float]) -> float:
    """Calculate the arithmetic mean."""
    return sum(values) / len(values) if values else 0.0


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Squared Error (MSE): (1/n) * sum((y_i - y_hat_i)^2)."""
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must match.")
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """Root Mean Squared Error (RMSE): sqrt(MSE)."""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Absolute Error (MAE): (1/n) * sum(|y_i - y_hat_i|)."""
    return sum(abs(yt - yp) ** 2 ** 0.5 for yt, yp in zip(y_true, y_pred)) / len(y_true)


def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    """Coefficient of Determination (R^2 Score):
    R^2 = 1 - (SS_res / SS_tot)
    """
    y_mean = mean(y_true)
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient r between two numeric vectors."""
    n = len(x)
    if n != len(y) or n == 0:
        raise ValueError("Vectors must be non-empty and of identical length.")
    mx, my = mean(x), mean(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    var_x = sum((xi - mx) ** 2 for xi in x)
    var_y = sum((yi - my) ** 2 for yi in y)
    denom = math.sqrt(var_x * var_y)
    return cov / denom if denom != 0.0 else 0.0


# =============================================================================
# 3. RAW PYTHON LINEAR REGRESSION MODEL (OLS & GRADIENT DESCENT)
# =============================================================================

class RawLinearRegression:
    """Multivariate Ordinary Least Squares (OLS) Linear Regression in Raw Python.
    Estimates optimal beta vector: beta = (X^T * X)^-1 * X^T * y
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coefficients: List[float] = []
        self.intercept: float = 0.0
        self.feature_names: List[str] = []

    def fit(
        self,
        X: List[List[float]],
        y: List[float],
        feature_names: Optional[List[str]] = None,
    ) -> "RawLinearRegression":
        """Fit the linear regression model using closed-form OLS normal equation."""
        n_samples = len(X)
        if n_samples == 0 or len(y) != n_samples:
            raise ValueError("Invalid training dimensions.")

        n_features = len(X[0])
        self.feature_names = (
            feature_names if feature_names else [f"x{i}" for i in range(n_features)]
        )

        # 1. Build design matrix with bias column of 1s if fit_intercept
        if self.fit_intercept:
            X_design = [[1.0] + list(row) for row in X]
        else:
            X_design = [[float(val) for val in row] for row in X]

        # 2. X_T = Transpose(X_design)
        X_T = transpose(X_design)

        # 3. Gram matrix: G = X_T * X_design
        G = matmul(X_T, X_design)

        # 4. Invert Gram matrix: G_inv = (X_T * X)^-1
        # Add tiny L2 ridge regularization (1e-7) for numerical stability
        for i in range(len(G)):
            G[i][i] += 1e-7
        G_inv = invert_matrix_gauss_jordan(G)

        # 5. X_T * y
        X_T_y = matvec_mul(X_T, y)

        # 6. beta = G_inv * (X_T * y)
        beta = matvec_mul(G_inv, X_T_y)

        if self.fit_intercept:
            self.intercept = beta[0]
            self.coefficients = beta[1:]
        else:
            self.intercept = 0.0
            self.coefficients = beta

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        """Generate predictions for feature matrix X."""
        predictions = []
        for row in X:
            val = self.intercept + sum(w * x for w, x in zip(self.coefficients, row))
            predictions.append(val)
        return predictions

    def score(self, X: List[List[float]], y: List[float]) -> float:
        """Compute R^2 score on test set."""
        preds = self.predict(X)
        return r2_score(y, preds)

    def formula_string(self) -> str:
        """Return the algebraic formula equation string."""
        terms = [f"{self.intercept:+.4f}"]
        for name, coef in zip(self.feature_names, self.coefficients):
            terms.append(f"{coef:+.4f} * {name}")
        return "G3 = " + " ".join(terms)


class RawGradientDescentRegressor:
    """Linear Regression trained via Batch Gradient Descent in Pure Python."""

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1500):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.loss_history: List[float] = []

    def fit(self, X: List[List[float]], y: List[float]) -> "RawGradientDescentRegressor":
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            # Compute predictions
            y_pred = [
                self.bias + sum(w * x for w, x in zip(self.weights, row)) for row in X
            ]

            # Compute gradients
            errors = [yp - yt for yp, yt in zip(y_pred, y)]
            grad_b = (2.0 / n_samples) * sum(errors)
            grad_w = [0.0 for _ in range(n_features)]
            for j in range(n_features):
                grad_w[j] = (2.0 / n_samples) * sum(
                    errors[i] * X[i][j] for i in range(n_samples)
                )

            # Update weights
            self.bias -= self.lr * grad_b
            for j in range(n_features):
                self.weights[j] -= self.lr * grad_w[j]

            # Record MSE
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                mse = sum(e ** 2 for e in errors) / n_samples
                self.loss_history.append(mse)

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        return [self.bias + sum(w * x for w, x in zip(self.weights, row)) for row in X]


# =============================================================================
# 4. RAW CSV DATA PIPELINE & PREPROCESSING (ZERO THIRD PARTY)
# =============================================================================

def load_student_data(
    csv_path: Union[str, Path],
    feature_cols: Optional[List[str]] = None,
    target_col: str = "G3",
) -> Tuple[List[List[float]], List[float], List[str], List[Dict[str, str]]]:
    """Parse semicolon-delimited CSV using only standard library `csv` module."""
    if feature_cols is None:
        feature_cols = ["studytime", "failures", "absences", "G1", "G2"]

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw_records = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            raw_records.append(row)

    X = []
    y = []
    for record in raw_records:
        x_row = [float(record[col].strip('"')) for col in feature_cols]
        y_val = float(record[target_col].strip('"'))
        X.append(x_row)
        y.append(y_val)

    return X, y, feature_cols, raw_records


def train_test_split_pure(
    X: List[List[float]],
    y: List[float],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    """Reproducible pseudo-random train/test split without scikit-learn."""
    n = len(X)
    indices = list(range(n))
    rng = random.Random(random_state)
    rng.shuffle(indices)

    split_idx = int(n * (1.0 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test


def compute_correlation_matrix(
    X: List[List[float]], feature_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Calculate the pairwise Pearson correlation matrix for features."""
    cols = transpose(X)
    corr_matrix = {}
    for i, name1 in enumerate(feature_names):
        corr_matrix[name1] = {}
        for j, name2 in enumerate(feature_names):
            r = pearson_correlation(cols[i], cols[j])
            corr_matrix[name1][name2] = round(r, 3)
    return corr_matrix


def analyze_attendance_pure(raw_records: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    """Replicate the research finding: Attendance > 60% yields better performance."""
    total_sessions = 90.0
    above_60 = []
    below_60 = []

    for r in raw_records:
        absences = float(r["absences"].strip('"'))
        g3 = float(r["G3"].strip('"'))
        att_rate = ((total_sessions - absences) / total_sessions) * 100.0
        if att_rate >= 60.0:
            above_60.append(g3)
        else:
            below_60.append(g3)

    return {
        "attendance_above_60": {
            "count": len(above_60),
            "avg_grade": round(mean(above_60), 2),
            "pass_rate_pct": round(sum(1 for g in above_60 if g >= 10.0) / len(above_60) * 100, 1),
        },
        "attendance_below_60": {
            "count": len(below_60),
            "avg_grade": round(mean(below_60), 2) if below_60 else 0.0,
            "pass_rate_pct": round(sum(1 for g in below_60 if g >= 10.0) / len(below_60) * 100, 1) if below_60 else 0.0,
        },
    }
