"""Data Loader and Preprocessing Module for Student Grade Prediction.
Handles loading, validation, and feature preparation of the UCI Student Performance dataset.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_MATH_PATH = Path(__file__).resolve().parent.parent / "data" / "student-mat.csv"
DEFAULT_POR_PATH = Path(__file__).resolve().parent.parent / "data" / "student-por.csv"

# Core numerical features identified in the research report
CORE_FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
TARGET_FEATURE = "G3"


def load_raw_data(
    file_path: Optional[Union[str, Path]] = None,
    subject: str = "mat",
) -> pd.DataFrame:
    """Load raw dataset with semicolon delimiter and standard schema validation.
    
    Args:
        file_path: Optional path to CSV file.
        subject: Subject type ('mat' or 'por') if file_path is None.
        
    Returns:
        pd.DataFrame containing student data.
    """
    if file_path is None:
        target_path = DEFAULT_MATH_PATH if subject == "mat" else DEFAULT_POR_PATH
    else:
        target_path = Path(file_path)

    if not target_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {target_path}")

    df = pd.read_csv(target_path, sep=";")
    
    # Strip whitespace/quotes from string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().str.strip('"')

    # Ensure numeric columns are properly typed
    numeric_cols = [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2", "G3"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pedagogical and behavioral features referenced in the study.
    
    Derived Features:
    - passed: Boolean flag for G3 >= 10 (Passing threshold in Portuguese secondary education).
    - grade_group: Categorical representation (A: 16-20, B: 14-15, C: 12-13, D: 10-11, F: 0-9).
    - attendance_rate: Approximate attendance % assuming ~90 school sessions.
    - high_attendance: Flag indicating attendance > 60% as highlighted in research report.
    - academic_momentum: G2 - G1 difference representing grade trajectory.
    """
    data = df.copy()

    # Pass / Fail target for classification
    data["passed"] = (data[TARGET_FEATURE] >= 10).astype(int)

    # Grade brackets
    def assign_grade_bracket(g: float) -> str:
        if g >= 16:
            return "A (Excellent)"
        elif g >= 14:
            return "B (Good)"
        elif g >= 12:
            return "C (Satisfactory)"
        elif g >= 10:
            return "D (Sufficient)"
        else:
            return "F (Fail)"

    data["grade_group"] = data[TARGET_FEATURE].apply(assign_grade_bracket)

    # Attendance calculations (Max Portuguese school days in term ~ 90)
    # Estimated attendance %: max(0, 100 - (absences / 90) * 100)
    data["attendance_pct"] = np.clip(100.0 - (data["absences"] / 90.0) * 100.0, 0.0, 100.0)
    data["attendance_above_60"] = (data["attendance_pct"] >= 60.0).astype(int)

    # Academic trajectory
    data["grade_trend"] = data["G2"] - data["G1"]

    return data


def prepare_train_test_data(
    df: Optional[pd.DataFrame] = None,
    features: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and test sets strictly BEFORE featurization/scaling.
    
    Adheres strictly to ml-best-practices:
    Split must occur prior to fitting scalers or encodings to prevent data leakage.
    """
    if df is None:
        df = load_raw_data(subject="mat")

    if features is None:
        features = CORE_FEATURES

    X = df[features].copy()
    y = df[TARGET_FEATURE].copy()

    # Check for missing values
    if X.isnull().any().any():
        X = X.fillna(X.median())
    if y.isnull().any():
        valid_idx = ~y.isnull()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """Compute comprehensive dataset statistics for UI and diagnostics."""
    return {
        "total_students": len(df),
        "mean_final_grade": round(float(df["G3"].mean()), 2),
        "median_final_grade": round(float(df["G3"].median()), 2),
        "std_final_grade": round(float(df["G3"].std()), 2),
        "pass_rate": round(float((df["G3"] >= 10).mean() * 100), 1),
        "mean_absences": round(float(df["absences"].mean()), 1),
        "mean_studytime_hrs": round(float(df["studytime"].mean()), 1),
        "high_attendance_pct": round(float((df["absences"] <= 36).mean() * 100), 1),
    }
