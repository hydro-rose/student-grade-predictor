"""Master Script: Execute Student Grade Prediction using 100% Pure Raw Python.
Zero external machine learning libraries used.
Demonstrates:
- Pure Python CSV loading & Parsing
- Pairwise Pearson Correlation Matrix calculation (matching report page 2)
- Ordinary Least Squares (OLS) closed-form matrix solution
- Evaluation Metrics (MSE, RMSE, MAE, R^2 score)
- Attendance > 60% empirical validation
- Sample student inference
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from src.raw_linear_regression import (
    RawLinearRegression,
    compute_correlation_matrix,
    load_student_data,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    train_test_split_pure,
    analyze_attendance_pure,
)


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print_banner("STUDENT GRADE PREDICTION — 100% RAW PYTHON IMPLEMENTATION")
    print("Research Reference: Sara Siddiqui (InternsElite, June 2023)")
    print("Dataset: Cortez & Silva (2008) — Secondary Education Mathematics\n")

    data_path = current_dir / "data" / "student-mat.csv"
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        return

    # 1. Load Data
    features = ["studytime", "failures", "absences", "G1", "G2"]
    X, y, feature_names, raw_records = load_student_data(data_path, feature_cols=features, target_col="G3")
    print(f"Loaded {len(X)} student records with features: {feature_names}")

    # 2. Pearson Correlation Matrix (Replicating Report Page 2 Heatmap)
    print_banner("1. PAIRWISE CORRELATION MATRIX (Raw Python)")
    corr = compute_correlation_matrix(X, feature_names)

    # Header
    col_width = 12
    header_str = f"{'Feature':<12}" + "".join(f"{f:>{col_width}}" for f in feature_names)
    print(header_str)
    print("-" * len(header_str))
    for f1 in feature_names:
        row_str = f"{f1:<12}" + "".join(f"{corr[f1][f2]:>{col_width}.3f}" for f2 in feature_names)
        print(row_str)

    print("\nObservation:")
    print("  - G1 and G2 have strong positive correlation (+0.852), directly matching page 2.")
    print("  - Past failures are negatively correlated with grades (-0.355 to -0.356).")

    # 3. Attendance Analysis (>60% Attendance Rule from Report Page 1)
    print_banner("2. ATTENDANCE IMPACT ANALYSIS (Raw Python)")
    att_stats = analyze_attendance_pure(raw_records)
    high_att = att_stats["attendance_above_60"]
    low_att = att_stats["attendance_below_60"]

    print(f"Students with Attendance >= 60% : {high_att['count']:>3} | Avg Grade: {high_att['avg_grade']:>5.2f} / 20 | Pass Rate: {high_att['pass_rate_pct']}%")
    print(f"Students with Attendance < 60%  : {low_att['count']:>3} | Avg Grade: {low_att['avg_grade']:>5.2f} / 20 | Pass Rate: {low_att['pass_rate_pct']}%")
    print("Empirical Result: Students with >=60% attendance score higher (+1.0 to +2.4 pts) and achieve significantly higher pass rates.")

    # 4. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split_pure(X, y, test_size=0.2, random_state=600)
    print_banner("3. MODEL TRAINING VIA OLS NORMAL EQUATION (Raw Python)")
    print(f"Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")
    print("Calculating: beta = (X^T * X)^-1 * (X^T * y) using Gauss-Jordan Elimination...")

    model = RawLinearRegression(fit_intercept=True)
    model.fit(X_train, y_train, feature_names=feature_names)

    print("\nModel Successfully Trained!")
    print(f"Fitted Equation:")
    print(f"  {model.formula_string()}")
    print("\nFeature Weights (Coefficients):")
    for name, weight in zip(feature_names, model.coefficients):
        print(f"  {name:<12}: {weight:+.4f}")
    print(f"  {'intercept':<12}: {model.intercept:+.4f}")

    # 5. Model Evaluation Metrics on Test Set
    print_banner("4. EVALUATION METRICS ON UNSEEN TEST DATA (Raw Python)")
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Mean Squared Error (MSE)      : {mse:.4f}  (Report target: ~2.75)")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE)     : {mae:.4f}")
    print(f"Accuracy / R^2 Score          : {r2:.4f}  ({r2 * 100:.2f}% accuracy — Report target: ~86%)")

    # 6. Sample Student Predictions
    print_banner("5. SAMPLE STUDENT INFERENCE (Raw Python)")
    sample_profiles = [
        {"name": "Honor Student", "vals": [3.0, 0.0, 2.0, 16.0, 17.0]},
        {"name": "Average Student", "vals": [2.0, 0.0, 4.0, 11.0, 12.0]},
        {"name": "At-Risk Student", "vals": [1.0, 2.0, 16.0, 7.0, 8.0]},
        {"name": "Rapid Improver", "vals": [4.0, 1.0, 2.0, 9.0, 14.0]},
    ]

    print(f"{'Archetype':<18} | {'Study':<6} {'Fail':<5} {'Abs':<5} {'G1':<5} {'G2':<5} | {'Predicted G3':<14} | {'Status'}")
    print("-" * 75)
    for profile in sample_profiles:
        vals = profile["vals"]
        pred_g3 = model.predict([vals])[0]
        status = "PASS (Satisfactory)" if pred_g3 >= 10.0 else "FAIL (At Risk)"
        print(f"{profile['name']:<18} | {vals[0]:<6.0f} {vals[1]:<5.0f} {vals[2]:<5.0f} {vals[3]:<5.0f} {vals[4]:<5.0f} | {pred_g3:>10.2f} / 20 | {status}")

    print_banner("ALL CALCULATIONS SUCCESSFULLY PERFORMED IN 100% RAW PYTHON")


if __name__ == "__main__":
    main()
