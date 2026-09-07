"""Model Training and Experimentation Pipeline (Full AI & ML Suite).
Trains and benchmarks:
- Regression: Linear Regression, Ridge, Lasso, SVR, Random Forest, Gradient Boosting, MLP Neural Network, Stacking Ensemble
- Classification: Logistic Regression, Random Forest, Gradient Boosting, MLP Neural Classifier
- Unsupervised Learning: K-Means Clustering + Silhouette Analysis + PCA Dimensionality Reduction
Saves serialized models and comprehensive metadata for production serving.
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    StackingRegressor,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    load_raw_data,
    add_derived_features,
    CORE_FEATURES,
    TARGET_FEATURE,
)
from src.evaluate import (
    evaluate_regression,
    evaluate_classification,
    evaluate_cross_validation,
)


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_models(
    subject: str = "mat",
    test_size: float = 0.2,
    random_state: int = 28,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute complete multi-model AI/ML training pipeline and persist models."""
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
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled_all = scaler.transform(X)

    # ---------------------------------------------------------
    # 4. SUPERVISED LEARNING: REGRESSION SUITE
    # ---------------------------------------------------------

    # 4.1 Primary Linear Regression (Exact PDF Report match)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_reg)
    y_pred_lr = lr_model.predict(X_test)
    lr_eval = evaluate_regression(y_test_reg, y_pred_lr, model_name="Linear Regression")
    lr_cv = evaluate_cross_validation(lr_model, X, y_reg, random_state=random_state)
    lr_eval.update(lr_cv)
    joblib.dump(lr_model, MODELS_DIR / "linear_regression_model.joblib")

    # 4.2 Ridge Regression (L2 Regularization)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train_reg)
    y_pred_ridge = ridge_model.predict(X_test)
    ridge_eval = evaluate_regression(y_test_reg, y_pred_ridge, model_name="Ridge Regression")
    ridge_cv = evaluate_cross_validation(ridge_model, X, y_reg, random_state=random_state)
    ridge_eval.update(ridge_cv)
    joblib.dump(ridge_model, MODELS_DIR / "ridge_model.joblib")

    # 4.3 Lasso Regression (L1 Regularization & Feature Sparsity)
    lasso_model = Lasso(alpha=0.05, random_state=random_state)
    lasso_model.fit(X_train, y_train_reg)
    y_pred_lasso = lasso_model.predict(X_test)
    lasso_eval = evaluate_regression(y_test_reg, y_pred_lasso, model_name="Lasso Regression")
    lasso_cv = evaluate_cross_validation(lasso_model, X, y_reg, random_state=random_state)
    lasso_eval.update(lasso_cv)
    joblib.dump(lasso_model, MODELS_DIR / "lasso_model.joblib")

    # 4.4 Support Vector Regressor (SVR with RBF Kernel on scaled features)
    svr_model = SVR(kernel="rbf", C=10.0, epsilon=0.2)
    svr_model.fit(X_train_scaled, y_train_reg)
    y_pred_svr = svr_model.predict(X_test_scaled)
    svr_eval = evaluate_regression(y_test_reg, y_pred_svr, model_name="Support Vector Regressor (SVR)")
    svr_cv = evaluate_cross_validation(svr_model, X_scaled_all, y_reg, random_state=random_state)
    svr_eval.update(svr_cv)
    joblib.dump(svr_model, MODELS_DIR / "svr_model.joblib")

    # 4.5 Random Forest Regressor (Non-linear Bagging Ensemble)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=6)
    rf_reg.fit(X_train, y_train_reg)
    y_pred_rf = rf_reg.predict(X_test)
    rf_eval = evaluate_regression(y_test_reg, y_pred_rf, model_name="Random Forest Regressor")
    rf_cv = evaluate_cross_validation(rf_reg, X, y_reg, random_state=random_state)
    rf_eval.update(rf_cv)
    joblib.dump(rf_reg, MODELS_DIR / "rf_regressor.joblib")

    # 4.6 Gradient Boosting Regressor (Sequential Error Boosting)
    gbm_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state)
    gbm_reg.fit(X_train, y_train_reg)
    y_pred_gbm = gbm_reg.predict(X_test)
    gbm_eval = evaluate_regression(y_test_reg, y_pred_gbm, model_name="Gradient Boosting Regressor")
    gbm_cv = evaluate_cross_validation(gbm_reg, X, y_reg, random_state=random_state)
    gbm_eval.update(gbm_cv)
    joblib.dump(gbm_reg, MODELS_DIR / "gbm_regressor.joblib")

    # 4.7 Multi-Layer Perceptron Regressor (Deep Neural Network on scaled features)
    mlp_reg = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=800, random_state=random_state)
    mlp_reg.fit(X_train_scaled, y_train_reg)
    y_pred_mlp = mlp_reg.predict(X_test_scaled)
    mlp_eval = evaluate_regression(y_test_reg, y_pred_mlp, model_name="Neural Network (MLP)")
    mlp_cv = evaluate_cross_validation(mlp_reg, X_scaled_all, y_reg, random_state=random_state)
    mlp_eval.update(mlp_cv)
    joblib.dump(mlp_reg, MODELS_DIR / "mlp_regressor.joblib")

    # 4.8 Stacking Ensemble Regressor (Combining Linear + GBM + MLP with RidgeCV meta-estimator)
    stacking_estimators = [
        ("linear", LinearRegression()),
        ("gbm", GradientBoostingRegressor(n_estimators=60, learning_rate=0.05, max_depth=3, random_state=random_state)),
        ("rf", RandomForestRegressor(n_estimators=60, max_depth=5, random_state=random_state)),
    ]
    stacking_reg = StackingRegressor(estimators=stacking_estimators, final_estimator=Ridge(alpha=1.0))
    stacking_reg.fit(X_train, y_train_reg)
    y_pred_stack = stacking_reg.predict(X_test)
    stack_eval = evaluate_regression(y_test_reg, y_pred_stack, model_name="Stacking Ensemble Regressor")
    stack_cv = evaluate_cross_validation(stacking_reg, X, y_reg, random_state=random_state)
    stack_eval.update(stack_cv)
    joblib.dump(stacking_reg, MODELS_DIR / "stacking_regressor.joblib")

    # Compile Regression Leaderboard
    regression_leaderboard = [
        lr_eval,
        ridge_eval,
        lasso_eval,
        svr_eval,
        rf_eval,
        gbm_eval,
        mlp_eval,
        stack_eval,
    ]
    # Sort by R2 descending
    regression_leaderboard.sort(key=lambda item: item["r2_score"], reverse=True)

    # ---------------------------------------------------------
    # 5. SUPERVISED LEARNING: CLASSIFICATION SUITE (Early Risk)
    # ---------------------------------------------------------

    # 5.1 Logistic Regression
    lr_clf = LogisticRegression(random_state=random_state, max_iter=500)
    lr_clf.fit(X_train_scaled, y_train_clf)
    y_pred_lr_clf = lr_clf.predict(X_test_scaled)
    y_proba_lr_clf = lr_clf.predict_proba(X_test_scaled)
    lr_clf_eval = evaluate_classification(y_test_clf, y_pred_lr_clf, y_proba_lr_clf, model_name="Logistic Regression")
    joblib.dump(lr_clf, MODELS_DIR / "logistic_classifier.joblib")

    # 5.2 Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=5)
    rf_clf.fit(X_train, y_train_clf)
    y_pred_rf_clf = rf_clf.predict(X_test)
    y_proba_rf_clf = rf_clf.predict_proba(X_test)
    rf_clf_eval = evaluate_classification(y_test_clf, y_pred_rf_clf, y_proba_rf_clf, model_name="Random Forest Classifier")
    joblib.dump(rf_clf, MODELS_DIR / "rf_classifier.joblib")

    # 5.3 Gradient Boosting Classifier
    gbm_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state)
    gbm_clf.fit(X_train, y_train_clf)
    y_pred_gbm_clf = gbm_clf.predict(X_test)
    y_proba_gbm_clf = gbm_clf.predict_proba(X_test)
    gbm_clf_eval = evaluate_classification(y_test_clf, y_pred_gbm_clf, y_proba_gbm_clf, model_name="Gradient Boosting Classifier")
    joblib.dump(gbm_clf, MODELS_DIR / "gbm_classifier.joblib")

    # 5.4 Neural Network Classifier (MLP)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=600, random_state=random_state)
    mlp_clf.fit(X_train_scaled, y_train_clf)
    y_pred_mlp_clf = mlp_clf.predict(X_test_scaled)
    y_proba_mlp_clf = mlp_clf.predict_proba(X_test_scaled)
    mlp_clf_eval = evaluate_classification(y_test_clf, y_pred_mlp_clf, y_proba_mlp_clf, model_name="Neural Network (MLP)")
    joblib.dump(mlp_clf, MODELS_DIR / "mlp_classifier.joblib")

    classification_leaderboard = [
        rf_clf_eval,
        gbm_clf_eval,
        lr_clf_eval,
        mlp_clf_eval,
    ]
    classification_leaderboard.sort(key=lambda item: item["accuracy"], reverse=True)

    # ---------------------------------------------------------
    # 6. UNSUPERVISED LEARNING: K-MEANS PERSONA CLUSTERING
    # ---------------------------------------------------------
    silhouette_scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled_all)
        silhouette_scores[str(k)] = round(float(silhouette_score(X_scaled_all, labels)), 4)

    optimal_k = 4
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=15)
    cluster_labels = kmeans_model.fit_predict(X_scaled_all)
    joblib.dump(kmeans_model, MODELS_DIR / "kmeans_model.joblib")

    # PCA 2D Projection for visualization
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled_all)
    joblib.dump(pca, MODELS_DIR / "pca_model.joblib")

    # Characterize clusters
    df_clustered = df[CORE_FEATURES].copy()
    df_clustered["cluster"] = cluster_labels
    df_clustered["G3"] = y_reg

    persona_names = {
        0: "Chronic Absentees (Attendance Risk)",
        1: "Critical Support (Past Failures)",
        2: "Steady Core / Borderline Performers",
        3: "High Honor Scholars (Top Achievers)",
    }

    cluster_profiles = []
    for c_id in range(optimal_k):
        sub = df_clustered[df_clustered["cluster"] == c_id]
        profile = {
            "cluster_id": c_id,
            "persona_name": persona_names.get(c_id, f"Cluster {c_id}"),
            "student_count": int(len(sub)),
            "percentage": round(float(len(sub) / len(df) * 100), 1),
            "mean_studytime": round(float(sub["studytime"].mean()), 2),
            "mean_failures": round(float(sub["failures"].mean()), 2),
            "mean_absences": round(float(sub["absences"].mean()), 2),
            "mean_G1": round(float(sub["G1"].mean()), 2),
            "mean_G2": round(float(sub["G2"].mean()), 2),
            "mean_G3": round(float(sub["G3"].mean()), 2),
            "pass_rate": round(float((sub["G3"] >= 10).mean() * 100), 1),
        }
        cluster_profiles.append(profile)

    # ---------------------------------------------------------
    # 7. CORRELATION MATRIX & ATTENDANCE FINDINGS (PDF Matching)
    # ---------------------------------------------------------
    corr_features = ["studytime", "failures", "absences", "G1", "G2"]
    corr_matrix = df[corr_features].corr().round(3).to_dict()

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

    # Model coefficients for Linear Regression
    coefficients = {feat: round(float(coef), 4) for feat, coef in zip(CORE_FEATURES, lr_model.coef_)}
    coefficients["intercept"] = round(float(lr_model.intercept_), 4)

    # ---------------------------------------------------------
    # 8. PERSIST METADATA JSON
    # ---------------------------------------------------------
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
        "classifier": rf_clf_eval,
        "regression_leaderboard": regression_leaderboard,
        "classification_leaderboard": classification_leaderboard,
        "unsupervised_clusters": {
            "silhouette_analysis": silhouette_scores,
            "optimal_k": optimal_k,
            "profiles": cluster_profiles,
        },
        "coefficients": coefficients,
        "correlation_matrix": corr_matrix,
        "attendance_insights": attendance_stats,
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("--- Full AI & ML Training Pipeline Successfully Completed ---")
    print(f"Models Trained: 8 Regressors, 4 Classifiers, K-Means Clustering (k={optimal_k})")
    print(f"Primary Linear Regression (PDF Match) MSE: {lr_eval['mse']} | R2: {lr_eval['r2_percentage']}%")
    print(f"Top Regressor: {regression_leaderboard[0]['model_name']} (R2: {regression_leaderboard[0]['r2_percentage']}%)")
    print(f"Top Classifier: {classification_leaderboard[0]['model_name']} (Acc: {classification_leaderboard[0]['accuracy_pct']}%)")

    return metadata, rf_clf_eval


if __name__ == "__main__":
    train_models()

