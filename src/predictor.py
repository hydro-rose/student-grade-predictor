"""Inference Engine for Student Grade Prediction.
Loads saved models and generates real-time predictions, letter grades,
pass/fail probabilities, and actionable recommendations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.data_loader import CORE_FEATURES


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class GradePredictor:
    """Production predictor for student academic performance."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.lr_model = None
        self.rf_classifier = None
        self.metadata = {}
        self._load_models()

    def _load_models(self):
        lr_path = self.models_dir / "linear_regression_model.joblib"
        rf_path = self.models_dir / "rf_classifier.joblib"
        meta_path = self.models_dir / "metadata.json"

        if lr_path.exists():
            self.lr_model = joblib.load(lr_path)
        if rf_path.exists():
            self.rf_classifier = joblib.load(rf_path)
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    @staticmethod
    def calculate_letter_grade(score: float) -> str:
        """Map 0-20 score to standard academic letter grades."""
        score = np.clip(score, 0.0, 20.0)
        if score >= 16:
            return "A (Excellent)"
        elif score >= 14:
            return "B (Good)"
        elif score >= 12:
            return "C (Satisfactory)"
        elif score >= 10:
            return "D (Sufficient)"
        else:
            return "F (Fail / At Risk)"

    @staticmethod
    def generate_recommendations(
        studytime: float,
        failures: int,
        absences: int,
        g1: float,
        g2: float,
        predicted_grade: float,
    ) -> List[str]:
        """Generate targeted academic advice based on empirical model feature weights."""
        recs = []

        # Attendance advice (directly reflecting report's attendance findings)
        if absences > 10:
            recs.append(
                f"High absence count ({absences} days). In the study, students with >60% attendance consistently achieved significantly higher final grades. Prioritize attending all regular class sessions."
            )
        elif absences <= 2:
            recs.append("Exemplary attendance record! Maintaining high attendance is strongly correlated with academic success.")

        # Study time advice
        if studytime < 2:
            recs.append(
                "Weekly study time is currently below 2 hours. Increasing to 2-5 hours (Category 2) or 5-10 hours (Category 3) provides a proven positive boost to the final score."
            )
        elif studytime >= 3:
            recs.append("Strong weekly study commitment (5+ hours). Consistency in regular review will protect your performance.")

        # Prior failure intervention
        if failures > 0:
            recs.append(
                f"Student has {failures} past course failure(s). Targeted tutoring, unit test retakes, and office hour check-ins are strongly advised to overcome past difficulty."
            )

        # Grade trajectory advice
        grade_trend = g2 - g1
        if grade_trend > 1.0:
            recs.append(f"Positive momentum detected: Grade improved by +{grade_trend:.1f} points from Period 1 to Period 2. Keep this upward momentum into the final exam!")
        elif grade_trend < -1.0:
            recs.append(f"Caution: Grade dropped by {abs(grade_trend):.1f} points between Period 1 and Period 2. Address conceptual gaps early before the final G3 exam.")

        # Final score expectation
        if predicted_grade < 10.0:
            recs.append("CRITICAL: Predicted final grade is below the 10.0 passing threshold. Immediate academic intervention and remedial coursework are recommended.")
        elif predicted_grade >= 16.0:
            recs.append("Outstanding trajectory: On track for top-tier academic honors.")

        return recs

    def predict_single(
        self,
        studytime: float,
        failures: int,
        absences: int,
        g1: float,
        g2: float,
    ) -> Dict[str, Any]:
        """Generate full prediction package for an individual student.
        
        Args:
            studytime: 1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)
            failures: number of past class failures (0 to 4)
            absences: number of absences (0 to 93)
            g1: Period 1 grade (0 to 20)
            g2: Period 2 grade (0 to 20)
            
        Returns:
            Dictionary with predicted G3, confidence intervals, classification, and recommendations.
        """
        if self.lr_model is None:
            raise RuntimeError("Linear Regression model not loaded. Run train.py first.")

        input_df = pd.DataFrame(
            [[studytime, failures, absences, g1, g2]],
            columns=CORE_FEATURES,
        )

        # Regression prediction
        raw_pred = float(self.lr_model.predict(input_df)[0])
        predicted_grade = round(float(np.clip(raw_pred, 0.0, 20.0)), 2)

        # Standard error / confidence interval based on test RMSE (~1.66)
        rmse = self.metadata.get("linear_regression", {}).get("rmse", 1.66)
        lower_bound = round(max(0.0, predicted_grade - 1.96 * rmse * 0.4), 2)
        upper_bound = round(min(20.0, predicted_grade + 1.96 * rmse * 0.4), 2)

        # Classification prediction
        pass_prob = None
        will_pass = predicted_grade >= 10.0
        if self.rf_classifier is not None:
            proba = self.rf_classifier.predict_proba(input_df)[0]
            pass_prob = round(float(proba[1] * 100), 1)
            will_pass = bool(self.rf_classifier.predict(input_df)[0])
        else:
            pass_prob = round(100.0 if will_pass else 0.0, 1)

        letter_grade = self.calculate_letter_grade(predicted_grade)
        recommendations = self.generate_recommendations(
            studytime, failures, absences, g1, g2, predicted_grade
        )

        return {
            "predicted_g3": predicted_grade,
            "letter_grade": letter_grade,
            "status": "PASS" if will_pass else "AT RISK / FAIL",
            "pass_probability": pass_prob,
            "confidence_interval": [lower_bound, upper_bound],
            "inputs": {
                "studytime": studytime,
                "failures": failures,
                "absences": absences,
                "g1": g1,
                "g2": g2,
            },
            "recommendations": recommendations,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for multiple students simultaneously."""
        if self.lr_model is None:
            raise RuntimeError("Linear Regression model not loaded.")

        missing = [c for c in CORE_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Batch input dataframe missing required columns: {missing}")

        X = df[CORE_FEATURES]
        preds = self.lr_model.predict(X)
        clipped_preds = np.clip(preds, 0.0, 20.0).round(2)

        res_df = df.copy()
        res_df["predicted_G3"] = clipped_preds
        res_df["predicted_status"] = np.where(res_df["predicted_G3"] >= 10.0, "PASS", "AT RISK")
        res_df["predicted_letter"] = [self.calculate_letter_grade(p) for p in clipped_preds]
        return res_df
