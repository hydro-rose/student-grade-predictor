"""Automated Test Suite for Student Grade Prediction.
Validates data loading, model predictions, edge cases, and API integrity.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_raw_data,
    add_derived_features,
    prepare_train_test_data,
    CORE_FEATURES,
    TARGET_FEATURE,
)
from src.predictor import GradePredictor
from src.evaluate import evaluate_regression, evaluate_classification


class TestDataLoader(unittest.TestCase):
    """Test data loader and integrity checks."""

    def test_load_raw_data(self):
        df_mat = load_raw_data(subject="mat")
        self.assertGreater(len(df_mat), 300)
        for feat in CORE_FEATURES + [TARGET_FEATURE]:
            self.assertIn(feat, df_mat.columns)

    def test_add_derived_features(self):
        df = load_raw_data(subject="mat")
        derived = add_derived_features(df)
        self.assertIn("passed", derived.columns)
        self.assertIn("grade_group", derived.columns)
        self.assertIn("attendance_pct", derived.columns)
        self.assertIn("attendance_above_60", derived.columns)
        self.assertTrue(all(derived["passed"].isin([0, 1])))

    def test_prepare_train_test_data(self):
        X_train, X_test, y_train, y_test = prepare_train_test_data()
        self.assertEqual(len(X_train) + len(X_test), 395)
        self.assertEqual(X_train.shape[1], len(CORE_FEATURES))


class TestPredictor(unittest.TestCase):
    """Test inference engine."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = GradePredictor()

    def test_single_prediction(self):
        result = self.predictor.predict_single(
            studytime=2, failures=0, absences=4, g1=12, g2=13
        )
        self.assertIn("predicted_g3", result)
        self.assertIn("letter_grade", result)
        self.assertIn("status", result)
        self.assertIn("pass_probability", result)
        self.assertIn("confidence_interval", result)
        self.assertIn("recommendations", result)

        self.assertGreaterEqual(result["predicted_g3"], 0.0)
        self.assertLessEqual(result["predicted_g3"], 20.0)
        self.assertEqual(len(result["confidence_interval"]), 2)

    def test_failing_student_alert(self):
        result = self.predictor.predict_single(
            studytime=1, failures=3, absences=25, g1=4, g2=4
        )
        self.assertLess(result["predicted_g3"], 10.0)
        self.assertEqual(result["status"], "AT RISK / FAIL")
        self.assertTrue(any("below the 10.0 passing threshold" in r for r in result["recommendations"]))

    def test_batch_prediction(self):
        sample_df = pd.DataFrame({
            "studytime": [2, 4, 1],
            "failures": [0, 0, 2],
            "absences": [2, 0, 18],
            "G1": [15, 18, 6],
            "G2": [15, 19, 5],
        })
        batch_out = self.predictor.predict_batch(sample_df)
        self.assertEqual(len(batch_out), 3)
        self.assertIn("predicted_G3", batch_out.columns)
        self.assertIn("predicted_status", batch_out.columns)
        self.assertIn("predicted_letter", batch_out.columns)


class TestEvaluation(unittest.TestCase):
    """Test evaluation logic."""

    def test_evaluate_regression(self):
        y_true = np.array([10.0, 15.0, 12.0, 8.0])
        y_pred = np.array([10.5, 14.8, 11.9, 8.2])
        metrics = evaluate_regression(y_true, y_pred)
        self.assertIn("mse", metrics)
        self.assertIn("r2_score", metrics)
        self.assertGreater(metrics["r2_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
