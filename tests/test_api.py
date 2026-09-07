"""Test FastAPI endpoints and application functionality.
"""

import unittest
from pathlib import Path
import json
import urllib.request
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestWebAppAPI(unittest.TestCase):
    """Verify live web application endpoints."""

    BASE_URL = "http://127.0.0.1:8000"

    def test_get_index(self):
        with urllib.request.urlopen(f"{self.BASE_URL}/") as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            self.assertIn("SynapseGrade AI", html)
            self.assertIn("Student Profile", html)

    def test_get_metrics(self):
        with urllib.request.urlopen(f"{self.BASE_URL}/api/metrics") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("linear_regression", data)
            self.assertEqual(data["linear_regression"]["r2_score"], 0.8614)
            self.assertIn("correlation_matrix", data)

    def test_sample_students(self):
        with urllib.request.urlopen(f"{self.BASE_URL}/api/sample-students") as resp:
            self.assertEqual(resp.status, 200)
            students = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(len(students), 5)
            self.assertEqual(students[0]["name"], "Maria Silva")

    def test_predict_endpoint_pass(self):
        payload = {"studytime": 3.0, "failures": 0, "absences": 2, "g1": 16.0, "g2": 17.0}
        req = urllib.request.Request(
            f"{self.BASE_URL}/api/predict",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "PASS")
            self.assertGreaterEqual(data["predicted_g3"], 16.0)

    def test_predict_endpoint_fail(self):
        payload = {"studytime": 1.0, "failures": 2, "absences": 24, "g1": 6.0, "g2": 7.0}
        req = urllib.request.Request(
            f"{self.BASE_URL}/api/predict",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "AT RISK / FAIL")
            self.assertLess(data["predicted_g3"], 10.0)

    def test_get_models(self):
        with urllib.request.urlopen(f"{self.BASE_URL}/api/models") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("available_models", data)
            self.assertIn("regression_leaderboard", data)
            self.assertGreaterEqual(len(data["regression_leaderboard"]), 6)

    def test_explain_endpoint(self):
        payload = {"studytime": 3.0, "failures": 0, "absences": 2, "g1": 16.0, "g2": 17.0}
        req = urllib.request.Request(
            f"{self.BASE_URL}/api/explain",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("baseline_grade", data)
            self.assertIn("contributions", data)

    def test_optimize_goal_endpoint(self):
        payload = {"target_g3": 15.0, "studytime": 2.0, "failures": 0, "absences": 8, "g1": 11.0, "g2": 12.0}
        req = urllib.request.Request(
            f"{self.BASE_URL}/api/optimize-goal",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["target_g3"], 15.0)
            self.assertIn("action_steps", data)

    def test_get_clusters(self):
        with urllib.request.urlopen(f"{self.BASE_URL}/api/clusters") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("optimal_k", data)
            self.assertIn("profiles", data)


if __name__ == "__main__":
    unittest.main()
