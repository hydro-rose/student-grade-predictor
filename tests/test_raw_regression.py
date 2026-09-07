"""Unit Tests for 100% Raw Python Linear Regression and Statistics Engine.
Verifies mathematical correctness of linear algebra, OLS, metrics, and data processing.
"""

import unittest
import math
from src.raw_linear_regression import (
    transpose,
    matmul,
    matvec_mul,
    invert_matrix_gauss_jordan,
    mean,
    mean_squared_error,
    r2_score,
    pearson_correlation,
    RawLinearRegression,
    RawGradientDescentRegressor,
    train_test_split_pure,
)


class TestRawLinearAlgebra(unittest.TestCase):
    """Test pure python matrix and vector arithmetic."""

    def test_transpose(self):
        A = [[1, 2, 3], [4, 5, 6]]
        A_T = transpose(A)
        self.assertEqual(A_T, [[1, 4], [2, 5], [3, 6]])

    def test_matmul(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        # [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        # [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        C = matmul(A, B)
        self.assertEqual(C, [[19.0, 22.0], [43.0, 50.0]])

    def test_matvec_mul(self):
        A = [[2, 3], [1, -1]]
        v = [4, 2]
        # [2*4 + 3*2, 1*4 - 1*2] = [14, 2]
        res = matvec_mul(A, v)
        self.assertEqual(res, [14.0, 2.0])

    def test_invert_matrix_2x2(self):
        # A = [[4, 7], [2, 6]], det = 24 - 14 = 10
        # A^-1 = [[0.6, -0.7], [-0.2, 0.4]]
        A = [[4.0, 7.0], [2.0, 6.0]]
        A_inv = invert_matrix_gauss_jordan(A)
        self.assertAlmostEqual(A_inv[0][0], 0.6, places=5)
        self.assertAlmostEqual(A_inv[0][1], -0.7, places=5)
        self.assertAlmostEqual(A_inv[1][0], -0.2, places=5)
        self.assertAlmostEqual(A_inv[1][1], 0.4, places=5)

        # Verify A * A^-1 = I
        ident = matmul(A, A_inv)
        self.assertAlmostEqual(ident[0][0], 1.0, places=5)
        self.assertAlmostEqual(ident[0][1], 0.0, places=5)
        self.assertAlmostEqual(ident[1][0], 0.0, places=5)
        self.assertAlmostEqual(ident[1][1], 1.0, places=5)


class TestRawMetrics(unittest.TestCase):
    """Test pure python statistics and error metrics."""

    def test_mean(self):
        self.assertEqual(mean([2.0, 4.0, 6.0, 8.0]), 5.0)

    def test_mean_squared_error(self):
        y_true = [10.0, 15.0, 20.0]
        y_pred = [12.0, 14.0, 18.0]
        # errors: 2, -1, -2 -> squared: 4, 1, 4 -> sum=9 -> mean=3.0
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 3.0, places=5)

    def test_r2_score(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(r2_score(y_true, y_pred), 1.0, places=5)

    def test_pearson_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        # Perfect positive correlation
        self.assertAlmostEqual(pearson_correlation(x, y), 1.0, places=5)


class TestRawRegressionModel(unittest.TestCase):
    """Test raw python Ordinary Least Squares fitting."""

    def test_perfect_linear_fit(self):
        # y = 2*x1 + 3*x2 + 5
        X = [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [5.0, 5.0],
        ]
        y = [2 * row[0] + 3 * row[1] + 5 for row in X]

        model = RawLinearRegression(fit_intercept=True)
        model.fit(X, y)

        self.assertAlmostEqual(model.intercept, 5.0, places=2)
        self.assertAlmostEqual(model.coefficients[0], 2.0, places=2)
        self.assertAlmostEqual(model.coefficients[1], 3.0, places=2)

        preds = model.predict(X)
        self.assertAlmostEqual(r2_score(y, preds), 1.0, places=4)

    def test_train_test_split_pure(self):
        X = [[i] for i in range(100)]
        y = [i for i in range(100)]
        X_train, X_test, y_train, y_test = train_test_split_pure(X, y, test_size=0.2, random_state=42)
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)


if __name__ == "__main__":
    unittest.main()
