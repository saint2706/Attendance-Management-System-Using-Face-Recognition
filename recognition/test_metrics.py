"""
Unit tests for evaluation metrics with confidence intervals.
"""

from django.test import TestCase

import numpy as np

from recognition.evaluation.metrics import (
    bootstrap_confidence_intervals,
    calculate_eer,
    calculate_verification_metrics,
    find_optimal_threshold,
)


class MetricsTest(TestCase):
    """Test suite for evaluation metrics."""

    def setUp(self):
        """Create synthetic evaluation data."""
        np.random.seed(42)
        n = 100

        # Create binary labels
        self.y_true = np.array([1] * 50 + [0] * 50)

        # Create scores (genuine pairs have higher scores)
        scores_genuine = np.random.beta(8, 2, 50)
        scores_impostor = np.random.beta(2, 8, 50)
        self.y_scores = np.concatenate([scores_genuine, scores_impostor])

        # Shuffle
        shuffle_idx = np.random.permutation(n)
        self.y_true = self.y_true[shuffle_idx]
        self.y_scores = self.y_scores[shuffle_idx]

    def test_calculate_eer(self):
        """Test Equal Error Rate calculation."""
        eer, threshold = calculate_eer(self.y_true, self.y_scores)

        # EER should be between 0 and 1
        self.assertGreaterEqual(eer, 0.0)
        self.assertLessEqual(eer, 1.0)

        # Threshold should be reasonable
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

        # With well-separated scores, EER should be relatively low
        self.assertLess(eer, 0.3)

    def test_find_optimal_threshold(self):
        """Test optimal F1 threshold selection."""
        threshold, f1 = find_optimal_threshold(self.y_true, self.y_scores)

        # Threshold should be between 0 and 1
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

        # F1 should be between 0 and 1
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)

        # With well-separated data, F1 should be high
        self.assertGreater(f1, 0.5)

    def test_calculate_verification_metrics(self):
        """Test comprehensive verification metrics calculation."""
        metrics = calculate_verification_metrics(self.y_true, self.y_scores, threshold=0.5)

        # Check required metrics exist
        required_metrics = [
            "roc_auc",
            "pr_auc",
            "eer",
            "eer_threshold",
            "brier_score",
            "optimal_threshold",
            "optimal_f1",
            "operating_points",
        ]

        for metric in required_metrics:
            self.assertIn(metric, metrics)

        # Check value ranges
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertLessEqual(metrics["roc_auc"], 1.0)

        self.assertGreaterEqual(metrics["eer"], 0.0)
        self.assertLessEqual(metrics["eer"], 1.0)

        self.assertGreaterEqual(metrics["brier_score"], 0.0)
        self.assertLessEqual(metrics["brier_score"], 1.0)

        # Check operating points structure
        self.assertIsInstance(metrics["operating_points"], dict)
        self.assertGreater(len(metrics["operating_points"]), 0)

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation."""
        ci_results = bootstrap_confidence_intervals(
            self.y_true, self.y_scores, n_bootstrap=100, random_state=42
        )

        # Check structure
        self.assertIn("auc", ci_results)
        self.assertIn("eer", ci_results)
        self.assertIn("optimal_f1", ci_results)

        # Check AUC confidence interval
        auc_ci = ci_results["auc"]
        self.assertIn("mean", auc_ci)
        self.assertIn("ci_lower", auc_ci)
        self.assertIn("ci_upper", auc_ci)

        # CI bounds should be ordered
        if auc_ci["mean"] is not None:
            self.assertLessEqual(auc_ci["ci_lower"], auc_ci["mean"])
            self.assertLessEqual(auc_ci["mean"], auc_ci["ci_upper"])

        # Check EER confidence interval
        eer_ci = ci_results["eer"]
        if eer_ci["mean"] is not None:
            self.assertLessEqual(eer_ci["ci_lower"], eer_ci["mean"])
            self.assertLessEqual(eer_ci["mean"], eer_ci["ci_upper"])

    def test_bootstrap_with_small_sample(self):
        """Test bootstrap handles small samples gracefully."""
        y_true_small = np.array([1, 1, 0, 0, 1])
        y_scores_small = np.array([0.9, 0.8, 0.3, 0.2, 0.7])

        ci_results = bootstrap_confidence_intervals(
            y_true_small, y_scores_small, n_bootstrap=50, random_state=42
        )

        # Should complete without error
        self.assertIn("auc", ci_results)

    def test_perfect_separation(self):
        """Test metrics with perfectly separated scores."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([1.0, 0.9, 0.8, 0.3, 0.2, 0.1])

        metrics = calculate_verification_metrics(y_true, y_scores, threshold=0.5)

        # Perfect separation should give AUC = 1.0
        self.assertAlmostEqual(metrics["roc_auc"], 1.0, places=5)

        # EER should be very low
        self.assertLess(metrics["eer"], 0.01)
