"""
Unit tests for failure analysis functionality.
"""

from django.test import TestCase

import numpy as np

from recognition.analysis.failures import analyze_failures, analyze_subgroups


class FailureAnalysisTest(TestCase):
    """Test suite for failure analysis functions."""

    def setUp(self):
        """Create synthetic test data."""
        np.random.seed(42)

        # Create predictions with some errors
        self.y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])  # 4 errors
        self.y_scores = np.array([0.9, 0.85, 0.8, 0.45, 0.4, 0.35, 0.3, 0.25, 0.6, 0.65])

    def test_analyze_failures_basic(self):
        """Test basic failure analysis."""
        fa_df, fr_df = analyze_failures(self.y_true, self.y_pred, self.y_scores, top_n=5)

        # Should have false accepts (predicted 1, true 0)
        # In our data: indices 8, 9 are false accepts
        self.assertEqual(len(fa_df), 2)

        # Should have false rejects (predicted 0, true 1)
        # In our data: indices 3, 4 are false rejects
        self.assertEqual(len(fr_df), 2)

        # Check columns exist
        for df in [fa_df, fr_df]:
            if len(df) > 0:
                self.assertIn("failure_type", df.columns)
                self.assertIn("score", df.columns)
                self.assertIn("true_label", df.columns)
                self.assertIn("predicted_label", df.columns)

    def test_analyze_failures_with_metadata(self):
        """Test failure analysis includes metadata fields."""
        fa_df, fr_df = analyze_failures(self.y_true, self.y_pred, self.y_scores, top_n=5)

        # Check metadata columns exist
        for df in [fa_df, fr_df]:
            if len(df) > 0:
                self.assertIn("lighting", df.columns)
                self.assertIn("pose", df.columns)
                self.assertIn("occlusion", df.columns)

    def test_analyze_failures_top_n_limit(self):
        """Test that top_n parameter limits results."""
        # Create more errors
        y_true = np.array([1] * 20 + [0] * 20)
        y_pred = np.array([0] * 20 + [1] * 20)  # All wrong
        y_scores = np.random.rand(40)

        fa_df, fr_df = analyze_failures(y_true, y_pred, y_scores, top_n=5)

        # Should limit to top 5 of each type
        self.assertLessEqual(len(fa_df), 5)
        self.assertLessEqual(len(fr_df), 5)

    def test_analyze_failures_no_errors(self):
        """Test failure analysis with perfect predictions."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])  # All correct
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])

        fa_df, fr_df = analyze_failures(y_true, y_pred, y_scores)

        # Should have no failures
        self.assertEqual(len(fa_df), 0)
        self.assertEqual(len(fr_df), 0)

    def test_analyze_subgroups_basic(self):
        """Test subgroup analysis."""
        groups = np.array(["camera1"] * 5 + ["camera2"] * 5)

        df = analyze_subgroups(self.y_true, self.y_pred, self.y_scores, groups, output_path=None)

        # Should have 2 groups
        self.assertEqual(len(df), 2)

        # Check columns
        self.assertIn("group", df.columns)
        self.assertIn("n_samples", df.columns)
        self.assertIn("accuracy", df.columns)
        self.assertIn("precision", df.columns)
        self.assertIn("recall", df.columns)
        self.assertIn("f1_score", df.columns)

        # Check metrics are in valid ranges
        for _, row in df.iterrows():
            self.assertGreaterEqual(row["accuracy"], 0.0)
            self.assertLessEqual(row["accuracy"], 1.0)

    def test_analyze_subgroups_unbalanced(self):
        """Test subgroup analysis with unbalanced groups."""
        # Create unbalanced groups
        groups = np.array(["A"] * 8 + ["B"] * 2)

        df = analyze_subgroups(self.y_true, self.y_pred, self.y_scores, groups, output_path=None)

        # Should handle both groups
        self.assertEqual(len(df), 2)

        # Check sample counts
        group_a = df[df["group"] == "A"]
        group_b = df[df["group"] == "B"]
        self.assertEqual(group_a.iloc[0]["n_samples"], 8)
        self.assertEqual(group_b.iloc[0]["n_samples"], 2)

    def test_failure_analysis_sorted_by_score(self):
        """Test that failures are sorted by score."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 1])  # All wrong
        y_scores = np.array([0.3, 0.5, 0.2, 0.7, 0.9, 0.6])

        fa_df, fr_df = analyze_failures(y_true, y_pred, y_scores, top_n=3)

        # False accepts (indices 3,4,5) should be sorted by score descending
        if len(fa_df) > 1:
            scores = fa_df["score"].values
            # Check descending order
            for i in range(len(scores) - 1):
                self.assertGreaterEqual(scores[i], scores[i + 1])

        # False rejects (indices 0,1,2) should also be sorted by score descending
        if len(fr_df) > 1:
            scores = fr_df["score"].values
            for i in range(len(scores) - 1):
                self.assertGreaterEqual(scores[i], scores[i + 1])
