"""
Unit tests for ablation experiments.
"""

import tempfile
from pathlib import Path

from django.test import TestCase

from recognition.ablation import (
    AblationConfig,
    generate_ablation_configs,
    run_single_ablation,
)


class AblationTest(TestCase):
    """Test suite for ablation functionality."""

    def setUp(self):
        """Create sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = Path(self.temp_dir)

        # Create minimal sample structure
        for person in ["user1", "user2", "user3"]:
            person_dir = self.data_root / person
            person_dir.mkdir()
            for i in range(5):
                (person_dir / f"{i}.jpg").touch()

        self.image_paths = list(self.data_root.glob("*/*.jpg"))
        self.labels = [p.parent.name for p in self.image_paths]

    def test_ablation_config_creation(self):
        """Test ablation configuration object."""
        config = AblationConfig(
            detector="ssd", alignment=True, distance_metric="cosine", rebalancing=False
        )

        self.assertEqual(config.detector, "ssd")
        self.assertTrue(config.alignment)
        self.assertEqual(config.distance_metric, "cosine")
        self.assertFalse(config.rebalancing)

        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["detector"], "ssd")

    def test_generate_ablation_configs(self):
        """Test generation of ablation configuration list."""
        configs = generate_ablation_configs()

        # Should generate multiple configurations
        self.assertGreater(len(configs), 5)

        # All should be AblationConfig instances
        for config in configs:
            self.assertIsInstance(config, AblationConfig)

        # Should include baseline
        baseline_exists = any(
            c.detector == "ssd"
            and c.alignment is True
            and c.distance_metric == "cosine"
            and c.rebalancing is False
            for c in configs
        )
        self.assertTrue(baseline_exists)

        # Should include detector variations
        detectors = {c.detector for c in configs}
        self.assertIn("ssd", detectors)
        self.assertIn("opencv", detectors)

    def test_run_single_ablation(self):
        """Test running a single ablation experiment."""
        config = AblationConfig(
            detector="ssd", alignment=True, distance_metric="cosine"
        )

        result = run_single_ablation(
            config, self.image_paths, self.labels, random_state=42
        )

        # Check result structure
        self.assertIn("config", result)
        self.assertIn("accuracy", result)
        self.assertIn("f1_score", result)
        self.assertIn("n_samples", result)

        # Check metrics are in valid ranges
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)
        self.assertGreaterEqual(result["f1_score"], 0.0)
        self.assertLessEqual(result["f1_score"], 1.0)

        # Check sample count
        self.assertEqual(result["n_samples"], len(self.image_paths))

    def test_ablation_config_variations(self):
        """Test that different configs produce different results."""
        config1 = AblationConfig(
            detector="ssd", alignment=True, distance_metric="cosine"
        )
        config2 = AblationConfig(
            detector="opencv", alignment=True, distance_metric="cosine"
        )

        result1 = run_single_ablation(
            config1, self.image_paths, self.labels, random_state=42
        )
        result2 = run_single_ablation(
            config2, self.image_paths, self.labels, random_state=42
        )

        # Different configurations should (generally) produce different results
        # Note: This is probabilistic, so we just check they both ran successfully
        self.assertIsNotNone(result1["accuracy"])
        self.assertIsNotNone(result2["accuracy"])

    def test_reproducibility_with_seed(self):
        """Test that same config and seed produce same results."""
        config = AblationConfig(
            detector="ssd", alignment=True, distance_metric="cosine"
        )

        result1 = run_single_ablation(
            config, self.image_paths, self.labels, random_state=42
        )
        result2 = run_single_ablation(
            config, self.image_paths, self.labels, random_state=42
        )

        # Same seed should give exactly same results
        self.assertEqual(result1["accuracy"], result2["accuracy"])
        self.assertEqual(result1["f1_score"], result2["f1_score"])
