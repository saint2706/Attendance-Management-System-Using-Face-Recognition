"""
Integration tests for the rigor pass features.

These tests verify that the components work together correctly.
"""

import json
import tempfile
from pathlib import Path

from django.test import TestCase

from recognition.data_splits import create_stratified_splits, save_split_summary_json
from src.common.seeding import set_global_seed


class IntegrationTest(TestCase):
    """Integration tests for reproducibility and workflow."""

    def setUp(self):
        """Set up test environment."""
        set_global_seed(42)
        self.temp_dir = tempfile.mkdtemp()

    def test_reproducibility_workflow(self):
        """Test that the reproducibility workflow produces consistent results."""
        # Create sample data
        data_root = Path(self.temp_dir) / "data"
        data_root.mkdir()

        for person in ["alice", "bob", "charlie", "david"]:
            person_dir = data_root / person
            person_dir.mkdir()
            for i in range(10):
                (person_dir / f"{i}.jpg").touch()

        image_paths = list(data_root.glob("*/*.jpg"))

        # Run split generation twice with same seed
        set_global_seed(42)
        train1, val1, test1, info1 = create_stratified_splits(
            image_paths, random_state=42
        )

        set_global_seed(42)
        train2, val2, test2, info2 = create_stratified_splits(
            image_paths, random_state=42
        )

        # Results should be identical
        self.assertEqual(len(train1), len(train2))
        self.assertEqual(len(val1), len(val2))
        self.assertEqual(len(test1), len(test2))

        # Person assignments should be identical
        train1_persons = {p.parent.name for p in train1}
        train2_persons = {p.parent.name for p in train2}
        self.assertEqual(train1_persons, train2_persons)

    def test_split_summary_roundtrip(self):
        """Test that split summaries can be saved and loaded."""
        # Create sample data
        data_root = Path(self.temp_dir) / "data2"
        data_root.mkdir()

        for person in ["user1", "user2", "user3"]:
            person_dir = data_root / person
            person_dir.mkdir()
            for i in range(5):
                (person_dir / f"{i}.jpg").touch()

        image_paths = list(data_root.glob("*/*.jpg"))

        # Generate splits
        _, _, _, split_info = create_stratified_splits(image_paths, random_state=42)

        # Save to JSON
        output_path = Path(self.temp_dir) / "split_info.json"
        save_split_summary_json(split_info, output_path)

        # Load and verify
        with open(output_path) as f:
            loaded_info = json.load(f)

        self.assertEqual(split_info["total_images"], loaded_info["total_images"])
        self.assertEqual(split_info["total_persons"], loaded_info["total_persons"])
        self.assertEqual(split_info["random_state"], loaded_info["random_state"])

    def test_seeding_consistency(self):
        """Test that seeding produces consistent random numbers."""
        import random

        import numpy as np

        # First run
        set_global_seed(123)
        random1 = random.random()
        numpy1 = np.random.rand()

        # Second run with same seed
        set_global_seed(123)
        random2 = random.random()
        numpy2 = np.random.rand()

        # Should be identical
        self.assertEqual(random1, random2)
        self.assertEqual(numpy1, numpy2)

    def test_policy_yaml_loadable(self):
        """Test that policy configuration is valid YAML."""
        import yaml

        policy_path = Path(__file__).parent.parent / "configs" / "policy.yaml"

        if policy_path.exists():
            with open(policy_path) as f:
                policy = yaml.safe_load(f)

            # Check structure
            self.assertIn("score_bands", policy)
            self.assertIn("actions", policy)
            self.assertIn("thresholds", policy)

            # Check score bands
            bands = policy["score_bands"]
            self.assertIn("confident_accept", bands)
            self.assertIn("uncertain", bands)
            self.assertIn("reject", bands)

            # Check each band has required fields
            for band_name, band_config in bands.items():
                self.assertIn("threshold_min", band_config)
                self.assertIn("threshold_max", band_config)
                self.assertIn("action", band_config)
