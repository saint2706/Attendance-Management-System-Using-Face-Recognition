"""
Unit tests for data splitting functionality.
"""

import tempfile
from pathlib import Path

from django.test import TestCase

from recognition.data_splits import (
    create_stratified_splits,
    filter_leakage_fields,
    save_split_summary_json,
    save_splits_to_csv,
)


class DataSplitsTest(TestCase):
    """Test suite for data splitting functions."""

    def setUp(self):
        """Create a temporary directory with sample images."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = Path(self.temp_dir)

        # Create sample structure with 3 persons, 10 images each
        for person in ["alice", "bob", "charlie"]:
            person_dir = self.data_root / person
            person_dir.mkdir()
            for i in range(10):
                (person_dir / f"{i}.jpg").touch()

    def test_create_stratified_splits(self):
        """Test that stratified splits maintain person-level separation."""
        image_paths = list(self.data_root.glob("*/*.jpg"))
        self.assertEqual(len(image_paths), 30)

        train_paths, val_paths, test_paths, split_info = create_stratified_splits(
            image_paths, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_state=42
        )

        # Check total count
        total = len(train_paths) + len(val_paths) + len(test_paths)
        self.assertEqual(total, 30)

        # Check that persons are not split across sets
        train_persons = {p.parent.name for p in train_paths}
        val_persons = {p.parent.name for p in val_paths}
        test_persons = {p.parent.name for p in test_paths}

        # No person should appear in multiple splits
        self.assertEqual(len(train_persons & val_persons), 0)
        self.assertEqual(len(train_persons & test_persons), 0)
        self.assertEqual(len(val_persons & test_persons), 0)

        # Total persons should be 3
        all_persons = train_persons | val_persons | test_persons
        self.assertEqual(len(all_persons), 3)

        # Check split_info structure
        self.assertIn("total_images", split_info)
        self.assertIn("total_persons", split_info)
        self.assertEqual(split_info["total_images"], 30)
        self.assertEqual(split_info["total_persons"], 3)

    def test_save_splits_to_csv(self):
        """Test CSV export of splits."""
        image_paths = list(self.data_root.glob("*/*.jpg"))
        train_paths, val_paths, test_paths, _ = create_stratified_splits(
            image_paths, random_state=42
        )

        output_path = Path(self.temp_dir) / "splits.csv"
        save_splits_to_csv(train_paths, val_paths, test_paths, output_path)

        self.assertTrue(output_path.exists())

        # Read and validate CSV
        import pandas as pd

        df = pd.read_csv(output_path)
        self.assertIn("split", df.columns)
        self.assertIn("person", df.columns)
        self.assertEqual(len(df), 30)

    def test_save_split_summary_json(self):
        """Test JSON export of split summary."""
        image_paths = list(self.data_root.glob("*/*.jpg"))
        _, _, _, split_info = create_stratified_splits(image_paths, random_state=42)

        output_path = Path(self.temp_dir) / "summary.json"
        save_split_summary_json(split_info, output_path)

        self.assertTrue(output_path.exists())

        # Read and validate JSON
        import json

        with open(output_path) as f:
            data = json.load(f)

        self.assertEqual(data["total_images"], 30)
        self.assertEqual(data["random_state"], 42)

    def test_filter_leakage_fields(self):
        """Test leakage field filtering."""
        data_dict = {
            "image": "path/to/image.jpg",
            "username": "alice",
            "employee_id": "EMP123",
            "full_name": "Alice Smith",
            "embedding": [0.1, 0.2, 0.3],
            "score": 0.95,
        }

        filtered = filter_leakage_fields(data_dict)

        # High-risk fields should be removed
        self.assertNotIn("username", filtered)
        self.assertNotIn("employee_id", filtered)
        self.assertNotIn("full_name", filtered)

        # Safe fields should remain
        self.assertIn("image", filtered)
        self.assertIn("embedding", filtered)
        self.assertIn("score", filtered)

    def test_insufficient_persons_raises_error(self):
        """Test that error is raised with too few persons."""
        # Create only 2 persons
        small_dir = Path(self.temp_dir) / "small"
        small_dir.mkdir()
        for person in ["alice", "bob"]:
            person_dir = small_dir / person
            person_dir.mkdir()
            (person_dir / "1.jpg").touch()

        image_paths = list(small_dir.glob("*/*.jpg"))

        with self.assertRaises(ValueError):
            create_stratified_splits(image_paths, random_state=42)
