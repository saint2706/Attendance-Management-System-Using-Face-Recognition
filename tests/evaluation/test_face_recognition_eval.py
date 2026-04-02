"""Tests for the evaluation pipeline utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation import UNKNOWN_LABEL
import json
import numpy as np

from src.evaluation.face_recognition_eval import (
    EvaluationConfig,
    SampleEvaluation,
    _calculate_far_frr,
    _infer_samples,
    _resolve_image_paths,
    compute_basic_metrics,
    compute_threshold_sweep,
    _save_metrics_json,
    _save_samples_csv,
    _save_confusion_outputs,
    _save_threshold_sweep_outputs,
    run_face_recognition_evaluation,
)


@pytest.fixture
def sample_predictions() -> list[SampleEvaluation]:
    """Return synthetic predictions for metric testing."""

    return [
        SampleEvaluation(Path("dataset/alice/0.jpg"), "alice", "alice", 0.30, True),
        SampleEvaluation(Path("dataset/bob/1.jpg"), "bob", "alice", 0.32, True),
        SampleEvaluation(Path("dataset/carol/2.jpg"), "carol", None, None, False),
        SampleEvaluation(Path("dataset/intruder/3.jpg"), UNKNOWN_LABEL, "alice", 0.25, True),
    ]


def test_compute_basic_metrics(sample_predictions: list[SampleEvaluation]) -> None:
    metrics, y_true, y_pred, labels = compute_basic_metrics(sample_predictions, threshold=0.35)

    assert metrics["samples"] == 4
    assert metrics["unknown_predictions"] == 1
    assert metrics["samples_without_embedding"] == 1
    assert metrics["accuracy"] == pytest.approx(0.25)
    assert metrics["far"] == pytest.approx(1.0)
    assert metrics["frr"] == pytest.approx(2 / 3)

    assert y_true[0] == "alice"
    assert y_pred[-1] == "alice"
    # Unknown label should always appear at the end of the label list
    assert labels[-1] == UNKNOWN_LABEL


def test_compute_threshold_sweep(sample_predictions: list[SampleEvaluation]) -> None:
    sweep = compute_threshold_sweep(sample_predictions, thresholds=[0.2, 0.35])

    assert len(sweep) == 2
    # Higher threshold should never decrease the number of predictions processed
    assert sweep[0]["samples"] == sweep[1]["samples"] == len(sample_predictions)
    assert sweep[0]["threshold"] == pytest.approx(0.2)
    assert sweep[1]["threshold"] == pytest.approx(0.35)
    # With a stricter threshold FAR should drop to zero in this synthetic dataset
    assert sweep[0]["far"] == pytest.approx(0.0)


@pytest.fixture
def mock_views_module():
    with patch("src.evaluation.face_recognition_eval._recognition_views") as mock_views:
        mock_module = MagicMock()
        mock_module.DEFAULT_DISTANCE_THRESHOLD = 0.4
        mock_module.TRAINING_DATASET_ROOT = "dataset/"
        mock_views.return_value = mock_module
        yield mock_module


class TestEvaluationConfig:
    def test_valid_config(self, mock_views_module):
        config = EvaluationConfig(threshold_start=0.1, threshold_stop=0.5, threshold_step=0.1)
        assert config.threshold_values == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_invalid_threshold_step(self, mock_views_module):
        with pytest.raises(ValueError, match="threshold_step must be positive"):
            EvaluationConfig(threshold_step=0)
        with pytest.raises(ValueError, match="threshold_step must be positive"):
            EvaluationConfig(threshold_step=-0.1)

    def test_invalid_threshold_stop(self, mock_views_module):
        with pytest.raises(ValueError, match="threshold_stop must be >= threshold_start"):
            EvaluationConfig(threshold_start=0.5, threshold_stop=0.1)

    def test_invalid_limit_samples(self, mock_views_module):
        with pytest.raises(ValueError, match="limit_samples must be positive when provided"):
            EvaluationConfig(limit_samples=0)
        with pytest.raises(ValueError, match="limit_samples must be positive when provided"):
            EvaluationConfig(limit_samples=-5)


class TestResolveImagePaths:
    def test_missing_dataset_root(self, tmp_path, mock_views_module):
        config = EvaluationConfig(dataset_root=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _resolve_image_paths(config)

    def test_empty_dataset_root(self, tmp_path, mock_views_module):
        config = EvaluationConfig(dataset_root=tmp_path)
        with pytest.raises(RuntimeError, match="No evaluation images found"):
            _resolve_image_paths(config)

    def test_fallback_to_dataset_root(self, tmp_path, mock_views_module):
        user_dir = tmp_path / "alice"
        user_dir.mkdir()
        (user_dir / "1.jpg").touch()
        (user_dir / "2.png").touch()
        (user_dir / "ignore.txt").touch()

        config = EvaluationConfig(dataset_root=tmp_path)
        paths = _resolve_image_paths(config)
        assert len(paths) == 2
        assert any(p.name == "1.jpg" for p in paths)
        assert any(p.name == "2.png" for p in paths)

    def test_load_from_csv(self, tmp_path, mock_views_module):
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        alice_img = dataset_root / "alice.jpg"
        bob_img = dataset_root / "bob.png"
        alice_img.touch()
        bob_img.touch()

        csv_path = tmp_path / "split.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "split"])
            writer.writeheader()
            writer.writerow({"image_path": "alice.jpg", "split": "test"})
            writer.writerow({"image_path": "bob.png", "split": "train"})
            writer.writerow({"image_path": "nonexistent.jpg", "split": "test"})

        config = EvaluationConfig(dataset_root=dataset_root, test_split_csv=csv_path)
        paths = _resolve_image_paths(config)
        assert len(paths) == 1
        assert paths[0] == alice_img

    def test_limit_samples(self, tmp_path, mock_views_module):
        user_dir = tmp_path / "alice"
        user_dir.mkdir()
        (user_dir / "1.jpg").touch()
        (user_dir / "2.jpg").touch()
        (user_dir / "3.jpg").touch()

        config = EvaluationConfig(dataset_root=tmp_path, limit_samples=2)
        paths = _resolve_image_paths(config)
        assert len(paths) == 2


@patch("src.evaluation.face_recognition_eval.find_closest_dataset_match")
def test_infer_samples(mock_find_closest):
    views_module = MagicMock()
    # Mock return values for _get_or_compute_cached_embedding
    # 1. Existing embedding
    # 2. Missing embedding
    views_module._get_or_compute_cached_embedding.side_effect = [[0.1, 0.2], None]

    mock_find_closest.return_value = ("alice", 0.15)

    paths = [Path("alice/1.jpg"), Path("bob/2.jpg")]

    samples = _infer_samples(
        paths,
        dataset_index=None,
        views_module=views_module,
        model_name="VGG-Face",
        detector_backend="opencv",
        enforce_detection=True,
        distance_metric="cosine",
    )

    assert len(samples) == 2

    # Check first sample (found embedding, found match)
    assert samples[0].ground_truth == "alice"
    assert samples[0].embedding_available is True
    assert samples[0].match_username == "alice"
    assert samples[0].distance == 0.15

    # Check second sample (missing embedding)
    assert samples[1].ground_truth == "bob"
    assert samples[1].embedding_available is False
    assert samples[1].match_username is None
    assert samples[1].distance is None


def test_calculate_far_frr_edge_cases():
    # Only genuine attempts, no impostors
    far, frr = _calculate_far_frr(y_true=["alice", "bob"], y_pred=["alice", "carol"])
    assert far == 0.0
    assert frr == 0.5  # 1 false reject out of 2 genuine

    # Only impostors, no genuine attempts
    far, frr = _calculate_far_frr(
        y_true=[UNKNOWN_LABEL, UNKNOWN_LABEL], y_pred=[UNKNOWN_LABEL, "alice"]
    )
    assert far == 0.5  # 1 false accept out of 2 impostor
    assert frr == 0.0

    # Empty inputs
    far, frr = _calculate_far_frr([], [])
    assert far == 0.0
    assert frr == 0.0


class TestFileIO:
    def test_save_metrics_json(self, tmp_path):
        out_path = tmp_path / "metrics.json"
        metrics = {"accuracy": 0.95, "far": 0.01}

        _save_metrics_json(metrics, out_path)

        assert out_path.exists()
        with out_path.open() as f:
            data = json.load(f)
            assert data["metrics"] == metrics
            assert "generated_at" in data

    def test_save_samples_csv(self, tmp_path, sample_predictions):
        out_path = tmp_path / "samples.csv"

        _save_samples_csv(sample_predictions, 0.35, out_path)

        assert out_path.exists()
        with out_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 4
            assert rows[0]["ground_truth"] == "alice"
            assert rows[0]["predicted_label"] == "alice"

    def test_save_confusion_outputs(self, tmp_path):
        csv_path = tmp_path / "cm.csv"
        png_path = tmp_path / "cm.png"

        cm = np.array([[10, 2], [1, 5]])
        labels = ["alice", "bob"]

        _save_confusion_outputs(cm, labels, csv_path, png_path)

        assert csv_path.exists()
        assert png_path.exists()

        with csv_path.open() as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert rows[0] == ["label", "alice", "bob"]
            assert rows[1] == ["alice", "10", "2"]
            assert rows[2] == ["bob", "1", "5"]

    def test_save_threshold_sweep_outputs(self, tmp_path):
        csv_path = tmp_path / "sweep.csv"
        png_path = tmp_path / "sweep.png"

        rows = [
            {"threshold": 0.2, "far": 0.0, "frr": 0.5, "f1": 0.6, "accuracy": 0.7},
            {"threshold": 0.4, "far": 0.1, "frr": 0.1, "f1": 0.9, "accuracy": 0.9},
        ]

        _save_threshold_sweep_outputs(rows, csv_path, png_path)

        assert csv_path.exists()
        assert png_path.exists()

        with csv_path.open() as f:
            reader = csv.DictReader(f)
            saved_rows = list(reader)
            assert len(saved_rows) == 2
            assert float(saved_rows[0]["threshold"]) == 0.2


@patch("src.evaluation.face_recognition_eval._resolve_image_paths")
@patch("src.evaluation.face_recognition_eval._load_dataset_index")
@patch("src.evaluation.face_recognition_eval._infer_samples")
def test_run_face_recognition_evaluation(
    mock_infer, mock_load_idx, mock_resolve_paths, tmp_path, mock_views_module
):
    # Setup mocks
    mock_resolve_paths.return_value = [Path("dataset/alice/1.jpg")]
    mock_load_idx.return_value = "fake_index"

    mock_infer.return_value = [
        SampleEvaluation(
            image_path=Path("dataset/alice/1.jpg"),
            ground_truth="alice",
            match_username="alice",
            distance=0.2,
            embedding_available=True,
        )
    ]

    config = EvaluationConfig(
        dataset_root=tmp_path, reports_dir=tmp_path / "reports", threshold=0.3
    )

    summary = run_face_recognition_evaluation(config)

    # Verify core metric generation
    assert summary.metrics["accuracy"] == 1.0
    assert summary.metrics["samples"] == 1

    # Verify artifacts were written
    assert "metrics" in summary.artifact_paths
    assert summary.artifact_paths["metrics"].exists()
    assert summary.artifact_paths["samples"].exists()
    assert summary.artifact_paths["confusion_png"].exists()
