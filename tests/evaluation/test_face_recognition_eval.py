"""Tests for the evaluation pipeline utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation import UNKNOWN_LABEL
from src.evaluation.face_recognition_eval import (
    EvaluationConfig,
    SampleEvaluation,
    _infer_samples,
    _resolve_image_paths,
    build_argument_parser,
    compute_basic_metrics,
    compute_threshold_sweep,
    main,
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


@patch("src.evaluation.face_recognition_eval._recognition_views")
def test_evaluation_config_post_init(mock_views):
    mock_views_module = MagicMock()
    mock_views_module.DEFAULT_DISTANCE_THRESHOLD = 0.4
    mock_views_module.TRAINING_DATASET_ROOT = "test/root"
    mock_views.return_value = mock_views_module

    # Test valid initialization
    config = EvaluationConfig(
        threshold=0.5,
        threshold_start=0.2,
        threshold_stop=0.8,
        threshold_step=0.1,
        limit_samples=100,
    )
    assert config.threshold == 0.5
    assert len(config.threshold_values) > 0

    # Test invalid threshold step
    with pytest.raises(ValueError, match="threshold_step must be positive"):
        EvaluationConfig(threshold_step=-0.1)

    # Test invalid threshold stop
    with pytest.raises(ValueError, match="threshold_stop must be >= threshold_start"):
        EvaluationConfig(threshold_start=0.5, threshold_stop=0.2)

    # Test invalid limit samples
    with pytest.raises(ValueError, match="limit_samples must be positive when provided"):
        EvaluationConfig(limit_samples=-5)


@patch("src.evaluation.face_recognition_eval._recognition_views")
def test_resolve_image_paths(mock_views, tmp_path):
    mock_views_module = MagicMock()
    mock_views_module.DEFAULT_DISTANCE_THRESHOLD = 0.4
    mock_views_module.TRAINING_DATASET_ROOT = str(tmp_path / "dataset")
    mock_views.return_value = mock_views_module

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    user1_dir = dataset_dir / "user1"
    user1_dir.mkdir()
    (user1_dir / "img1.jpg").write_text("test")
    (user1_dir / "img2.jpg").write_text("test")

    # Test fallback to scanning dataset root
    config = EvaluationConfig(dataset_root=dataset_dir)
    paths = _resolve_image_paths(config)
    assert len(paths) == 2

    # Test CSV split
    split_csv = tmp_path / "split.csv"
    with open(split_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image_path"])
        writer.writerow(["test", str(user1_dir / "img1.jpg")])
        writer.writerow(["train", str(user1_dir / "img2.jpg")])

    config_csv = EvaluationConfig(dataset_root=dataset_dir, test_split_csv=split_csv)
    paths_csv = _resolve_image_paths(config_csv)
    assert len(paths_csv) == 1
    assert paths_csv[0] == user1_dir / "img1.jpg"

    # Test empty returns error
    with pytest.raises(RuntimeError, match="No evaluation images"):
        empty_config = EvaluationConfig(dataset_root=tmp_path / "empty")
        empty_config.dataset_root.mkdir()
        _resolve_image_paths(empty_config)


def test_infer_samples():
    mock_views = MagicMock()
    mock_views._get_or_compute_cached_embedding.return_value = [0.1, 0.2]

    with patch("src.evaluation.face_recognition_eval.find_closest_dataset_match") as mock_match:
        mock_match.return_value = ("alice", 0.15)

        paths = [Path("test/user1/img1.jpg")]
        samples = _infer_samples(
            paths,
            dataset_index={},
            views_module=mock_views,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=True,
            distance_metric="cosine",
        )

        assert len(samples) == 1
        assert samples[0].match_username == "alice"
        assert samples[0].distance == 0.15
        assert samples[0].embedding_available is True


@patch("src.evaluation.face_recognition_eval._recognition_views")
@patch("src.evaluation.face_recognition_eval._resolve_image_paths")
@patch("src.evaluation.face_recognition_eval._load_dataset_index")
@patch("src.evaluation.face_recognition_eval._infer_samples")
@patch("src.evaluation.face_recognition_eval._save_artifacts")
def test_run_face_recognition_evaluation(
    mock_save, mock_infer, mock_load, mock_resolve, mock_views, sample_predictions
):
    mock_views_module = MagicMock()
    mock_views.return_value = mock_views_module

    mock_resolve.return_value = [Path("test.jpg")]
    mock_infer.return_value = sample_predictions
    mock_save.return_value = {"metrics": Path("metrics.json")}

    config = EvaluationConfig(threshold=0.3)
    summary = run_face_recognition_evaluation(config)

    assert summary.metrics is not None
    assert summary.threshold_sweep is not None
    assert len(summary.samples) == 4
    assert mock_save.called


def test_build_argument_parser():
    parser = build_argument_parser()
    args = parser.parse_args(["--threshold", "0.5", "--max-samples", "10"])
    assert args.threshold == 0.5
    assert args.max_samples == 10


@patch("django.setup")
@patch("src.evaluation.face_recognition_eval.run_face_recognition_evaluation")
def test_main(mock_run, mock_setup):
    mock_summary = MagicMock()
    mock_summary.artifact_paths = {"metrics": Path("test")}
    mock_run.return_value = mock_summary

    result = main(["--threshold", "0.5"])

    assert mock_setup.called
    assert mock_run.called
    assert result == mock_summary
