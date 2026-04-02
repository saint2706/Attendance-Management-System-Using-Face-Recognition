"""Tests for the evaluation pipeline utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation import UNKNOWN_LABEL
from src.evaluation.face_recognition_eval import (
    SampleEvaluation,
    compute_basic_metrics,
    compute_threshold_sweep,
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


def test_sample_evaluation_predicted_label():
    sample = SampleEvaluation(Path("dataset/alice/0.jpg"), "alice", "alice", 0.30, True)
    assert sample.predicted_label(0.35) == "alice"
    assert sample.predicted_label(0.20) == UNKNOWN_LABEL
    assert sample.to_row(0.35)["predicted_label"] == "alice"


def test_evaluation_config_defaults(monkeypatch):
    # Mock _recognition_views to return an object with DEFAULT_DISTANCE_THRESHOLD
    import types

    from src.evaluation.face_recognition_eval import EvaluationConfig

    mock_views = types.ModuleType("mock_views")
    mock_views.DEFAULT_DISTANCE_THRESHOLD = 0.4
    mock_views.TRAINING_DATASET_ROOT = "/tmp/dataset"

    from src.evaluation import face_recognition_eval

    monkeypatch.setattr(face_recognition_eval, "_recognition_views", lambda: mock_views)

    config = EvaluationConfig()

    # Reports dir should default to BASE_DIR / reports / evaluation
    assert config.reports_dir is not None
    assert str(config.reports_dir).endswith("reports/evaluation")
    assert config.threshold is not None
    assert config.threshold_values[0] == pytest.approx(0.2)


def test_evaluation_config_validation(monkeypatch):
    import types

    from src.evaluation.face_recognition_eval import EvaluationConfig

    mock_views = types.ModuleType("mock_views")
    mock_views.DEFAULT_DISTANCE_THRESHOLD = 0.4
    mock_views.TRAINING_DATASET_ROOT = "/tmp/dataset"

    from src.evaluation import face_recognition_eval

    monkeypatch.setattr(face_recognition_eval, "_recognition_views", lambda: mock_views)

    with pytest.raises(ValueError, match="threshold_step must be positive"):
        EvaluationConfig(threshold_step=-0.1)

    with pytest.raises(ValueError, match="threshold_stop must be >= threshold_start"):
        EvaluationConfig(threshold_start=0.5, threshold_stop=0.2)

    with pytest.raises(ValueError, match="limit_samples must be positive when provided"):
        EvaluationConfig(limit_samples=-5)


def test_calculate_far_frr_edge_cases():
    from src.evaluation.face_recognition_eval import _calculate_far_frr

    # Empty dataset
    far, frr = _calculate_far_frr([], [])
    assert far == 0.0
    assert frr == 0.0

    # No genuine cases
    y_true = [UNKNOWN_LABEL, UNKNOWN_LABEL]
    y_pred = [UNKNOWN_LABEL, "alice"]
    far, frr = _calculate_far_frr(y_true, y_pred)
    assert far == 0.5
    assert frr == 0.0

    # No impostor cases
    y_true = ["alice", "bob"]
    y_pred = ["alice", "carol"]
    far, frr = _calculate_far_frr(y_true, y_pred)
    assert far == 0.0
    assert frr == 0.5


def test_compute_basic_metrics_empty():
    with pytest.raises(ValueError, match="No samples provided for metric computation"):
        compute_basic_metrics([], threshold=0.5)


def test_resolve_image_paths(tmp_path, monkeypatch):
    from src.evaluation.face_recognition_eval import EvaluationConfig, _resolve_image_paths

    # Create mock dataset
    dataset_dir = tmp_path / "dataset"
    alice_dir = dataset_dir / "alice"
    alice_dir.mkdir(parents=True)
    (alice_dir / "1.jpg").touch()
    (alice_dir / "2.png").touch()
    (alice_dir / "ignore.txt").touch()

    # Default fallback to scanning
    import types

    mock_views = types.ModuleType("mock_views")
    mock_views.DEFAULT_DISTANCE_THRESHOLD = 0.4
    mock_views.TRAINING_DATASET_ROOT = str(dataset_dir)
    from src.evaluation import face_recognition_eval

    monkeypatch.setattr(face_recognition_eval, "_recognition_views", lambda: mock_views)

    config = EvaluationConfig(dataset_root=dataset_dir)
    paths = _resolve_image_paths(config)
    assert len(paths) == 2

    # Limit samples
    config_limited = EvaluationConfig(dataset_root=dataset_dir, limit_samples=1)
    paths_limited = _resolve_image_paths(config_limited)
    assert len(paths_limited) == 1

    # Split CSV
    csv_path = tmp_path / "split.csv"
    csv_path.write_text(
        "image_path,split\n" "alice/1.jpg,test\n" "alice/2.png,train\n" "alice/missing.jpg,test\n",
        encoding="utf-8",
    )
    config_csv = EvaluationConfig(dataset_root=dataset_dir, test_split_csv=csv_path)
    paths_csv = _resolve_image_paths(config_csv)
    assert len(paths_csv) == 1
    assert paths_csv[0].name == "1.jpg"

    # Empty dataset error
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    config_empty = EvaluationConfig(dataset_root=empty_dir)
    with pytest.raises(RuntimeError, match="No evaluation images found"):
        _resolve_image_paths(config_empty)


def test_save_artifacts(tmp_path, monkeypatch):
    import types

    from src.evaluation.face_recognition_eval import EvaluationConfig, _save_artifacts

    mock_views = types.ModuleType("mock_views")
    mock_views.DEFAULT_DISTANCE_THRESHOLD = 0.4
    mock_views.TRAINING_DATASET_ROOT = "/tmp/dataset"

    from src.evaluation import face_recognition_eval

    monkeypatch.setattr(face_recognition_eval, "_recognition_views", lambda: mock_views)

    config = EvaluationConfig(reports_dir=tmp_path)
    metrics = {
        "threshold": 0.5,
        "samples": 1,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "far": 0.0,
        "frr": 0.0,
        "unknown_predictions": 0,
        "samples_without_embedding": 0,
    }
    sample = SampleEvaluation(Path("dataset/alice/0.jpg"), "alice", "alice", 0.30, True)

    paths = _save_artifacts(
        config=config,
        metrics=metrics,
        samples=[sample],
        y_true=["alice"],
        y_pred=["alice"],
        labels=["alice", UNKNOWN_LABEL],
        threshold_sweep=[metrics],
    )

    assert paths["metrics"].exists()
    assert paths["samples"].exists()
    assert paths["confusion_csv"].exists()
    assert paths["confusion_png"].exists()
    assert paths["threshold_csv"].exists()
    assert paths["threshold_png"].exists()
