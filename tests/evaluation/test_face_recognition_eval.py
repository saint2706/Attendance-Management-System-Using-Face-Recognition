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
        SampleEvaluation(
            Path("dataset/intruder/3.jpg"), UNKNOWN_LABEL, "alice", 0.25, True
        ),
    ]


def test_compute_basic_metrics(sample_predictions: list[SampleEvaluation]) -> None:
    metrics, y_true, y_pred, labels = compute_basic_metrics(
        sample_predictions, threshold=0.35
    )

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
