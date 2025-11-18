"""Unit tests for the lightweight face-recognition pipeline utilities."""

import math

import numpy as np
import pytest

from recognition import pipeline


def test_extract_embedding_normalises_deepface_payload() -> None:
    """Lists of dictionaries from DeepFace should be coerced into numpy vectors."""

    embedding, facial_area = pipeline.extract_embedding(
        [
            {
                "embedding": ["1.0", 0.5, 0],
                "facial_area": {"x": 1, "y": 2, "w": 3, "h": 4},
            }
        ]
    )

    assert facial_area == {"x": 1, "y": 2, "w": 3, "h": 4}
    assert isinstance(embedding, np.ndarray)
    np.testing.assert_allclose(embedding, np.array([1.0, 0.5, 0.0], dtype=float))


def test_extract_embedding_rejects_invalid_payload() -> None:
    """Non-numeric payloads must be rejected to avoid propagating NaNs downstream."""

    embedding, facial_area = pipeline.extract_embedding(
        [{"embedding": ["not-a-number"], "facial_area": {"x": 1}}]
    )

    assert embedding is None
    assert facial_area == {"x": 1}


@pytest.mark.parametrize(
    "metric,expected",
    [
        ("euclidean_l2", pytest.approx(math.sqrt(2))),
        ("manhattan", 2.0),
    ],
)
def test_calculate_embedding_distance_supported_metrics(metric: str, expected) -> None:
    """Distances should be computed deterministically across supported metrics."""

    source = np.array([1.0, 0.0], dtype=float)
    candidate = np.array([0.0, 1.0], dtype=float)

    assert pipeline.calculate_embedding_distance(candidate, source, metric) == expected


def test_calculate_embedding_distance_cosine_handles_zero_vectors() -> None:
    """Cosine similarity should yield ``None`` when either vector has zero magnitude."""

    zero_vector = np.zeros(3, dtype=float)
    other_vector = np.array([1.0, 2.0, 3.0], dtype=float)

    assert pipeline.calculate_embedding_distance(zero_vector, other_vector, "cosine") is None


def test_find_closest_dataset_match_returns_best_candidate() -> None:
    """The best candidate should be returned when multiple embeddings are available."""

    probe = np.array([0.9, 0.1], dtype=float)
    dataset = [
        {
            "username": "alice",
            "identity": "dataset/alice/1.jpg",
            "embedding": np.array([0.0, 1.0]),
        },
        {
            "username": "bob",
            "identity": "dataset/bob/1.jpg",
            "embedding": np.array([1.0, 0.0]),
        },
    ]

    username, distance, identity = pipeline.find_closest_dataset_match(
        probe, dataset, "euclidean_l2"
    )

    assert username == "bob"
    assert identity == "dataset/bob/1.jpg"
    assert distance < 1.0


@pytest.mark.parametrize(
    "distance,threshold,expected",
    [
        (0.25, 0.3, True),
        (0.45, 0.3, False),
        (None, 0.3, False),
        (float("nan"), 0.3, False),
    ],
)
def test_is_within_distance_threshold_handles_edge_cases(
    distance: float | None, threshold: float, expected: bool
) -> None:
    """Distance threshold helper should defensively handle non-finite inputs."""

    assert pipeline.is_within_distance_threshold(distance, threshold) is expected
