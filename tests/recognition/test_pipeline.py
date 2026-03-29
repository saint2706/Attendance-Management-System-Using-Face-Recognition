"""Unit tests for the lightweight face-recognition pipeline utilities."""

import math
from typing import Any

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


def test_extract_embedding_2d_ndarray() -> None:
    """2D numpy arrays should extract the first row."""
    payload = np.array([[1.0, 2.0], [3.0, 4.0]])
    embedding, area = pipeline.extract_embedding(payload)
    assert area is None
    assert isinstance(embedding, np.ndarray)
    np.testing.assert_allclose(embedding, np.array([1.0, 2.0], dtype=float))


def test_extract_embedding_list_of_lists() -> None:
    """1D lists of raw values should be extracted."""
    payload = [[1.0, 2.0, 3.0]]
    embedding, area = pipeline.extract_embedding(payload)
    assert area is None
    assert isinstance(embedding, np.ndarray)
    np.testing.assert_allclose(embedding, np.array([1.0, 2.0, 3.0], dtype=float))


def test_extract_embedding_dict_payload() -> None:
    """Direct dict payload should be extracted."""
    payload = {
        "embedding": [0.5, 0.5],
        "facial_area": {"x": 0, "y": 0, "w": 10, "h": 10},
    }
    embedding, area = pipeline.extract_embedding(payload)
    assert area == {"x": 0, "y": 0, "w": 10, "h": 10}
    assert isinstance(embedding, np.ndarray)
    np.testing.assert_allclose(embedding, np.array([0.5, 0.5], dtype=float))


def test_extract_embedding_invalid_size() -> None:
    """Empty valid representations should return None."""
    embedding, area = pipeline.extract_embedding([])
    assert embedding is None
    assert area is None

    embedding, area = pipeline.extract_embedding(np.array([[]]))
    assert embedding is None

    embedding, area = pipeline.extract_embedding({"embedding": []})
    assert embedding is None

    # Missing coverage: np.ndarray where ndim is NOT 2
    embedding, area = pipeline.extract_embedding(np.array([1.0, 2.0]))
    assert embedding is None

    # Missing coverage: list where first element is NOT dict/list/tuple/ndarray
    embedding, area = pipeline.extract_embedding(["invalid"])
    assert embedding is None


def test_extract_embedding_rejects_invalid_payload() -> None:
    """Non-numeric payloads must be rejected to avoid propagating NaNs downstream."""

    embedding, facial_area = pipeline.extract_embedding(
        [{"embedding": ["not-a-number"], "facial_area": {"x": 1}}]
    )

    assert embedding is None
    assert facial_area == {"x": 1}


def test_extract_all_embeddings_dict_payload() -> None:
    """Single representation as dict should return a list with one item."""
    payload = {
        "embedding": [0.5, 0.5],
        "facial_area": {"x": 0, "y": 0, "w": 10, "h": 10},
    }
    results = pipeline.extract_all_embeddings(payload)
    assert len(results) == 1
    assert isinstance(results[0][0], np.ndarray)
    np.testing.assert_allclose(results[0][0], np.array([0.5, 0.5], dtype=float))
    assert results[0][1] == {"x": 0, "y": 0, "w": 10, "h": 10}


def test_extract_all_embeddings_list_of_dicts() -> None:
    """List of representations should return a list with multiple items."""
    payload = [
        {
            "embedding": [0.1, 0.2],
            "facial_area": {"x": 1, "y": 1, "w": 10, "h": 10},
        },
        {
            "embedding": [0.3, 0.4],
            "facial_area": {"x": 2, "y": 2, "w": 20, "h": 20},
        },
    ]
    results = pipeline.extract_all_embeddings(payload)
    assert len(results) == 2
    np.testing.assert_allclose(results[0][0], np.array([0.1, 0.2], dtype=float))
    assert results[0][1] == {"x": 1, "y": 1, "w": 10, "h": 10}
    np.testing.assert_allclose(results[1][0], np.array([0.3, 0.4], dtype=float))
    assert results[1][1] == {"x": 2, "y": 2, "w": 20, "h": 20}


def test_extract_all_embeddings_list_of_lists() -> None:
    """List of legacy raw value formats should return multiple items."""
    # extract_all_embeddings iterates through the list.
    # Each item must be valid for extract_embedding.
    # If the item is a list (e.g., [0.1, 0.2]), extract_embedding sees it as list of length 2
    # but the logic there expects representations as either a list of dicts or list of lists.
    # Passing [[[0.1, 0.2]], [[0.3, 0.4]]] allows extract_embedding to parse each internal list properly.
    payload = [[[0.1, 0.2]], [[0.3, 0.4]]]
    results = pipeline.extract_all_embeddings(payload)
    assert len(results) == 2
    np.testing.assert_allclose(results[0][0], np.array([0.1, 0.2], dtype=float))
    assert results[0][1] is None
    np.testing.assert_allclose(results[1][0], np.array([0.3, 0.4], dtype=float))
    assert results[1][1] is None


def test_extract_all_embeddings_2d_ndarray() -> None:
    """2D numpy arrays should return a list with one item."""
    payload = np.array([[1.0, 2.0], [3.0, 4.0]])
    results = pipeline.extract_all_embeddings(payload)
    assert len(results) == 1
    np.testing.assert_allclose(results[0][0], np.array([1.0, 2.0], dtype=float))
    assert results[0][1] is None


def test_extract_all_embeddings_no_valid_embeddings() -> None:
    """Empty valid representations should return an empty list."""
    assert pipeline.extract_all_embeddings([]) == []
    assert pipeline.extract_all_embeddings(np.array([[]])) == []
    assert pipeline.extract_all_embeddings({"embedding": []}) == []
    assert pipeline.extract_all_embeddings([{"facial_area": {}}]) == []
    assert pipeline.extract_all_embeddings([[]]) == []
    assert pipeline.extract_all_embeddings("invalid-type") == []
    assert pipeline.extract_all_embeddings([{"invalid": "format"}]) == []
    assert pipeline.extract_all_embeddings(["invalid_string"]) == []


@pytest.mark.parametrize(
    "metric,expected",
    [
        ("euclidean_l2", pytest.approx(math.sqrt(2))),
        ("manhattan", 2.0),
        ("cosine", 1.0),  # (1 - dot(A,B)/(||A||*||B||)) => 1 - 0 = 1.0
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


def test_calculate_embedding_distance_fallback_metric() -> None:
    """Unknown metrics should fallback to Euclidean L2 distance."""
    source = np.array([1.0, 0.0], dtype=float)
    candidate = np.array([0.0, 1.0], dtype=float)

    expected = float(np.linalg.norm(candidate - source))
    assert pipeline.calculate_embedding_distance(
        candidate, source, "unknown_metric"
    ) == pytest.approx(expected)


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

    result = pipeline.find_closest_dataset_match(probe, dataset, "euclidean_l2")
    assert result is not None
    username, distance, identity = result

    assert username == "bob"
    assert identity == "dataset/bob/1.jpg"
    assert distance < 1.0


def test_find_closest_dataset_match_empty_probe() -> None:
    """Empty probe embedding should return None."""
    probe = np.array([])
    dataset = [{"embedding": np.array([1.0])}]
    assert pipeline.find_closest_dataset_match(probe, dataset, "euclidean_l2") is None


def test_find_closest_dataset_match_empty_dataset() -> None:
    """Empty dataset should return None."""
    probe = np.array([1.0])
    assert pipeline.find_closest_dataset_match(probe, [], "euclidean_l2") is None


def test_find_closest_dataset_match_invalid_entries() -> None:
    """Dataset entries without 'embedding' or with non-ndarray embeddings should be ignored."""
    probe = np.array([0.9, 0.1], dtype=float)
    dataset: list[Any] = [
        {"username": "no_embedding"},
        {"username": "bad_embedding", "embedding": [1.0, 0.0]},  # list, not ndarray
        "not_a_mapping",  # completely invalid entry
        {
            "username": "bob",
            "identity": "dataset/bob/1.jpg",
            "embedding": np.array([1.0, 0.0]),
        },
    ]

    result = pipeline.find_closest_dataset_match(probe, dataset, "euclidean_l2")
    assert result is not None
    username, distance, identity = result
    assert username == "bob"


def test_find_closest_dataset_match_calculation_fails() -> None:
    """When calculate_embedding_distance returns None, the candidate should be skipped."""
    probe = np.array([0.9, 0.1], dtype=float)
    dataset: list[Any] = [
        {
            "username": "alice",
            "embedding": np.zeros(2, dtype=float),  # this will cause cosine distance to return None
        },
        {
            "username": "bob",
            "embedding": np.array([1.0, 0.0]),
        },
        {
            "username": "charlie",
            "embedding": np.array([0.9, 0.1]),  # An exact match to ensure we take the < path
        },
    ]

    result = pipeline.find_closest_dataset_match(probe, dataset, "cosine")
    assert result is not None
    username, distance, identity = result
    assert username == "charlie"


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


def test_find_closest_match_faiss_invalid_index() -> None:
    """Invalid FAISS index type should return None."""
    probe = np.array([0.9, 0.1], dtype=float)
    assert pipeline.find_closest_match_faiss(probe, None) is None  # type: ignore
    assert pipeline.find_closest_match_faiss(probe, "not_faiss_index") is None  # type: ignore


def test_find_closest_match_faiss_empty_probe() -> None:
    """Empty probe embedding should return None."""
    from recognition.faiss_index import FAISSIndex

    faiss_index = FAISSIndex(dimension=2)
    probe = np.array([])
    assert pipeline.find_closest_match_faiss(probe, faiss_index) is None


def test_find_closest_match_faiss_returns_match() -> None:
    """Valid search that returns a match."""
    from recognition.faiss_index import FAISSIndex

    faiss_index = FAISSIndex(dimension=2)

    # Add a mock entry to the index manually to ensure search works
    faiss_index.add_embeddings(np.array([[1.0, 0.0]], dtype=np.float32), ["bob"])

    probe = np.array([1.0, 0.0], dtype=float)
    result = pipeline.find_closest_match_faiss(probe, faiss_index)

    assert result is not None
    username, distance = result
    assert username == "bob"
    assert distance == pytest.approx(0.0)


def test_find_closest_match_faiss_returns_none() -> None:
    """Search that returns None (e.g., when the index is empty)."""
    from recognition.faiss_index import FAISSIndex

    faiss_index = FAISSIndex(dimension=2)

    probe = np.array([1.0, 0.0], dtype=float)
    assert pipeline.find_closest_match_faiss(probe, faiss_index) is None
