"""Core face recognition pipeline utilities.

This module centralises the pure functions that power the face recognition
workflow so they can be exercised in isolation.  By extracting the embedding
normalisation, dataset matching, and distance thresholding logic we can cover
the most critical behaviour with fast unit tests that use synthetic vectors
instead of the heavy DeepFace integration.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_embedding(
    representations,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]:
    """Normalise DeepFace representations into a single embedding vector.

    Args:
        representations: The raw payload returned by ``DeepFace.represent``.

    Returns:
        A tuple containing the embedding as a NumPy vector (or ``None`` when no
        usable embedding is available) and the optional facial area metadata.
    """

    embedding_vector: Optional[Sequence[float]] = None
    facial_area: Optional[Dict[str, int]] = None

    if isinstance(representations, np.ndarray):
        if representations.ndim == 2 and len(representations) > 0:
            embedding_vector = representations[0]
    elif isinstance(representations, list) and representations:
        first = representations[0]
        if isinstance(first, dict):
            embedding_vector = first.get("embedding")
            area = first.get("facial_area")
            facial_area = area if isinstance(area, dict) else None
        elif isinstance(first, (list, tuple, np.ndarray)):
            embedding_vector = first
    elif isinstance(representations, dict) and "embedding" in representations:
        embedding_vector = representations.get("embedding")
        area = representations.get("facial_area")
        facial_area = area if isinstance(area, dict) else None

    if embedding_vector is None:
        return None, facial_area

    try:
        normalized = np.array([float(value) for value in embedding_vector], dtype=float)
    except (TypeError, ValueError):
        logger.debug(
            "Unable to coerce embedding values to floats: %r", embedding_vector
        )
        return None, facial_area

    if normalized.size == 0:
        return None, facial_area

    return normalized, facial_area


def calculate_embedding_distance(
    candidate: np.ndarray, embedding_vector: np.ndarray, metric: str
) -> Optional[float]:
    """Compute a distance score between two embeddings.

    The function supports cosine, Euclidean (L2), and Manhattan (L1) metrics.
    When the metric cannot be evaluated—for example because one of the vectors
    has zero magnitude for cosine similarity—it returns ``None`` so callers can
    gracefully skip the candidate.
    """

    metric = metric.lower()
    try:
        if metric in {"cosine", "cosine_similarity"}:
            candidate_norm = float(np.linalg.norm(candidate))
            vector_norm = float(np.linalg.norm(embedding_vector))
            if candidate_norm == 0.0 or vector_norm == 0.0:
                return None
            similarity = float(
                np.dot(candidate, embedding_vector) / (candidate_norm * vector_norm)
            )
            return 1.0 - similarity

        if metric in {"euclidean", "euclidean_l2", "l2"}:
            return float(np.linalg.norm(candidate - embedding_vector))

        if metric in {"manhattan", "l1", "euclidean_l1"}:
            return float(np.sum(np.abs(candidate - embedding_vector)))

    except Exception as exc:  # pragma: no cover - defensive programming
        logger.debug("Failed to compute %s distance: %s", metric, exc)
        return None

    # Default to Euclidean L2 if the metric is unrecognised.
    try:
        return float(np.linalg.norm(candidate - embedding_vector))
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.debug("Failed to compute fallback distance: %s", exc)
        return None


def find_closest_dataset_match(
    embedding_vector: np.ndarray,
    dataset_index: Iterable[Mapping[str, object]],
    metric: str,
) -> Optional[Tuple[str, float, str]]:
    """Return the nearest neighbour match for the provided embedding.

    Args:
        embedding_vector: The embedding produced for the probe face.
        dataset_index: Iterable of dataset entries each containing an
            ``embedding`` ``numpy.ndarray`` along with metadata.
        metric: Name of the distance metric used when computing similarity.

    Returns:
        A tuple containing the matched username, the distance value, and the
        stored identity path. ``None`` is returned when no candidate can be
        evaluated or the dataset is empty.
    """

    if embedding_vector.size == 0:
        return None

    best_entry: Optional[Mapping[str, object]] = None
    best_distance: Optional[float] = None

    for entry in dataset_index:
        candidate = entry.get("embedding") if isinstance(entry, Mapping) else None
        if not isinstance(candidate, np.ndarray):
            continue

        distance = calculate_embedding_distance(candidate, embedding_vector, metric)
        if distance is None:
            continue

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_entry = entry

    if best_entry is None or best_distance is None:
        return None

    username = str(best_entry.get("username")) if best_entry.get("username") else ""
    identity = str(best_entry.get("identity")) if best_entry.get("identity") else ""
    return username, best_distance, identity


def is_within_distance_threshold(distance: Optional[float], threshold: float) -> bool:
    """Return ``True`` when the distance does not exceed the configured threshold."""

    if distance is None:
        return False

    if math.isnan(distance):
        return False

    return bool(distance <= threshold)
