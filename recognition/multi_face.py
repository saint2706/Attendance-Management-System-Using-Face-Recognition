"""Multi-face detection utilities for group recognition scenarios.

This module provides helper functions for processing multiple faces in a single frame,
supporting group check-in workflows while maintaining backward compatibility with
single-face detection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from django.conf import settings

from .pipeline import (
    extract_all_embeddings,
    extract_embedding,
    find_closest_dataset_match,
    is_within_distance_threshold,
)

logger = logging.getLogger(__name__)


def is_multi_face_enabled() -> bool:
    """Check if multi-face detection mode is enabled.
    
    Returns:
        True if multi-face mode is enabled, False for single-person mode (default).
    """
    return getattr(settings, "RECOGNITION_MULTI_FACE_ENABLED", False)


def get_max_faces_limit() -> int:
    """Get the maximum number of faces to process per frame.
    
    Returns:
        Maximum faces count (default: 5).
    """
    return getattr(settings, "RECOGNITION_MAX_FACES_PER_FRAME", 5)


def filter_faces_by_size(
    face_embeddings: List[Tuple[np.ndarray, Optional[Dict[str, int]]]]
) -> List[Tuple[np.ndarray, Optional[Dict[str, int]]]]:
    """Filter out faces that are too small based on configured threshold.
    
    Args:
        face_embeddings: List of (embedding, facial_area) tuples.
    
    Returns:
        Filtered list with only faces meeting minimum size requirement.
    """
    min_size = getattr(settings, "RECOGNITION_MULTI_FACE_MIN_SIZE", 50)
    
    filtered = []
    for embedding, facial_area in face_embeddings:
        if facial_area is None:
            # No size info, keep it
            filtered.append((embedding, facial_area))
            continue
        
        # Calculate face dimensions
        width = facial_area.get("w", 0)
        height = facial_area.get("h", 0)
        
        # Check if face meets minimum size
        if width >= min_size and height >= min_size:
            filtered.append((embedding, facial_area))
        else:
            logger.debug(f"Filtered out small face: {width}x{height}px (min: {min_size}px)")
    
    return filtered


def process_multi_face_recognition(
    representations,
    dataset_index: List[Dict[str, Any]],
    distance_metric: str,
    distance_threshold: float,
) -> Dict[str, Any]:
    """Process multiple faces from representations and find matches.
    
    Args:
        representations: Raw DeepFace.represent() output.
        dataset_index: List of dataset entries with embeddings.
        distance_metric: Distance metric to use (cosine, euclidean, etc.).
        distance_threshold: Threshold for recognition.
    
    Returns:
        Dictionary with multi-face results:
        {
            "faces": [{"embedding": [...], "match": {...}, "facial_area": {...}}, ...],
            "count": int,
            "mode": "multi"
        }
    """
    # Extract all embeddings
    face_embeddings = extract_all_embeddings(representations)
    
    if not face_embeddings:
        return {"faces": [], "count": 0, "mode": "multi", "error": "No faces detected"}
    
    # Filter by size
    face_embeddings = filter_faces_by_size(face_embeddings)
    
    # Apply max faces limit
    max_faces = get_max_faces_limit()
    if len(face_embeddings) > max_faces:
        logger.info(f"Limiting faces from {len(face_embeddings)} to {max_faces}")
        face_embeddings = face_embeddings[:max_faces]
    
    # Process each face
    results = []
    for embedding_vector, facial_area in face_embeddings:
        # Find match for this face
        match = find_closest_dataset_match(embedding_vector, dataset_index, distance_metric)
        
        face_result: Dict[str, Any] = {
            "embedding": embedding_vector.tolist(),
            "facial_area": facial_area,
        }
        
        if match is None:
            face_result.update({
                "recognized": False,
                "match": None,
            })
        else:
            username, distance_value, identity_path = match
            recognized = is_within_distance_threshold(distance_value, distance_threshold)
            
            face_result.update({
                "recognized": recognized,
                "match": {
                    "username": username,
                    "distance": float(distance_value),
                    "identity": identity_path,
                    "threshold": float(distance_threshold),
                },
            })
        
        results.append(face_result)
    
    return {
        "faces": results,
        "count": len(results),
        "mode": "multi",
        "threshold": float(distance_threshold),
        "distance_metric": distance_metric,
    }


def process_single_face_recognition(
    representations,
    dataset_index: List[Dict[str, Any]],
    distance_metric: str,
    distance_threshold: float,
) -> Dict[str, Any]:
    """Process single face (standard/legacy mode).
    
    Args:
        representations: Raw DeepFace.represent() output.
        dataset_index: List of dataset entries with embeddings.
        distance_metric: Distance metric to use.
        distance_threshold: Threshold for recognition.
    
    Returns:
        Dictionary with single-face results (current format).
    """
    extracted_embedding, facial_area = extract_embedding(representations)
    
    if extracted_embedding is None:
        return {
            "recognized": False,
            "error": "No face embedding could be extracted",
            "mode": "single",
        }
    
    try:
        embedding_vector = np.array(extracted_embedding, dtype=float)
    except (TypeError, ValueError):
        return {
            "recognized": False,
            "error": "Invalid embedding generated",
            "mode": "single",
        }
    
    match = find_closest_dataset_match(embedding_vector, dataset_index, distance_metric)
    
    response: Dict[str, Any] = {
        "recognized": False,
        "threshold": float(distance_threshold),
        "distance_metric": distance_metric,
        "mode": "single",
        "facial_area": facial_area,
    }
    
    if match is None:
        return response
    
    username, distance_value, identity_path = match
    response.update({
        "distance": float(distance_value),
        "identity": identity_path,
    })
    
    if username:
        response["username"] = username
    
    recognized = is_within_distance_threshold(distance_value, distance_threshold)
    response["recognized"] = recognized
    
    return response


def process_face_recognition(
    representations,
    dataset_index: List[Dict[str, Any]],
    distance_metric: str,
    distance_threshold: float,
) -> Dict[str, Any]:
    """Process face recognition in either single or multi-face mode.
    
    Automatically selects the appropriate mode based on RECOGNITION_MULTI_FACE_ENABLED setting.
    
    Args:
        representations: Raw DeepFace.represent() output.
        dataset_index: List of dataset entries with embeddings.
        distance_metric: Distance metric to use.
        distance_threshold: Threshold for recognition.
    
    Returns:
        Recognition results dictionary. Format depends on mode:
        - Single mode: {"recognized": bool, "username": str, "distance": float, ...}
        - Multi mode: {"faces": [...], "count": int, ...}
    """
    if is_multi_face_enabled():
        logger.info("Processing in multi-face mode")
        return process_multi_face_recognition(
            representations, dataset_index, distance_metric, distance_threshold
        )
    else:
        logger.debug("Processing in single-face mode (default)")
        return process_single_face_recognition(
            representations, dataset_index, distance_metric, distance_threshold
        )


__all__ = [
    "is_multi_face_enabled",
    "get_max_faces_limit",
    "process_face_recognition",
    "process_multi_face_recognition",
    "process_single_face_recognition",
]
