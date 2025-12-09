"""Embedding cache utilities for face recognition.

This module provides a caching layer for face embeddings to reduce redundant
computation during recognition. It uses Django's cache framework with a
dedicated 'embeddings' cache backend.

When Redis is configured (via REDIS_URL), embeddings are cached in Redis
for sharing across multiple worker processes. In development, LocMemCache
is used as a fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from django.conf import settings
from django.core.cache import caches

if TYPE_CHECKING:
    from django.core.cache.backends.base import BaseCache

logger = logging.getLogger(__name__)

# Cache key prefixes
_EMBEDDING_KEY_PREFIX = "emb"
_DATASET_INDEX_KEY = "dataset_index"
_DATASET_HASH_KEY = "dataset_hash"


def _get_cache() -> "BaseCache":
    """Return the embeddings cache backend."""
    try:
        return caches["embeddings"]
    except Exception:
        # Fallback to default cache if embeddings cache is not configured
        return caches["default"]


def _embedding_key(username: str, image_hash: Optional[str] = None) -> str:
    """Generate a cache key for a user's embedding."""
    if image_hash:
        return f"{_EMBEDDING_KEY_PREFIX}:{username}:{image_hash}"
    return f"{_EMBEDDING_KEY_PREFIX}:{username}"


def _serialize_embedding(embedding: np.ndarray) -> str:
    """Serialize a numpy embedding to JSON string."""
    return json.dumps(embedding.tolist())


def _deserialize_embedding(data: str) -> Optional[np.ndarray]:
    """Deserialize a JSON string to numpy embedding."""
    try:
        return np.array(json.loads(data), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Failed to deserialize embedding: %s", exc)
        return None


def get_cached_embedding(
    username: str,
    image_hash: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Retrieve a cached embedding for a user.

    Args:
        username: The username to look up.
        image_hash: Optional hash of the specific image.

    Returns:
        The cached embedding as a numpy array, or None if not found.
    """
    cache = _get_cache()
    key = _embedding_key(username, image_hash)

    try:
        data = cache.get(key)
        if data is None:
            return None
        return _deserialize_embedding(data)
    except Exception as exc:
        logger.debug("Cache get failed for %s: %s", key, exc)
        return None


def set_cached_embedding(
    username: str,
    embedding: np.ndarray,
    image_hash: Optional[str] = None,
    ttl: Optional[int] = None,
) -> bool:
    """Cache an embedding for a user.

    Args:
        username: The username to cache for.
        embedding: The embedding vector to cache.
        image_hash: Optional hash of the specific image.
        ttl: Time-to-live in seconds (defaults to EMBEDDING_CACHE_TTL).

    Returns:
        True if caching succeeded, False otherwise.
    """
    cache = _get_cache()
    key = _embedding_key(username, image_hash)

    if ttl is None:
        ttl = getattr(settings, "EMBEDDING_CACHE_TTL", 3600)

    try:
        serialized = _serialize_embedding(embedding)
        cache.set(key, serialized, timeout=ttl)
        return True
    except Exception as exc:
        logger.warning("Cache set failed for %s: %s", key, exc)
        return False


def invalidate_user_embeddings(username: str) -> bool:
    """Invalidate all cached embeddings for a user.

    Args:
        username: The username whose embeddings should be invalidated.

    Returns:
        True if invalidation succeeded, False otherwise.
    """
    cache = _get_cache()
    key_pattern = _embedding_key(username)

    try:
        # Delete the base key
        cache.delete(key_pattern)

        # For Redis, we could use pattern matching deletion
        # For LocMemCache, we rely on TTL expiration for image-specific keys
        logger.info("Invalidated embeddings cache for user: %s", username)
        return True
    except Exception as exc:
        logger.warning("Cache invalidation failed for %s: %s", username, exc)
        return False


def invalidate_all_embeddings() -> bool:
    """Invalidate all cached embeddings.

    This should be called after model retraining to ensure fresh embeddings.

    Returns:
        True if invalidation succeeded, False otherwise.
    """
    cache = _get_cache()

    try:
        cache.clear()
        logger.info("Cleared all embeddings cache")
        return True
    except Exception as exc:
        logger.warning("Failed to clear embeddings cache: %s", exc)
        return False


def get_cached_dataset_index() -> Optional[list]:
    """Retrieve the cached dataset embedding index.

    Returns:
        The cached dataset index, or None if not found.
    """
    cache = _get_cache()

    try:
        data = cache.get(_DATASET_INDEX_KEY)
        if data is None:
            return None

        # Deserialize embeddings in the index
        index = json.loads(data)
        for entry in index:
            if "embedding" in entry and entry["embedding"]:
                entry["embedding"] = np.array(entry["embedding"], dtype=float)
        return index
    except Exception as exc:
        logger.debug("Dataset index cache get failed: %s", exc)
        return None


def set_cached_dataset_index(
    index: list,
    dataset_hash: Optional[str] = None,
    ttl: Optional[int] = None,
) -> bool:
    """Cache the dataset embedding index.

    Args:
        index: The dataset index to cache.
        dataset_hash: Optional hash of the dataset for invalidation.
        ttl: Time-to-live in seconds.

    Returns:
        True if caching succeeded, False otherwise.
    """
    cache = _get_cache()

    if ttl is None:
        ttl = getattr(settings, "EMBEDDING_CACHE_TTL", 3600)

    try:
        # Serialize embeddings in the index
        serializable_index = []
        for entry in index:
            entry_copy = dict(entry)
            if "embedding" in entry_copy and isinstance(entry_copy["embedding"], np.ndarray):
                entry_copy["embedding"] = entry_copy["embedding"].tolist()
            serializable_index.append(entry_copy)

        cache.set(_DATASET_INDEX_KEY, json.dumps(serializable_index), timeout=ttl)

        if dataset_hash:
            cache.set(_DATASET_HASH_KEY, dataset_hash, timeout=ttl)

        return True
    except Exception as exc:
        logger.warning("Dataset index cache set failed: %s", exc)
        return False


def get_dataset_hash() -> Optional[str]:
    """Get the hash of the currently cached dataset.

    Returns:
        The dataset hash, or None if not cached.
    """
    cache = _get_cache()
    try:
        return cache.get(_DATASET_HASH_KEY)
    except Exception:
        return None


def compute_dataset_hash(dataset_path: str) -> str:
    """Compute a hash of the dataset for cache invalidation.

    Args:
        dataset_path: Path to the dataset directory.

    Returns:
        A hash string representing the dataset state.
    """
    import os
    from pathlib import Path

    hasher = hashlib.md5()
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists():
        return "empty"

    try:
        # Hash based on file paths and modification times
        for root, _, files in os.walk(dataset_dir):
            for filename in sorted(files):
                filepath = Path(root) / filename
                stat = filepath.stat()
                hasher.update(f"{filepath}:{stat.st_mtime}:{stat.st_size}".encode())
    except OSError as exc:
        logger.warning("Failed to compute dataset hash: %s", exc)
        return "error"

    return hasher.hexdigest()
