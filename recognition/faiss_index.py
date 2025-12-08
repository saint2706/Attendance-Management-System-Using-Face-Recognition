"""FAISS vector database integration for face recognition.

This module provides a FAISSIndex class that wraps FAISS for efficient
approximate nearest-neighbour search of face embeddings. It handles:

- Automatic index type selection based on dataset size
- Encrypted persistence matching existing security patterns
- Thread-safe search operations
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Optional, Sequence

import faiss
import numpy as np

from src.common import decrypt_bytes, encrypt_bytes

logger = logging.getLogger(__name__)

# Threshold for switching from exact (IndexFlatL2) to approximate (IndexIVFFlat) search
APPROXIMATE_SEARCH_THRESHOLD = 1000

# Default embedding dimension for Facenet model
DEFAULT_EMBEDDING_DIMENSION = 128


class FAISSIndex:
    """Manage FAISS index for face embedding similarity search.

    The index automatically selects the appropriate FAISS index type:
    - IndexFlatL2 for small datasets (< 1000 embeddings) - exact search
    - IndexIVFFlat for large datasets (>= 1000 embeddings) - approximate search

    All persistence operations use encryption to protect facial data.

    Example:
        >>> index = FAISSIndex(dimension=128)
        >>> embeddings = np.random.randn(100, 128).astype(np.float32)
        >>> labels = [f"user_{i}" for i in range(100)]
        >>> index.add_embeddings(embeddings, labels)
        >>> results = index.search(query_embedding, k=3)
        >>> for label, distance in results:
        ...     print(f"{label}: {distance:.4f}")
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION):
        """Initialize an empty FAISS index.

        Args:
            dimension: The embedding vector dimension (default 128 for Facenet).
        """
        self.dimension = dimension
        self._index: Optional[faiss.Index] = None
        self._labels: list[str] = []
        self._is_trained = False

    @property
    def size(self) -> int:
        """Return the number of embeddings in the index."""
        return len(self._labels)

    @property
    def is_empty(self) -> bool:
        """Return True if the index contains no embeddings."""
        return self.size == 0

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Sequence[str],
    ) -> None:
        """Add embeddings with associated identity labels.

        Args:
            embeddings: Array of shape (n_samples, dimension) containing
                the face embeddings to add.
            labels: Sequence of identity labels corresponding to each embedding.
                Must have the same length as embeddings.

        Raises:
            ValueError: If embeddings and labels have mismatched lengths,
                or if embedding dimension doesn't match index dimension.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")

        if embeddings.shape[0] != len(labels):
            raise ValueError(
                f"Embeddings count ({embeddings.shape[0]}) must match "
                f"labels count ({len(labels)})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) must match "
                f"index dimension ({self.dimension})"
            )

        # Ensure float32 for FAISS
        embeddings = embeddings.astype(np.float32)

        # Rebuild index with all data (including existing)
        total_embeddings = embeddings
        total_labels = list(labels)

        if self._index is not None and not self.is_empty:
            # Retrieve existing embeddings
            existing = self._get_all_embeddings()
            if existing is not None and existing.shape[0] > 0:
                total_embeddings = np.vstack([existing, embeddings])
                total_labels = self._labels + list(labels)

        self._build_index(total_embeddings, total_labels)

    def _build_index(self, embeddings: np.ndarray, labels: list[str]) -> None:
        """Build the appropriate FAISS index for the given embeddings.

        Selects IndexFlatL2 for small datasets or IndexIVFFlat for larger ones.
        """
        n_samples = embeddings.shape[0]
        embeddings = embeddings.astype(np.float32)

        if n_samples < APPROXIMATE_SEARCH_THRESHOLD:
            # Use exact search for small datasets
            self._index = faiss.IndexFlatL2(self.dimension)
            self._index.add(embeddings)
            logger.debug(
                "Built IndexFlatL2 with %d embeddings (exact search)",
                n_samples,
            )
        else:
            # Use approximate search with IVF for larger datasets
            # nlist (number of clusters) = sqrt(n_samples) is a common heuristic
            nlist = max(int(math.sqrt(n_samples)), 4)

            # Create quantizer and IVF index
            quantizer = faiss.IndexFlatL2(self.dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

            # Train the index on the embeddings
            self._index.train(embeddings)
            self._index.add(embeddings)

            # Set nprobe for search (number of clusters to visit)
            # Higher nprobe = more accurate but slower
            self._index.nprobe = min(nlist, max(1, nlist // 4))

            logger.debug(
                "Built IndexIVFFlat with %d embeddings, nlist=%d, nprobe=%d",
                n_samples,
                nlist,
                self._index.nprobe,
            )

        self._labels = labels
        self._is_trained = True

    def _get_all_embeddings(self) -> Optional[np.ndarray]:
        """Retrieve all embeddings from the current index."""
        if self._index is None or self.is_empty:
            return None

        # Use reconstruct for all index types
        try:
            embeddings = np.zeros((self.size, self.dimension), dtype=np.float32)
            for i in range(self.size):
                embeddings[i] = self._index.reconstruct(i)
            return embeddings
        except RuntimeError as exc:
            logger.warning("Failed to reconstruct embeddings: %s", exc)
            return None

    def search(
        self,
        query: np.ndarray,
        k: int = 1,
    ) -> list[tuple[str, float]]:
        """Return k nearest neighbours for the query embedding.

        Args:
            query: Query embedding vector of shape (dimension,) or (1, dimension).
            k: Number of nearest neighbours to return.

        Returns:
            List of (label, distance) tuples sorted by distance (ascending).
            Returns an empty list if the index is empty or search fails.
        """
        if self._index is None or self.is_empty:
            return []

        # Ensure correct shape
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.dimension:
            logger.warning(
                "Query dimension (%d) doesn't match index dimension (%d)",
                query.shape[1],
                self.dimension,
            )
            return []

        # Ensure float32
        query = query.astype(np.float32)

        # Limit k to available embeddings
        k = min(k, self.size)

        try:
            distances, indices = self._index.search(query, k)
        except Exception as exc:
            logger.warning("FAISS search failed: %s", exc)
            return []

        results: list[tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._labels):
                continue
            results.append((self._labels[idx], float(dist)))

        return results

    def search_single(
        self,
        query: np.ndarray,
    ) -> Optional[tuple[str, float]]:
        """Return the single nearest neighbour for the query embedding.

        This is a convenience method equivalent to search(query, k=1)[0].

        Args:
            query: Query embedding vector.

        Returns:
            Tuple of (label, distance) or None if the index is empty.
        """
        results = self.search(query, k=1)
        return results[0] if results else None

    def save(self, path: Path) -> None:
        """Persist the encrypted index to disk.

        The index is serialized and encrypted using the application's
        encryption key before saving.

        Args:
            path: File path to save the encrypted index.

        Raises:
            ValueError: If the index is empty or not trained.
        """
        if self._index is None or not self._is_trained:
            raise ValueError("Cannot save an empty or untrained index")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize index to bytes
        index_bytes = faiss.serialize_index(self._index)

        # Create payload with labels
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            index_bytes=index_bytes,
            labels=np.array(self._labels, dtype=object),
            dimension=np.array([self.dimension]),
        )

        # Encrypt and save
        encrypted = encrypt_bytes(buffer.getvalue())
        path.write_bytes(encrypted)

        logger.info("Saved FAISS index with %d embeddings to %s", self.size, path)

    @classmethod
    def load(cls, path: Path) -> "FAISSIndex":
        """Load an encrypted index from disk.

        Args:
            path: File path to the encrypted index.

        Returns:
            Loaded FAISSIndex instance.

        Raises:
            FileNotFoundError: If the index file doesn't exist.
            ValueError: If the index file is corrupted or invalid.
        """
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")

        try:
            encrypted = path.read_bytes()
            decrypted = decrypt_bytes(encrypted)

            buffer = io.BytesIO(decrypted)
            data = np.load(buffer, allow_pickle=True)

            index_bytes = data["index_bytes"]
            labels = data["labels"].tolist()
            dimension = int(data["dimension"][0])

        except Exception as exc:
            raise ValueError(f"Failed to load FAISS index: {exc}") from exc

        instance = cls(dimension=dimension)
        instance._index = faiss.deserialize_index(index_bytes)
        instance._labels = labels
        instance._is_trained = True

        logger.info("Loaded FAISS index with %d embeddings from %s", instance.size, path)

        return instance

    def clear(self) -> None:
        """Remove all embeddings from the index."""
        self._index = None
        self._labels = []
        self._is_trained = False


def build_faiss_index_from_embeddings(
    embeddings: np.ndarray,
    labels: Sequence[str],
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
) -> FAISSIndex:
    """Build a FAISS index from embeddings and labels.

    This is a convenience function for creating and populating an index
    in a single call.

    Args:
        embeddings: Array of shape (n_samples, dimension).
        labels: Identity labels for each embedding.
        dimension: Embedding dimension (default 128).

    Returns:
        Populated FAISSIndex instance.
    """
    index = FAISSIndex(dimension=dimension)
    index.add_embeddings(embeddings, labels)
    return index


__all__ = [
    "FAISSIndex",
    "build_faiss_index_from_embeddings",
    "APPROXIMATE_SEARCH_THRESHOLD",
    "DEFAULT_EMBEDDING_DIMENSION",
]
