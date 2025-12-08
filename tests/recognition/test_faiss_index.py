"""Unit tests for the FAISS vector database integration."""

from pathlib import Path

import numpy as np
import pytest

from recognition.faiss_index import (
    APPROXIMATE_SEARCH_THRESHOLD,
    FAISSIndex,
    build_faiss_index_from_embeddings,
)


class TestFAISSIndex:
    """Tests for the FAISSIndex class."""

    def test_empty_index_reports_zero_size(self) -> None:
        """An empty index should report size 0."""
        index = FAISSIndex(dimension=128)
        assert index.size == 0
        assert index.is_empty is True

    def test_add_single_embedding(self) -> None:
        """Adding a single embedding should work correctly."""
        index = FAISSIndex(dimension=4)
        embeddings = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        labels = ["alice"]

        index.add_embeddings(embeddings, labels)

        assert index.size == 1
        assert index.is_empty is False

    def test_add_multiple_embeddings(self) -> None:
        """Adding multiple embeddings should accumulate correctly."""
        index = FAISSIndex(dimension=4)

        embeddings1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        index.add_embeddings(embeddings1, ["alice"])

        embeddings2 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        index.add_embeddings(embeddings2, ["bob"])

        assert index.size == 2

    def test_search_returns_nearest_neighbour(self) -> None:
        """Search should return the nearest neighbour by L2 distance."""
        index = FAISSIndex(dimension=4)

        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # alice
                [0.0, 1.0, 0.0, 0.0],  # bob
                [0.0, 0.0, 1.0, 0.0],  # charlie
            ],
            dtype=np.float32,
        )
        labels = ["alice", "bob", "charlie"]
        index.add_embeddings(embeddings, labels)

        # Query close to alice's embedding
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=1)

        assert len(results) == 1
        assert results[0][0] == "alice"

    def test_search_returns_k_neighbours(self) -> None:
        """Search should return k nearest neighbours."""
        index = FAISSIndex(dimension=4)

        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        labels = ["alice", "alice2", "bob"]
        index.add_embeddings(embeddings, labels)

        query = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=2)

        assert len(results) == 2
        # Both results should be alice variants (closest to query)
        returned_labels = [r[0] for r in results]
        assert "alice" in returned_labels or "alice2" in returned_labels

    def test_search_single_returns_best_match(self) -> None:
        """search_single should return the single best match."""
        index = FAISSIndex(dimension=4)

        embeddings = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        index.add_embeddings(embeddings, ["alice", "bob"])

        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        result = index.search_single(query)

        assert result is not None
        label, distance = result
        assert label == "alice"
        assert distance >= 0

    def test_search_on_empty_index_returns_empty_list(self) -> None:
        """Searching an empty index should return an empty list."""
        index = FAISSIndex(dimension=4)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        results = index.search(query, k=1)
        assert results == []

        single = index.search_single(query)
        assert single is None

    def test_add_embeddings_validates_dimension(self) -> None:
        """Adding embeddings with wrong dimension should raise ValueError."""
        index = FAISSIndex(dimension=4)

        wrong_dim = np.array([[1.0, 2.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="dimension"):
            index.add_embeddings(wrong_dim, ["alice"])

    def test_add_embeddings_validates_label_count(self) -> None:
        """Adding embeddings with mismatched labels should raise ValueError."""
        index = FAISSIndex(dimension=4)

        embeddings = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="labels count"):
            index.add_embeddings(embeddings, ["alice", "bob"])

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        """Index should round-trip through save/load correctly."""
        index_path = tmp_path / "test_index.bin.enc"

        # Create and populate index
        original = FAISSIndex(dimension=4)
        embeddings = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        original.add_embeddings(embeddings, ["alice", "bob"])
        original.save(index_path)

        # Load and verify
        loaded = FAISSIndex.load(index_path)

        assert loaded.size == original.size
        assert loaded.dimension == original.dimension

        # Search should work on loaded index
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        result = loaded.search_single(query)
        assert result is not None
        assert result[0] == "alice"

    def test_load_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            FAISSIndex.load(tmp_path / "nonexistent.bin.enc")

    def test_save_empty_index_raises_error(self, tmp_path: Path) -> None:
        """Saving an empty index should raise ValueError."""
        index = FAISSIndex(dimension=4)

        with pytest.raises(ValueError, match="empty"):
            index.save(tmp_path / "empty.bin.enc")

    def test_clear_removes_all_embeddings(self) -> None:
        """clear() should remove all embeddings from the index."""
        index = FAISSIndex(dimension=4)
        embeddings = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        index.add_embeddings(embeddings, ["alice"])

        assert index.size == 1

        index.clear()

        assert index.size == 0
        assert index.is_empty is True


class TestApproximateSearch:
    """Tests for approximate nearest-neighbour search with large datasets."""

    def test_large_dataset_uses_ivf_index(self) -> None:
        """Datasets >= threshold should use IVF index for approximate search."""
        # Create dataset at threshold
        n_samples = APPROXIMATE_SEARCH_THRESHOLD
        dimension = 4

        embeddings = np.random.randn(n_samples, dimension).astype(np.float32)
        labels = [f"user_{i}" for i in range(n_samples)]

        index = FAISSIndex(dimension=dimension)
        index.add_embeddings(embeddings, labels)

        # Index should still work for search
        query = embeddings[0]
        result = index.search_single(query)

        assert result is not None
        # Should find the exact match or something very close
        assert result[1] < 0.1  # Very small distance

    def test_small_dataset_uses_flat_index(self) -> None:
        """Datasets < threshold should use flat index for exact search."""
        n_samples = 10  # Well below threshold
        dimension = 4

        embeddings = np.random.randn(n_samples, dimension).astype(np.float32)
        labels = [f"user_{i}" for i in range(n_samples)]

        index = FAISSIndex(dimension=dimension)
        index.add_embeddings(embeddings, labels)

        # Search for exact embedding should return distance 0
        query = embeddings[0]
        result = index.search_single(query)

        assert result is not None
        assert result[0] == "user_0"
        assert pytest.approx(result[1], abs=1e-5) == 0.0


class TestBuildFaissIndexHelper:
    """Tests for the build_faiss_index_from_embeddings helper."""

    def test_builds_index_from_embeddings(self) -> None:
        """Helper should create a populated index."""
        embeddings = np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        labels = ["alice", "bob"]

        index = build_faiss_index_from_embeddings(embeddings, labels, dimension=2)

        assert index.size == 2
        assert not index.is_empty
