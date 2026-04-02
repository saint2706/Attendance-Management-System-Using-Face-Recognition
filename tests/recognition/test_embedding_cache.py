import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from recognition import embedding_cache


@pytest.fixture
def mock_cache():
    with patch("recognition.embedding_cache._get_cache") as mock_get_cache:
        cache = MagicMock()
        mock_get_cache.return_value = cache
        yield cache


def test_get_cached_embedding(mock_cache):
    # Test valid data
    mock_cache.get.return_value = json.dumps([1.0, 2.0, 3.0])
    result = embedding_cache.get_cached_embedding("testuser", "hash123")
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
    mock_cache.get.assert_called_with("emb:testuser:hash123")

    # Test None data
    mock_cache.get.return_value = None
    result = embedding_cache.get_cached_embedding("testuser")
    assert result is None
    mock_cache.get.assert_called_with("emb:testuser")

    # Test cache exception
    mock_cache.get.side_effect = Exception("Cache error")
    result = embedding_cache.get_cached_embedding("testuser")
    assert result is None


def test_set_cached_embedding(mock_cache):
    embedding = np.array([1.0, 2.0, 3.0])

    # Test successful set
    mock_cache.set.return_value = True
    result = embedding_cache.set_cached_embedding("testuser", embedding, "hash123", ttl=3600)
    assert result is True
    mock_cache.set.assert_called_with("emb:testuser:hash123", "[1.0, 2.0, 3.0]", timeout=3600)

    # Test default TTL
    with patch("recognition.embedding_cache.settings") as mock_settings:
        mock_settings.EMBEDDING_CACHE_TTL = 7200
        result = embedding_cache.set_cached_embedding("testuser", embedding)
        assert result is True
        mock_cache.set.assert_called_with("emb:testuser", "[1.0, 2.0, 3.0]", timeout=7200)

    # Test cache exception
    mock_cache.set.side_effect = Exception("Cache error")
    result = embedding_cache.set_cached_embedding("testuser", embedding)
    assert result is False


def test_deserialize_embedding():
    # Test valid JSON
    result = embedding_cache._deserialize_embedding("[1.0, 2.0, 3.0]")
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    # Test invalid JSON
    result = embedding_cache._deserialize_embedding("invalid json")
    assert result is None

    # Test invalid type (not a list)
    result = embedding_cache._deserialize_embedding('{"key": "value"}')
    assert result is None


def test_invalidate_user_embeddings(mock_cache):
    # Test successful invalidation
    result = embedding_cache.invalidate_user_embeddings("testuser")
    assert result is True
    mock_cache.delete.assert_called_with("emb:testuser")

    # Test cache exception
    mock_cache.delete.side_effect = Exception("Cache error")
    result = embedding_cache.invalidate_user_embeddings("testuser")
    assert result is False


def test_invalidate_all_embeddings(mock_cache):
    # Test successful invalidation
    result = embedding_cache.invalidate_all_embeddings()
    assert result is True
    mock_cache.clear.assert_called_once()

    # Test cache exception
    mock_cache.clear.side_effect = Exception("Cache error")
    result = embedding_cache.invalidate_all_embeddings()
    assert result is False


def test_get_cached_dataset_index(mock_cache):
    # Test valid data
    index_data = [{"name": "test1", "embedding": [1.0, 2.0]}, {"name": "test2", "embedding": None}]
    mock_cache.get.return_value = json.dumps(index_data)

    result = embedding_cache.get_cached_dataset_index()
    assert result is not None
    assert len(result) == 2
    assert result[0]["name"] == "test1"
    assert isinstance(result[0]["embedding"], np.ndarray)
    np.testing.assert_array_equal(result[0]["embedding"], np.array([1.0, 2.0]))
    assert result[1]["name"] == "test2"
    assert result[1]["embedding"] is None

    mock_cache.get.assert_called_with("dataset_index")

    # Test None data
    mock_cache.get.return_value = None
    result = embedding_cache.get_cached_dataset_index()
    assert result is None

    # Test cache exception
    mock_cache.get.side_effect = Exception("Cache error")
    result = embedding_cache.get_cached_dataset_index()
    assert result is None


def test_set_cached_dataset_index(mock_cache):
    index_data = [{"name": "test1", "embedding": np.array([1.0, 2.0])}, {"name": "test2"}]

    # Test successful set with hash
    result = embedding_cache.set_cached_dataset_index(index_data, dataset_hash="hash123", ttl=3600)
    assert result is True

    # Verify the serialized data
    call_args = mock_cache.set.call_args_list[0]
    assert call_args[0][0] == "dataset_index"
    saved_data = json.loads(call_args[0][1])
    assert saved_data[0]["embedding"] == [1.0, 2.0]
    assert "embedding" not in saved_data[1]

    # Verify hash was also saved
    mock_cache.set.assert_any_call("dataset_hash", "hash123", timeout=3600)

    # Test cache exception
    mock_cache.set.side_effect = Exception("Cache error")
    result = embedding_cache.set_cached_dataset_index(index_data)
    assert result is False


def test_get_dataset_hash(mock_cache):
    # Test success
    mock_cache.get.return_value = "hash123"
    result = embedding_cache.get_dataset_hash()
    assert result == "hash123"
    mock_cache.get.assert_called_with("dataset_hash")

    # Test cache exception
    mock_cache.get.side_effect = Exception("Cache error")
    result = embedding_cache.get_dataset_hash()
    assert result is None


def test_compute_dataset_hash(tmp_path):
    # Test non-existent directory
    missing_dir = tmp_path / "missing"
    result = embedding_cache.compute_dataset_hash(str(missing_dir))
    assert result == "empty"

    # Test existing directory with files
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    user1_dir = dataset_dir / "user1"
    user1_dir.mkdir()
    file1 = user1_dir / "img1.jpg"
    file1.write_text("fake image data 1")

    user2_dir = dataset_dir / "user2"
    user2_dir.mkdir()
    file2 = user2_dir / "img1.jpg"
    file2.write_text("fake image data 2")

    # Compute initial hash
    hash1 = embedding_cache.compute_dataset_hash(str(dataset_dir))
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 is 64 hex chars

    # Verify consistency
    hash2 = embedding_cache.compute_dataset_hash(str(dataset_dir))
    assert hash1 == hash2

    # Verify it changes when a file is modified
    file1.write_text("modified data")
    hash3 = embedding_cache.compute_dataset_hash(str(dataset_dir))
    assert hash1 != hash3

    # Test OSError handling
    with patch("os.walk", side_effect=OSError("Mock OS error")):
        result = embedding_cache.compute_dataset_hash(str(dataset_dir))
        assert result == "error"
