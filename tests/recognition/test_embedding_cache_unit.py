import hashlib
from unittest.mock import MagicMock, patch

import numpy as np

from recognition.embedding_cache import (
    _deserialize_embedding,
    _embedding_key,
    _get_cache,
    _serialize_embedding,
    compute_dataset_hash,
    get_cached_dataset_index,
    get_cached_embedding,
    get_dataset_hash,
    invalidate_all_embeddings,
    invalidate_user_embeddings,
    set_cached_dataset_index,
    set_cached_embedding,
)


def test_get_cache():
    # Should use django's configured cache
    cache = _get_cache()
    assert cache is not None


def test_embedding_key():
    assert _embedding_key("user1") == "emb:user1"
    assert _embedding_key("user1", "hash123") == "emb:user1:hash123"


def test_serialize_deserialize():
    arr = np.array([1.5, 2.5, 3.5])
    serialized = _serialize_embedding(arr)
    assert isinstance(serialized, str)
    deserialized = _deserialize_embedding(serialized)
    assert isinstance(deserialized, np.ndarray)
    np.testing.assert_array_equal(arr, deserialized)


def test_deserialize_invalid():
    assert _deserialize_embedding("not json") is None


@patch("recognition.embedding_cache._get_cache")
def test_get_cached_embedding(mock_get_cache):
    mock_cache = MagicMock()
    mock_get_cache.return_value = mock_cache

    # Cache miss
    mock_cache.get.return_value = None
    assert get_cached_embedding("user1") is None

    # Cache hit
    mock_cache.get.return_value = "[1.0, 2.0]"
    res = get_cached_embedding("user1")
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, np.array([1.0, 2.0]))


@patch("recognition.embedding_cache._get_cache")
def test_get_cached_embedding_exception(mock_get_cache):
    mock_cache = MagicMock()
    mock_get_cache.return_value = mock_cache

    mock_cache.get.side_effect = Exception("error")
    assert get_cached_embedding("user1") is None


@patch("recognition.embedding_cache._get_cache")
def test_set_cached_embedding(mock_get_cache):
    mock_cache = MagicMock()
    mock_get_cache.return_value = mock_cache

    arr = np.array([1.0, 2.0])

    # Default TTL
    assert set_cached_embedding("user1", arr) is True
    mock_cache.set.assert_called_with("emb:user1", "[1.0, 2.0]", timeout=3600)

    # Custom TTL
    assert set_cached_embedding("user1", arr, ttl=100) is True
    mock_cache.set.assert_called_with("emb:user1", "[1.0, 2.0]", timeout=100)

    # Verify exception handling
    mock_cache.set.side_effect = Exception("error")
    assert set_cached_embedding("user1", arr) is False


@patch("recognition.embedding_cache._get_cache")
def test_invalidate_user_embeddings(mock_get_cache):
    mock_cache = MagicMock()
    mock_get_cache.return_value = mock_cache

    assert invalidate_user_embeddings("user1") is True
    mock_cache.delete.assert_called_with("emb:user1")

    mock_cache.delete.side_effect = Exception("error")
    assert invalidate_user_embeddings("user1") is False


@patch("recognition.embedding_cache._get_cache")
def test_invalidate_all_embeddings(mock_get_cache):
    mock_cache = MagicMock()
    mock_get_cache.return_value = mock_cache

    assert invalidate_all_embeddings() is True
    mock_cache.clear.assert_called_once()

    mock_cache.clear.side_effect = Exception("error")
    assert invalidate_all_embeddings() is False


@patch("recognition.embedding_cache._get_cache")
def test_dataset_index_cache(mock_get_cache):
    mock_cache = MagicMock()
    mock_get_cache.return_value = mock_cache

    # Test setting with TTL
    index = [{"id": 1, "embedding": np.array([1.0, 2.0])}]
    assert set_cached_dataset_index(index, "hash123", ttl=100) is True
    mock_cache.set.assert_any_call(
        "dataset_index", '[{"id": 1, "embedding": [1.0, 2.0]}]', timeout=100
    )
    mock_cache.set.assert_any_call("dataset_hash", "hash123", timeout=100)

    # Test setting missing embedding key
    index_no_emb = [{"id": 1}]
    assert set_cached_dataset_index(index_no_emb, "hash123") is True

    # Test setting exception
    mock_cache.set.side_effect = Exception("error")
    assert set_cached_dataset_index(index) is False

    mock_cache.set.side_effect = None

    # Test getting
    mock_cache.get.return_value = '[{"id": 1, "embedding": [1.0, 2.0]}]'
    res = get_cached_dataset_index()
    assert len(res) == 1
    assert res[0]["id"] == 1
    assert isinstance(res[0]["embedding"], np.ndarray)
    np.testing.assert_array_equal(res[0]["embedding"], np.array([1.0, 2.0]))

    # Test getting missing embedding key
    mock_cache.get.return_value = '[{"id": 1}]'
    res = get_cached_dataset_index()
    assert len(res) == 1
    assert res[0]["id"] == 1
    assert "embedding" not in res[0]

    # Test getting empty embedding
    mock_cache.get.return_value = '[{"id": 1, "embedding": ""}]'
    res = get_cached_dataset_index()
    assert len(res) == 1
    assert res[0]["embedding"] == ""

    # Test getting exception
    mock_cache.get.side_effect = Exception("error")
    assert get_cached_dataset_index() is None


@patch("recognition.embedding_cache._get_cache")
def test_get_dataset_hash(mock_get_cache):
    mock_cache = MagicMock()
    mock_get_cache.return_value = mock_cache

    mock_cache.get.return_value = "hash123"
    assert get_dataset_hash() == "hash123"
    mock_cache.get.assert_called_with("dataset_hash")

    mock_cache.get.side_effect = Exception("error")
    assert get_dataset_hash() is None


@patch("pathlib.Path.exists")
def test_compute_dataset_hash(mock_exists, tmp_path):
    # Non-existent dir
    mock_exists.return_value = False
    assert compute_dataset_hash("/non/existent/path") == "empty"
    mock_exists.return_value = True

    # Empty dir
    assert compute_dataset_hash(str(tmp_path)) == hashlib.sha256().hexdigest()

    # With files
    f = tmp_path / "test.txt"
    f.write_text("hello")

    with patch("pathlib.Path.stat") as mock_stat:
        mock_stat_res = MagicMock()
        mock_stat_res.st_mtime = 1.0
        mock_stat_res.st_size = 5
        mock_stat.return_value = mock_stat_res

        hash1 = compute_dataset_hash(str(tmp_path))
        assert hash1 != "empty"
        assert hash1 != "error"

        # Change file
        mock_stat_res.st_mtime = 2.0
        hash2 = compute_dataset_hash(str(tmp_path))
        assert hash1 != hash2

    # OSError
    with patch("os.walk") as mock_walk:
        mock_walk.side_effect = OSError("error")
        assert compute_dataset_hash(str(tmp_path)) == "error"


def test_get_cache_exception():
    with patch("recognition.embedding_cache.caches") as mock_caches:
        mock_caches.__getitem__.side_effect = Exception("error")

        # Should fallback to default but exception on that too should probably raise or return default if we mock it right
        # Let's just mock the first failure, and have it return default
        # mock_caches.__getitem__ takes a key.
        def getitem(key):
            if key == "embeddings":
                raise Exception("no embeddings")
            return "default_cache"

        mock_caches.__getitem__.side_effect = getitem
        assert _get_cache() == "default_cache"


@patch("recognition.embedding_cache._get_cache")
def test_dataset_index_cache_none(mock_get_cache):
    mock_cache = MagicMock()
    mock_get_cache.return_value = mock_cache
    mock_cache.get.return_value = None
    assert get_cached_dataset_index() is None


@patch("recognition.embedding_cache._get_cache")
def test_dataset_index_cache_no_hash(mock_get_cache):
    mock_cache = MagicMock()
    mock_get_cache.return_value = mock_cache
    index = [{"id": 1, "embedding": np.array([1.0, 2.0])}]
    assert set_cached_dataset_index(index) is True
