import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from recognition.views_legacy import _build_dataset_embeddings_for_matching


class PathMock:
    def __init__(self, path_str):
        self.path_str = path_str
        self.stat_mock = MagicMock()
        self.stat_mock.st_mtime_ns = 123456789
        self.stat_mock.st_size = 1024
        self.parent = MagicMock()
        self.parent.name = "user"

    def __str__(self):
        return self.path_str

    def __lt__(self, other):
        return str(self) < str(other)

    def __le__(self, other):
        return str(self) <= str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __ge__(self, other):
        return str(self) >= str(other)

    def resolve(self):
        return Path(f"/abs/{self.path_str}")

    def stat(self):
        return self.stat_mock

    def __repr__(self):
        return f"PathMock('{self.path_str}')"


def test_embedding_cache_performance():
    # Setup mocks
    num_images = 100
    mock_paths = []

    for i in range(num_images):
        p = PathMock(f"fake/path/img_{i}.jpg")
        mock_paths.append(p)

    # Mock file read (decryption simulation) - SIMULATE SLOWNESS
    def side_effect_decrypt_image_bytes(path):
        time.sleep(0.001)  # Simulate 1ms decryption/read time
        return b"fake_decrypted_bytes"

    with (
        patch("recognition.views_legacy.TRAINING_DATASET_ROOT") as mock_root,
        patch(
            "recognition.views_legacy._decrypt_image_bytes",
            side_effect=side_effect_decrypt_image_bytes,
        ) as mock_decrypt,
        patch("recognition.views_legacy._decode_image_bytes") as mock_decode,
        patch("recognition.views_legacy.DeepFace.represent"),
        patch("recognition.views_legacy.extract_embedding") as mock_extract,
        patch("recognition.views_legacy.cache") as mock_cache,
    ):

        mock_root.glob.return_value = mock_paths
        mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract.return_value = ([0.1] * 128, None)

        # Simulate cache HIT for all items
        # Need to return a dict mapping cache keys to embeddings for get_many
        def get_many_side_effect(keys):
            return {key: [0.1] * 128 for key in keys}

        mock_cache.get_many.side_effect = get_many_side_effect
        mock_cache.get.return_value = [0.1] * 128

        start_time = time.time()
        _build_dataset_embeddings_for_matching("model", "backend", False)
        _ = time.time() - start_time

        # With optimization, this should be 0 because cache hit prevents decryption
        assert (
            mock_decrypt.call_count == 0
        ), f"Expected 0 decrypt calls, got {mock_decrypt.call_count}"


def test_embedding_cache_stores_on_miss():
    """Test that cache.set is called when embeddings are computed (cache miss)."""
    num_images = 5
    mock_paths = []

    for i in range(num_images):
        p = PathMock(f"fake/path/img_{i}.jpg")
        mock_paths.append(p)

    with (
        patch("recognition.views_legacy.TRAINING_DATASET_ROOT") as mock_root,
        patch("recognition.views_legacy._decrypt_image_bytes") as mock_decrypt,
        patch("recognition.views_legacy._decode_image_bytes") as mock_decode,
        patch("recognition.views_legacy.DeepFace.represent"),
        patch("recognition.views_legacy.extract_embedding") as mock_extract,
        patch("recognition.views_legacy.cache") as mock_cache,
    ):

        mock_root.glob.return_value = mock_paths
        mock_decrypt.return_value = b"fake_decrypted_bytes"
        mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract.return_value = ([0.1] * 128, None)

        # Simulate cache MISS - get_many returns empty dict, get returns None
        mock_cache.get_many.return_value = {}
        mock_cache.get.return_value = None

        _build_dataset_embeddings_for_matching("model", "backend", False)

        # Verify cache.set was called for each computed embedding
        assert (
            mock_cache.set.call_count == num_images
        ), f"Expected {num_images} cache.set calls, got {mock_cache.set.call_count}"

        # Verify decryption happened for all images (cache miss)
        assert (
            mock_decrypt.call_count == num_images
        ), f"Expected {num_images} decrypt calls on cache miss, got {mock_decrypt.call_count}"
