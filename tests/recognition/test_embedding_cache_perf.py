import time
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
from django.conf import settings
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

    def resolve(self):
        return Path(f"/abs/{self.path_str}")

    def stat(self):
        return self.stat_mock

    def __repr__(self):
        return f"PathMock('{self.path_str}')"

@pytest.mark.django_db
def test_embedding_cache_performance():
    # Setup mocks
    num_images = 100
    mock_paths = []

    for i in range(num_images):
        p = PathMock(f"fake/path/img_{i}.jpg")
        mock_paths.append(p)

    # Mock file read (decryption simulation) - SIMULATE SLOWNESS
    def side_effect_decrypt_image_bytes(path):
        time.sleep(0.001) # Simulate 1ms decryption/read time
        return b"fake_decrypted_bytes"

    with patch("recognition.views_legacy.TRAINING_DATASET_ROOT") as mock_root, \
         patch("recognition.views_legacy._decrypt_image_bytes", side_effect=side_effect_decrypt_image_bytes) as mock_decrypt, \
         patch("recognition.views_legacy._decode_image_bytes") as mock_decode, \
         patch("recognition.views_legacy.DeepFace.represent") as mock_deepface, \
         patch("recognition.views_legacy.extract_embedding") as mock_extract, \
         patch("recognition.views_legacy.cache") as mock_cache:

        mock_root.glob.return_value = mock_paths
        mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract.return_value = ([0.1] * 128, None)

        # Simulate cache HIT for all items
        mock_cache.get.return_value = [0.1] * 128

        start_time = time.time()
        _build_dataset_embeddings_for_matching("model", "backend", False)
        duration = time.time() - start_time

        print(f"\nExecution time for {num_images} images: {duration:.4f}s")
        print(f"Decrypt calls: {mock_decrypt.call_count}")

        # With optimization, this should be 0 because cache hit prevents decryption
        assert mock_decrypt.call_count == 0
