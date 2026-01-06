from __future__ import annotations

import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import django
from django.core.cache import cache
from django.test import TestCase, override_settings

import numpy as np
from cryptography.fernet import Fernet

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")

_fake_cv2 = MagicMock(name="cv2")
sys.modules.setdefault("cv2", _fake_cv2)

# Only setup Django if it hasn't been configured yet (e.g., running standalone)
if not django.apps.apps.ready:
    django.setup()

from recognition import views, views_legacy  # noqa: E402
from src.common import FaceDataEncryption  # noqa: E402

TEST_FACE_KEY = Fernet.generate_key()


@override_settings(FACE_DATA_ENCRYPTION_KEY=TEST_FACE_KEY)
class DatasetEmbeddingCacheTests(TestCase):
    def setUp(self):
        self.dataset_root = views.TRAINING_DATASET_ROOT
        self.data_root = views.DATA_ROOT
        shutil.rmtree(self.dataset_root, ignore_errors=True)
        shutil.rmtree(self.data_root, ignore_errors=True)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        views._dataset_embedding_cache.invalidate()

    def tearDown(self):  # pragma: no cover - cleanup
        views._dataset_embedding_cache.invalidate()
        shutil.rmtree(self.dataset_root, ignore_errors=True)
        shutil.rmtree(self.data_root, ignore_errors=True)

    def _seed_dataset(self, username: str = "alice") -> Path:
        user_dir = self.dataset_root / username
        user_dir.mkdir(parents=True, exist_ok=True)
        image_path = user_dir / "1.jpg"
        image_path.write_bytes(b"dummy")
        return image_path

    def test_cached_embeddings_reused_without_dataset_changes(self):
        self._seed_dataset()

        fake_index = [{"identity": "alice/1.jpg", "embedding": np.array([0.1, 0.2])}]
        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            autospec=True,
            return_value=fake_index,
        ) as mock_builder:
            first = views._load_dataset_embeddings_for_matching("Facenet", "ssd", True)
            second = views._load_dataset_embeddings_for_matching("Facenet", "ssd", True)

        self.assertIs(first, fake_index)
        self.assertIs(second, fake_index)
        self.assertEqual(mock_builder.call_count, 1)

    def test_cache_refreshes_when_dataset_files_change(self):
        self._seed_dataset()

        updated_index = [
            {"identity": "alice/1.jpg", "embedding": np.array([0.1, 0.2])},
        ]
        refreshed_index = [
            {"identity": "alice/1.jpg", "embedding": np.array([0.3, 0.4])},
            {"identity": "bob/1.jpg", "embedding": np.array([0.5, 0.6])},
        ]

        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            autospec=True,
            side_effect=[updated_index, refreshed_index],
        ) as mock_builder:
            first = views._load_dataset_embeddings_for_matching("Facenet", "ssd", True)
            (self.dataset_root / "bob").mkdir(parents=True, exist_ok=True)
            new_image = self.dataset_root / "bob" / "1.jpg"
            new_image.write_bytes(b"dummy2")
            # Invalidate the cache manually to simulate the behavior of capture_dataset task
            cache.delete("recognition:dataset_state")
            second = views._load_dataset_embeddings_for_matching("Facenet", "ssd", True)

        self.assertIs(first, updated_index)
        self.assertIs(second, refreshed_index)
        self.assertEqual(mock_builder.call_count, 2)

    def test_cache_files_encrypted_and_round_trip(self):
        image_path = self._seed_dataset()

        dataset_index = [
            {
                "identity": str(image_path),
                "embedding": np.array([0.3, 0.4], dtype=float),
            }
        ]

        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            autospec=True,
            return_value=dataset_index,
        ):
            views._load_dataset_embeddings_for_matching("Facenet", "ssd", True)

        cache_file = views._dataset_embedding_cache._cache_file_path("Facenet", "ssd", True)
        self.assertTrue(cache_file.exists())

        encrypted_payload = cache_file.read_bytes()
        helper = FaceDataEncryption(TEST_FACE_KEY)
        decrypted_payload = helper.decrypt(encrypted_payload)
        self.assertNotEqual(encrypted_payload, decrypted_payload)

        payload = pickle.loads(decrypted_payload)
        stored_index = payload["dataset_index"]
        self.assertIsInstance(stored_index, list)
        # ⚡ Optimization: We now store numpy arrays directly in the cache
        self.assertIsInstance(stored_index[0]["embedding"], np.ndarray)

        views._dataset_embedding_cache._memory_cache.clear()

        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            side_effect=AssertionError("should not rebuild"),
        ):
            round_tripped = views._load_dataset_embeddings_for_matching("Facenet", "ssd", True)

        self.assertEqual(len(round_tripped), 1)
        restored_embedding = round_tripped[0]["embedding"]
        self.assertIsInstance(restored_embedding, np.ndarray)
        np.testing.assert_allclose(restored_embedding, dataset_index[0]["embedding"])

    def test_backward_compatibility_with_legacy_list_embeddings(self):
        """Test that legacy cache files with list-based embeddings can still be loaded."""
        image_path = self._seed_dataset()

        # Get the actual dataset state that will be computed by the cache system
        current_dataset_state = views._dataset_embedding_cache._compute_dataset_state()

        # Create a legacy cache file with embeddings stored as lists (old format)
        legacy_embedding = [0.3, 0.4, 0.5]  # List instead of numpy array
        dataset_index = [
            {
                "identity": str(image_path.relative_to(self.dataset_root)),
                "embedding": legacy_embedding,  # Store as list (legacy format)
            }
        ]

        # Manually create a legacy cache file with the correct dataset state
        cache_file = views._dataset_embedding_cache._cache_file_path("Facenet", "ssd", True)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dataset_state": current_dataset_state,
            "dataset_index": dataset_index,
        }
        serialized = pickle.dumps(payload)
        helper = FaceDataEncryption(TEST_FACE_KEY)
        encrypted = helper.encrypt(serialized)
        cache_file.write_bytes(encrypted)

        # Clear memory cache to force loading from disk
        views._dataset_embedding_cache._memory_cache.clear()

        # Load the legacy cache - should convert lists to numpy arrays
        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            side_effect=AssertionError("should not rebuild - legacy cache should load"),
        ):
            loaded_index = views._load_dataset_embeddings_for_matching("Facenet", "ssd", True)

        # Verify the cache was loaded successfully
        self.assertEqual(len(loaded_index), 1)
        restored_embedding = loaded_index[0]["embedding"]

        # Verify the list was converted to numpy array
        self.assertIsInstance(restored_embedding, np.ndarray)
        np.testing.assert_allclose(restored_embedding, legacy_embedding)

    @override_settings(RECOGNITION_DATASET_STATE_CACHE_TIMEOUT=1)
    def test_cache_expires_after_timeout(self):
        """Test that cached dataset state expires after the configured timeout and triggers a filesystem rescan."""
        self._seed_dataset()

        # Mock the filesystem state computation to verify it's called after timeout
        original_compute = views._dataset_embedding_cache._compute_dataset_state

        with patch.object(
            views._dataset_embedding_cache,
            "_compute_dataset_state",
            wraps=original_compute,
        ) as mock_compute:
            # First call should compute dataset state
            views._dataset_embedding_cache._current_dataset_state()
            self.assertEqual(mock_compute.call_count, 1)

            # Subsequent call within timeout should use cached state
            views._dataset_embedding_cache._current_dataset_state()
            self.assertEqual(mock_compute.call_count, 1)

            # Wait for cache to expire (timeout is set to 1 second)
            time.sleep(1.5)

            # Call after timeout should recompute dataset state
            views._dataset_embedding_cache._current_dataset_state()
            self.assertEqual(mock_compute.call_count, 2)
