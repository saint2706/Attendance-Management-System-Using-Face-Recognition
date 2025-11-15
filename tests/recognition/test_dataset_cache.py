from __future__ import annotations

import os
import pickle
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import django
from django.test import TestCase, override_settings

import numpy as np
from cryptography.fernet import Fernet

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")

_fake_cv2 = MagicMock(name="cv2")
sys.modules.setdefault("cv2", _fake_cv2)

django.setup()

from recognition import views  # noqa: E402
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
            views,
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
            views,
            "_build_dataset_embeddings_for_matching",
            autospec=True,
            side_effect=[updated_index, refreshed_index],
        ) as mock_builder:
            first = views._load_dataset_embeddings_for_matching("Facenet", "ssd", True)
            (self.dataset_root / "bob").mkdir(parents=True, exist_ok=True)
            new_image = self.dataset_root / "bob" / "1.jpg"
            new_image.write_bytes(b"dummy2")
            second = views._load_dataset_embeddings_for_matching("Facenet", "ssd", True)

        self.assertIs(first, updated_index)
        self.assertIs(second, refreshed_index)
        self.assertEqual(mock_builder.call_count, 2)

    def test_cache_files_encrypted_and_round_trip(self):
        image_path = self._seed_dataset()

        dataset_index = [
            {"identity": str(image_path), "embedding": np.array([0.3, 0.4], dtype=float)}
        ]

        with patch.object(
            views,
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
        self.assertIsInstance(stored_index[0]["embedding"], list)

        views._dataset_embedding_cache._memory_cache.clear()

        with patch.object(
            views,
            "_build_dataset_embeddings_for_matching",
            side_effect=AssertionError("should not rebuild"),
        ):
            round_tripped = views._load_dataset_embeddings_for_matching("Facenet", "ssd", True)

        self.assertEqual(len(round_tripped), 1)
        restored_embedding = round_tripped[0]["embedding"]
        self.assertIsInstance(restored_embedding, np.ndarray)
        np.testing.assert_allclose(restored_embedding, dataset_index[0]["embedding"])
