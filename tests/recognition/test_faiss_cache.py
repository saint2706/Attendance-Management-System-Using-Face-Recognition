"""
Tests for FAISS index caching in DatasetEmbeddingCache.

This module verifies that the FAISS index is cached correctly and only rebuilt
when the dataset changes, eliminating redundant construction during kiosk-mode
attendance marking.
"""

from __future__ import annotations

import os
import shutil
import sys
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
from recognition.faiss_index import FAISSIndex  # noqa: E402
from src.common import FaceDataEncryption  # noqa: E402

TEST_FACE_KEY = Fernet.generate_key()


@override_settings(FACE_DATA_ENCRYPTION_KEY=TEST_FACE_KEY)
class FAISSIndexCacheTests(TestCase):
    """Test suite for FAISS index caching behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.dataset_root = views.TRAINING_DATASET_ROOT
        self.data_root = views.DATA_ROOT
        shutil.rmtree(self.dataset_root, ignore_errors=True)
        shutil.rmtree(self.data_root, ignore_errors=True)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        views._dataset_embedding_cache.invalidate()

    def tearDown(self):  # pragma: no cover - cleanup
        """Clean up test fixtures."""
        views._dataset_embedding_cache.invalidate()
        shutil.rmtree(self.dataset_root, ignore_errors=True)
        shutil.rmtree(self.data_root, ignore_errors=True)

    def _seed_dataset(self, username: str = "alice") -> Path:
        """Create a dummy dataset entry for testing."""
        user_dir = self.dataset_root / username
        user_dir.mkdir(parents=True, exist_ok=True)
        image_path = user_dir / "1.jpg"
        image_path.write_bytes(b"dummy")
        return image_path

    def _create_fake_faiss_index(self, embeddings: list, labels: list) -> FAISSIndex:
        """Create a fake FAISS index for testing."""
        embeddings_array = np.array(embeddings, dtype=np.float32)
        mock_index = MagicMock(spec=FAISSIndex)
        mock_index.size = len(embeddings)
        mock_index.embeddings = embeddings_array
        mock_index.labels = labels
        return mock_index

    def test_faiss_index_built_only_once_for_unchanged_dataset(self):
        """Verify FAISS index is only built once when dataset doesn't change."""
        self._seed_dataset()

        fake_index = [{"identity": "alice/1.jpg", "embedding": np.array([0.1, 0.2, 0.3, 0.4])}]
        fake_faiss = self._create_fake_faiss_index([[0.1, 0.2, 0.3, 0.4]], ["alice/1.jpg"])

        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            autospec=True,
            return_value=fake_index,
        ), patch(
            "recognition.views_legacy.build_faiss_index_from_embeddings",
            autospec=True,
            return_value=fake_faiss,
        ) as mock_build_faiss:
            # First call should build the FAISS index
            first_index = views._dataset_embedding_cache.get_faiss_index(
                "Facenet", "ssd", True, lambda ds=None: fake_index
            )
            # Second call should use the cached FAISS index
            second_index = views._dataset_embedding_cache.get_faiss_index(
                "Facenet", "ssd", True, lambda ds=None: fake_index
            )

        # Verify the index builder was called only once
        self.assertEqual(mock_build_faiss.call_count, 1)
        self.assertIsNotNone(first_index)
        self.assertIsNotNone(second_index)
        self.assertIs(first_index, second_index)

    def test_faiss_index_rebuilt_when_dataset_changes(self):
        """Verify FAISS index is rebuilt when the dataset changes."""
        self._seed_dataset()

        initial_index = [{"identity": "alice/1.jpg", "embedding": np.array([0.1, 0.2, 0.3, 0.4])}]
        updated_index = [
            {"identity": "alice/1.jpg", "embedding": np.array([0.1, 0.2, 0.3, 0.4])},
            {"identity": "bob/1.jpg", "embedding": np.array([0.5, 0.6, 0.7, 0.8])},
        ]

        fake_faiss_1 = self._create_fake_faiss_index([[0.1, 0.2, 0.3, 0.4]], ["alice/1.jpg"])
        fake_faiss_2 = self._create_fake_faiss_index(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], ["alice/1.jpg", "bob/1.jpg"]
        )

        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            autospec=True,
            side_effect=[initial_index, updated_index],
        ), patch(
            "recognition.views_legacy.build_faiss_index_from_embeddings",
            autospec=True,
            side_effect=[fake_faiss_1, fake_faiss_2],
        ) as mock_build_faiss:
            # First call with initial dataset
            first_index = views._dataset_embedding_cache.get_faiss_index(
                "Facenet", "ssd", True, lambda ds=None: initial_index
            )

            # Modify the dataset
            (self.dataset_root / "bob").mkdir(parents=True, exist_ok=True)
            new_image = self.dataset_root / "bob" / "1.jpg"
            new_image.write_bytes(b"dummy2")
            cache.delete("recognition:dataset_state")

            # Second call with updated dataset
            second_index = views._dataset_embedding_cache.get_faiss_index(
                "Facenet", "ssd", True, lambda ds=None: updated_index
            )

        # Verify the FAISS index builder was called twice (once for each dataset state)
        self.assertEqual(mock_build_faiss.call_count, 2)
        self.assertIsNotNone(first_index)
        self.assertIsNotNone(second_index)
        self.assertIsNot(first_index, second_index)

    def test_faiss_index_returns_none_for_empty_dataset(self):
        """Verify get_faiss_index returns None when dataset is empty."""
        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            autospec=True,
            return_value=[],
        ):
            faiss_index = views._dataset_embedding_cache.get_faiss_index(
                "Facenet", "ssd", True, lambda ds=None: []
            )

        self.assertIsNone(faiss_index)

    def test_faiss_index_returns_none_for_invalid_embeddings(self):
        """Verify get_faiss_index returns None when embeddings are invalid."""
        self._seed_dataset()

        # Dataset with no valid embeddings (missing embedding or identity)
        invalid_index = [
            {"identity": "alice/1.jpg"},  # Missing embedding
            {"embedding": np.array([0.1, 0.2])},  # Missing identity
        ]

        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            autospec=True,
            return_value=invalid_index,
        ):
            faiss_index = views._dataset_embedding_cache.get_faiss_index(
                "Facenet", "ssd", True, lambda ds=None: invalid_index
            )

        self.assertIsNone(faiss_index)

    def test_faiss_index_cache_invalidation(self):
        """Verify FAISS cache is properly cleared on invalidation."""
        self._seed_dataset()

        fake_index = [{"identity": "alice/1.jpg", "embedding": np.array([0.1, 0.2, 0.3, 0.4])}]
        fake_faiss = self._create_fake_faiss_index([[0.1, 0.2, 0.3, 0.4]], ["alice/1.jpg"])

        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            autospec=True,
            return_value=fake_index,
        ), patch(
            "recognition.views_legacy.build_faiss_index_from_embeddings",
            autospec=True,
            return_value=fake_faiss,
        ) as mock_build_faiss:
            # Build FAISS index
            first_index = views._dataset_embedding_cache.get_faiss_index(
                "Facenet", "ssd", True, lambda ds=None: fake_index
            )

            # Invalidate cache
            views._dataset_embedding_cache.invalidate()

            # Rebuild FAISS index after invalidation
            second_index = views._dataset_embedding_cache.get_faiss_index(
                "Facenet", "ssd", True, lambda ds=None: fake_index
            )

        # Verify the FAISS index was rebuilt after invalidation
        self.assertEqual(mock_build_faiss.call_count, 2)
        self.assertIsNotNone(first_index)
        self.assertIsNotNone(second_index)

    def test_faiss_index_handles_build_failure_gracefully(self):
        """Verify get_faiss_index handles FAISS build failures gracefully."""
        self._seed_dataset()

        fake_index = [{"identity": "alice/1.jpg", "embedding": np.array([0.1, 0.2, 0.3, 0.4])}]

        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            autospec=True,
            return_value=fake_index,
        ), patch(
            "recognition.views_legacy.build_faiss_index_from_embeddings",
            autospec=True,
            side_effect=Exception("FAISS build error"),
        ):
            faiss_index = views._dataset_embedding_cache.get_faiss_index(
                "Facenet", "ssd", True, lambda ds=None: fake_index
            )

        # Should return None and not crash
        self.assertIsNone(faiss_index)

    def test_faiss_index_uses_dataset_state_from_cache_entry(self):
        """Verify FAISS index uses dataset state from the same cache entry to avoid race conditions."""
        self._seed_dataset()

        fake_index = [{"identity": "alice/1.jpg", "embedding": np.array([0.1, 0.2, 0.3, 0.4])}]
        fake_faiss = self._create_fake_faiss_index([[0.1, 0.2, 0.3, 0.4]], ["alice/1.jpg"])

        with patch.object(
            views_legacy,
            "_build_dataset_embeddings_for_matching",
            autospec=True,
            return_value=fake_index,
        ), patch(
            "recognition.views_legacy.build_faiss_index_from_embeddings",
            autospec=True,
            return_value=fake_faiss,
        ):
            # Get FAISS index
            views._dataset_embedding_cache.get_faiss_index(
                "Facenet", "ssd", True, lambda ds=None: fake_index
            )

        # Verify that the FAISS cache and memory cache have matching dataset states
        key = ("Facenet", "ssd", True)
        memory_entry = views._dataset_embedding_cache._memory_cache.get(key)
        faiss_entry = views._dataset_embedding_cache._faiss_cache.get(key)
        
        self.assertIsNotNone(memory_entry)
        self.assertIsNotNone(faiss_entry)
        
        # The dataset state in both caches should match, confirming they're in sync
        memory_state = memory_entry[0]
        faiss_state = faiss_entry[0]
        self.assertEqual(memory_state, faiss_state)
