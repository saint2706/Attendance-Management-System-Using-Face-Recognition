"""Integration tests for encrypted dataset and model workflows."""

from __future__ import annotations

import io
import os
import pickle
import shutil
import sys
from typing import List
from unittest.mock import MagicMock, patch

import django
import numpy as np
from cryptography.fernet import Fernet
from django.contrib.messages.storage.fallback import FallbackStorage
from django.test import RequestFactory, TestCase, override_settings

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")
django.setup()

from django.contrib.auth.models import User

_fake_cv2 = MagicMock(name="cv2")
sys.modules.setdefault("cv2", _fake_cv2)

from src.common import decrypt_bytes, encrypt_bytes

from recognition import views

TEST_FERNET_KEY = Fernet.generate_key()


class DummyModel:
    """Minimal model used for pickling and inference in tests."""

    def __init__(self):
        self._label = None

    def fit(self, X: List[List[float]], y: List[str]):  # pragma: no cover - invoked via test
        self._label = y[0] if y else None

    def predict(self, X: List[List[float]]):
        if self._label is None:
            return np.array([], dtype=object)
        return np.array([self._label for _ in X], dtype=object)


class EncryptionWorkflowTests(TestCase):
    """Validate dataset encryption, training, and loading of encrypted artifacts."""

    def setUp(self):
        self.factory = RequestFactory()
        self.dataset_root = views.TRAINING_DATASET_ROOT
        self.data_root = views.DATA_ROOT
        shutil.rmtree(self.dataset_root, ignore_errors=True)
        shutil.rmtree(self.data_root, ignore_errors=True)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):  # pragma: no cover - cleanup
        shutil.rmtree(self.dataset_root, ignore_errors=True)
        shutil.rmtree(self.data_root, ignore_errors=True)

    @override_settings(
        DATA_ENCRYPTION_KEY=TEST_FERNET_KEY,
        RECOGNITION_HEADLESS=True,
        RECOGNITION_HEADLESS_DATASET_FRAMES=1,
    )
    def test_create_dataset_encrypts_frames(self):
        """Captured frames should be stored encrypted on disk."""

        output_dir = self.dataset_root / "alice"
        output_dir.mkdir(parents=True, exist_ok=True)

        dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        encoded_bytes = np.arange(12, dtype=np.uint8)

        stream = MagicMock()
        stream.start.return_value = stream
        stream.read.side_effect = [dummy_frame, None]

        with (
            patch.object(views, "VideoStream", return_value=stream),
            patch.object(views, "_is_headless_environment", return_value=True),
            patch.object(views, "time") as mock_time,
            patch.object(views, "cv2") as mock_cv2,
            patch.object(views._dataset_embedding_cache, "invalidate") as mock_invalidate,
        ):
            mock_time.sleep.return_value = None
            mock_cv2.imencode.return_value = (True, encoded_bytes)

            views.create_dataset("alice")

        stored_files = sorted(output_dir.glob("*.jpg"))
        self.assertEqual(len(stored_files), 1)

        mock_invalidate.assert_called_once()

        encrypted_payload = stored_files[0].read_bytes()
        self.assertNotEqual(encrypted_payload, bytes(encoded_bytes))

        decrypted = decrypt_bytes(encrypted_payload)
        self.assertEqual(decrypted, bytes(encoded_bytes))

    @override_settings(
        DATA_ENCRYPTION_KEY=TEST_FERNET_KEY,
        RECOGNITION_HEADLESS=True,
        RECOGNITION_HEADLESS_ATTENDANCE_FRAMES=1,
    )
    @patch.object(views, "train_test_split")
    @patch.object(views, "SVC")
    @patch.object(views, "DeepFace")
    @patch.object(views, "_get_or_compute_cached_embedding")
    def test_training_and_mark_attendance_use_encrypted_artifacts(
        self,
        mock_get_embedding,
        mock_deepface,
        mock_svc,
        mock_train_test_split,
    ):
        """Training should persist encrypted artifacts and mark view should load them."""

        alice_path = self.dataset_root / "alice" / "1.jpg"
        bob_path = self.dataset_root / "bob" / "1.jpg"
        for path in (alice_path, bob_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encrypt_bytes(b"placeholder"))

        mock_get_embedding.side_effect = [
            np.array([0.1, 0.2], dtype=float),
            np.array([0.9, 0.8], dtype=float),
        ]
        mock_deepface.represent.side_effect = [
            [{"embedding": [0.1, 0.2]}],
            [{"embedding": [0.9, 0.8]}],
        ]

        mock_train_test_split.return_value = (
            [[0.1, 0.2]],
            [[0.9, 0.8]],
            ["alice"],
            ["bob"],
        )

        model_instance = DummyModel()
        mock_svc.return_value = model_instance

        staff_user = User.objects.create_user("admin", "admin@example.com", "password", is_staff=True)
        request = self.factory.get("/train/")
        request.user = staff_user
        setattr(request, "session", {})
        messages = FallbackStorage(request)
        setattr(request, "_messages", messages)

        response = views.train_view(request)
        self.assertEqual(response.status_code, 200)

        model_path = self.data_root / "svc.sav"
        classes_path = self.data_root / "classes.npy"
        self.assertTrue(model_path.exists())
        self.assertTrue(classes_path.exists())

        encrypted_model = model_path.read_bytes()
        decrypted_model = decrypt_bytes(encrypted_model)
        self.assertNotEqual(encrypted_model, decrypted_model)
        loaded_model = pickle.loads(decrypted_model)
        self.assertIsInstance(loaded_model, DummyModel)

        encrypted_classes = classes_path.read_bytes()
        decrypted_classes = decrypt_bytes(encrypted_classes)
        self.assertNotEqual(encrypted_classes, decrypted_classes)
        class_names = np.load(io.BytesIO(decrypted_classes), allow_pickle=True)
        self.assertListEqual(class_names.tolist(), ["alice", "bob"])

        mock_get_embedding.reset_mock()
        mock_deepface.represent.reset_mock()
        mock_deepface.represent.side_effect = lambda *args, **kwargs: np.array([[0.1, 0.2]])

        consumer = MagicMock()
        consumer.__enter__.return_value = consumer
        consumer.__exit__.return_value = False
        consumer.read.side_effect = [np.zeros((10, 10, 3), dtype=np.uint8), None]
        manager = MagicMock()
        manager.frame_consumer.return_value = consumer

        request_mark = self.factory.get("/mark_attendance/")
        request_mark.user = staff_user
        setattr(request_mark, "session", {})
        messages_mark = FallbackStorage(request_mark)
        setattr(request_mark, "_messages", messages_mark)

        captured_models: list[DummyModel] = []

        with (
            patch.object(views, "_load_dataset_embeddings_for_matching") as mock_loader,
            patch.object(views, "get_webcam_manager", return_value=manager),
            patch.object(views, "cv2") as mock_cv2,
            patch.object(views, "_is_headless_environment", return_value=True),
            patch.object(views, "time") as mock_time,
            patch.object(views, "update_attendance_in_db_in") as mock_update_db,
            patch.object(views, "_predict_identity_from_embedding") as mock_predict,
        ):
            mock_loader.return_value = [
                {
                    "identity": str(alice_path),
                    "embedding": np.array([0.1, 0.2], dtype=float),
                    "username": "alice",
                }
            ]
            mock_time.sleep.return_value = None
            mock_cv2.waitKey.return_value = ord("q")

            def _predict(*args, **kwargs):
                captured_models.append(args[3])
                return "alice", False, {"x": 1, "y": 1, "w": 2, "h": 2}

            mock_predict.side_effect = _predict

            views.mark_attendance_view(request_mark, "in")

        self.assertTrue(captured_models, "Encrypted model was not provided to predictor")
        self.assertIsInstance(captured_models[0], DummyModel)
        mock_update_db.assert_called_once()
        attendance_payload = mock_update_db.call_args.args[0]
        self.assertTrue(attendance_payload.get("alice"))
        self.assertEqual(mock_loader.call_count, 2)
        mock_deepface.represent.assert_called()
