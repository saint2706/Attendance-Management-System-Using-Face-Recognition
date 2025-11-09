"""Unit tests for incremental training tasks."""

from __future__ import annotations

import io
import pickle
from typing import List
from unittest.mock import patch

import os

import django
import django_rq
import numpy as np
import pytest
from cryptography.fernet import Fernet

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")
django.setup()

from src.common import decrypt_bytes

from recognition import tasks
from django.conf import settings as django_settings


@pytest.mark.django_db
def test_incremental_training_updates_only_target_employee(tmp_path, monkeypatch):
    """The incremental job should update only the targeted employee and refresh the model."""

    django_settings.DATA_ENCRYPTION_KEY = Fernet.generate_key()
    django_settings.FACE_DATA_ENCRYPTION_KEY = Fernet.generate_key()
    django_settings.RQ_QUEUES = {"default": {"URL": "redis://localhost:6379/0", "ASYNC": False}}

    data_root = tmp_path / "face_data"
    enc_root = data_root / "encodings"

    monkeypatch.setattr(tasks, "DATA_ROOT", data_root)
    monkeypatch.setattr(tasks, "ENCODINGS_DIR", enc_root)
    monkeypatch.setattr(tasks, "MODEL_PATH", data_root / "svc.sav")
    monkeypatch.setattr(tasks, "CLASSES_PATH", data_root / "classes.npy")

    enc_root.mkdir(parents=True, exist_ok=True)

    tasks.save_employee_encodings("alice", [[1.0, 1.0]])
    tasks.save_employee_encodings("bob", [[2.0, 2.0]])

    new_image_paths: List[str] = []
    for index in range(2):
        image_path = tmp_path / f"bob_{index}.jpg"
        image_path.write_bytes(b"encrypted")
        new_image_paths.append(str(image_path))

    new_embeddings = [np.array([3.0, 3.0]), np.array([4.0, 4.0])]

    with (
        patch.object(tasks, "_get_or_compute_cached_embedding", side_effect=new_embeddings) as mock_get,
        patch.object(tasks._dataset_embedding_cache, "invalidate") as mock_invalidate,
        patch.object(django_rq, "enqueue", side_effect=lambda func, *a, **kw: func(*a, **kw)),
    ):
        django_rq.enqueue(tasks.incremental_face_training, "bob", new_image_paths)

    assert mock_get.call_count == len(new_image_paths)
    mock_invalidate.assert_called_once()

    alice_encodings = tasks.load_existing_encodings("alice")
    np.testing.assert_allclose(alice_encodings, np.array([[1.0, 1.0]]))

    bob_encodings = tasks.load_existing_encodings("bob")
    np.testing.assert_allclose(
        bob_encodings,
        np.array(
            [
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        ),
    )

    assert tasks.MODEL_PATH.exists()
    encrypted_model = tasks.MODEL_PATH.read_bytes()
    model = pickle.loads(decrypt_bytes(encrypted_model))
    assert model.__class__.__name__ == "SGDClassifier"
    assert set(model.classes_) == {"alice", "bob"}

    assert tasks.CLASSES_PATH.exists()
    decrypted_classes = decrypt_bytes(tasks.CLASSES_PATH.read_bytes())
    class_names = np.load(io.BytesIO(decrypted_classes), allow_pickle=True)
    assert sorted(class_names.tolist()) == ["alice", "bob"]
