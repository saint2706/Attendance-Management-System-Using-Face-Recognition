"""Regression tests for the rotate_encryption_keys management command."""

from __future__ import annotations

import os
from pathlib import Path

import django
from django.conf import settings as django_settings
from django.core.management import call_command
from django.test import override_settings

import numpy as np
import pytest
from cryptography.fernet import Fernet, InvalidToken


pytestmark = [pytest.mark.django_db]

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")
os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DJANGO_DEBUG", "1")
if not django.apps.apps.ready:
    django.setup()


def _encrypt_with_key(key: bytes, payload: bytes) -> bytes:
    return Fernet(key).encrypt(payload)


@override_settings(USE_TZ=True)
def test_command_reencrypts_all_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    dataset_root = data_root / "training_dataset"
    encodings_root = data_root / "encodings"

    data_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    encodings_root.mkdir(parents=True, exist_ok=True)

    old_data_key = Fernet.generate_key()
    old_face_key = Fernet.generate_key()
    new_data_key = Fernet.generate_key()
    new_face_key = Fernet.generate_key()

    django_settings.DATA_ENCRYPTION_KEY = old_data_key
    django_settings.FACE_DATA_ENCRYPTION_KEY = old_face_key

    dataset_file = dataset_root / "1.jpg"
    dataset_file.write_bytes(_encrypt_with_key(old_data_key, b"dataset"))

    model_path = data_root / "svc.sav"
    classes_path = data_root / "classes.npy"
    model_path.write_bytes(_encrypt_with_key(old_data_key, b"model-bytes"))
    classes_path.write_bytes(_encrypt_with_key(old_data_key, b"classes-bytes"))

    face_path = encodings_root / "user" / "encodings.npy.enc"
    face_path.parent.mkdir(parents=True, exist_ok=True)
    face_path.write_bytes(_encrypt_with_key(old_face_key, b"face-bytes"))

    call_command(
        "rotate_encryption_keys",
        new_data_key=new_data_key.decode(),
        new_face_key=new_face_key.decode(),
        data_root=data_root,
        dataset_root=dataset_root,
    )

    for encrypted_path, expected_key, legacy_key in [
        (dataset_file, new_data_key, old_data_key),
        (model_path, new_data_key, old_data_key),
        (classes_path, new_data_key, old_data_key),
        (face_path, new_face_key, old_face_key),
    ]:
        decrypted = Fernet(expected_key).decrypt(encrypted_path.read_bytes())
        assert decrypted
        with pytest.raises(InvalidToken):
            Fernet(legacy_key).decrypt(encrypted_path.read_bytes())


@override_settings(USE_TZ=True)
def test_command_supports_dry_run(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    dataset_root = data_root / "training_dataset"
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    key = Fernet.generate_key()
    django_settings.DATA_ENCRYPTION_KEY = key
    django_settings.FACE_DATA_ENCRYPTION_KEY = key

    dataset_file = dataset_root / "1.jpg"
    dataset_file.write_bytes(_encrypt_with_key(key, b"dataset"))

    before = dataset_file.read_bytes()
    call_command(
        "rotate_encryption_keys",
        new_data_key=key.decode(),
        new_face_key=key.decode(),
        data_root=data_root,
        dataset_root=dataset_root,
        dry_run=True,
    )
    after = dataset_file.read_bytes()

    assert before == after
