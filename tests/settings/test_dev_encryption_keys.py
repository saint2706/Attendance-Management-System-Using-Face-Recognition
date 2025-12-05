"""Regression coverage for development encryption key handling."""

from __future__ import annotations

import importlib
import json
import sys

import pytest
from cryptography.fernet import Fernet


def _reload_base_settings():
    for module in [
        "attendance_system_facial_recognition.settings.base",
        "attendance_system_facial_recognition.settings",
    ]:
        sys.modules.pop(module, None)
    return importlib.import_module("attendance_system_facial_recognition.settings.base")


@pytest.fixture(autouse=True)
def _reset_environment(monkeypatch):
    """Ensure encryption-specific environment variables do not leak between tests."""

    for env_var in [
        "DATA_ENCRYPTION_KEY",
        "FACE_DATA_ENCRYPTION_KEY",
        "DJANGO_DEBUG",
    ]:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("DJANGO_DEBUG", "1")
    yield


def test_dev_keys_are_persisted_and_reusable(tmp_path, monkeypatch):
    cache_path = tmp_path / "dev_keys.json"
    monkeypatch.setenv("DEV_ENCRYPTION_KEY_FILE", str(cache_path))
    monkeypatch.setenv("LOCAL_ENV_PATH", str(tmp_path / ".env"))

    settings_base = _reload_base_settings()

    first_data_key = settings_base.DATA_ENCRYPTION_KEY
    first_face_key = settings_base.FACE_DATA_ENCRYPTION_KEY

    payload_token = Fernet(first_data_key).encrypt(b"payload")
    face_payload = Fernet(first_face_key).encrypt(b"face-bytes")

    assert cache_path.exists()
    cache = json.loads(cache_path.read_text())
    assert cache["DATA_ENCRYPTION_KEY"] == first_data_key.decode()
    assert cache["FACE_DATA_ENCRYPTION_KEY"] == first_face_key.decode()

    settings_base = _reload_base_settings()

    assert settings_base.DATA_ENCRYPTION_KEY == first_data_key
    assert settings_base.FACE_DATA_ENCRYPTION_KEY == first_face_key
    assert Fernet(settings_base.DATA_ENCRYPTION_KEY).decrypt(payload_token) == b"payload"
    assert Fernet(settings_base.FACE_DATA_ENCRYPTION_KEY).decrypt(face_payload) == b"face-bytes"


def test_dotenv_values_are_respected(tmp_path, monkeypatch):
    cache_path = tmp_path / "dev_keys.json"
    dotenv_path = tmp_path / ".env"
    data_key = Fernet.generate_key()
    face_key = Fernet.generate_key()
    dotenv_path.write_text(
        f"DATA_ENCRYPTION_KEY={data_key.decode()}\nFACE_DATA_ENCRYPTION_KEY='{face_key.decode()}'\n"
    )

    monkeypatch.setenv("DEV_ENCRYPTION_KEY_FILE", str(cache_path))
    monkeypatch.setenv("LOCAL_ENV_PATH", str(dotenv_path))

    settings_base = _reload_base_settings()

    assert settings_base.DATA_ENCRYPTION_KEY == data_key
    assert settings_base.FACE_DATA_ENCRYPTION_KEY == face_key
    assert not cache_path.exists()
