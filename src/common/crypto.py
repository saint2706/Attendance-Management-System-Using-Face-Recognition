"""Centralised Fernet helpers for encrypting sensitive application data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

BytesLike = Union[bytes, bytearray, memoryview]


def _coerce_key_bytes(key: BytesLike | str) -> bytes:
    """Normalise the configured Fernet key to ``bytes``."""

    if isinstance(key, str):
        return key.encode()
    return bytes(key)


@dataclass(slots=True)
class _FernetWrapper:
    """Lazily instantiate a Fernet cipher using a Django setting."""

    setting_name: str
    key_override: BytesLike | str | None = None
    _cipher: Fernet | None = None

    def _resolve_key(self) -> bytes:
        key = self.key_override
        if key is None:
            key = getattr(settings, self.setting_name, None)
        if key is None:
            raise ImproperlyConfigured(f"{self.setting_name} is not configured.")

        key_bytes = _coerce_key_bytes(key)
        try:
            Fernet(key_bytes)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ImproperlyConfigured(f"{self.setting_name} is invalid.") from exc
        return key_bytes

    def _get_cipher(self) -> Fernet:
        if self._cipher is None:
            self._cipher = Fernet(self._resolve_key())
        return self._cipher

    def encrypt(self, payload: BytesLike) -> bytes:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("encrypt expects a bytes-like object")
        return self._get_cipher().encrypt(bytes(payload))

    def decrypt(self, token: BytesLike) -> bytes:
        if not isinstance(token, (bytes, bytearray, memoryview)):
            raise TypeError("decrypt expects a bytes-like object")
        return self._get_cipher().decrypt(bytes(token))

    def encrypt_encoding(self, encoding: np.ndarray) -> bytes:
        if not isinstance(encoding, np.ndarray):
            raise TypeError("encrypt_encoding expects a numpy.ndarray")
        return self.encrypt(encoding.astype(np.float64).tobytes())

    def decrypt_encoding(self, token: BytesLike, dtype: np.dtype = np.float64) -> np.ndarray:
        decrypted = self.decrypt(token)
        return np.frombuffer(decrypted, dtype=dtype)


class FaceDataEncryption:
    """Provide helpers for encrypting facial encodings with Fernet."""

    def __init__(self, key: BytesLike | str | None = None) -> None:
        self._helper = _FernetWrapper("FACE_DATA_ENCRYPTION_KEY", key_override=key)

    def encrypt(self, data: BytesLike) -> bytes:
        return self._helper.encrypt(data)

    def decrypt(self, token: BytesLike) -> bytes:
        return self._helper.decrypt(token)

    def encrypt_encoding(self, encoding: np.ndarray) -> bytes:
        return self._helper.encrypt_encoding(encoding)

    def decrypt_encoding(self, token: BytesLike, dtype: np.dtype = np.float64) -> np.ndarray:
        return self._helper.decrypt_encoding(token, dtype=dtype)


_data_encryption = _FernetWrapper("DATA_ENCRYPTION_KEY")
_face_encryption = _FernetWrapper("FACE_DATA_ENCRYPTION_KEY")


def encrypt_bytes(data: BytesLike) -> bytes:
    """Encrypt an arbitrary payload using the data encryption key."""

    return _data_encryption.encrypt(data)


def decrypt_bytes(token: BytesLike) -> bytes:
    """Decrypt data previously encrypted via :func:`encrypt_bytes`."""

    return _data_encryption.decrypt(token)


def encrypt_face_bytes(data: BytesLike) -> bytes:
    """Encrypt bytes with the facial data encryption key."""

    return _face_encryption.encrypt(data)


def decrypt_face_bytes(token: BytesLike) -> bytes:
    """Decrypt bytes encrypted via :func:`encrypt_face_bytes`."""

    return _face_encryption.decrypt(token)


def encrypt_face_encoding(encoding: np.ndarray) -> bytes:
    """Encrypt a numpy facial encoding array."""

    return _face_encryption.encrypt_encoding(encoding)


def decrypt_face_encoding(token: BytesLike, dtype: np.dtype = np.float64) -> np.ndarray:
    """Decrypt a stored facial encoding back to an array."""

    return _face_encryption.decrypt_encoding(token, dtype=dtype)


__all__ = [
    "FaceDataEncryption",
    "InvalidToken",
    "decrypt_bytes",
    "decrypt_face_bytes",
    "decrypt_face_encoding",
    "encrypt_bytes",
    "encrypt_face_bytes",
    "encrypt_face_encoding",
]
