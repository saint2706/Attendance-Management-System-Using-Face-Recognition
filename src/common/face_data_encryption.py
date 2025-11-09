"""Helpers for encrypting sensitive face data encodings."""

from __future__ import annotations

from typing import Union

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

BytesLike = Union[bytes, bytearray, memoryview]


class FaceDataEncryption:
    """Provide Fernet-backed helpers for encrypting facial encodings."""

    def __init__(self, key: BytesLike | str | None = None) -> None:
        self._key_override = key
        self._cipher: Fernet | None = None

    def _resolve_key(self) -> bytes:
        """Return the Fernet key configured for face data encryption."""

        key = self._key_override
        if key is None:
            key = getattr(settings, "FACE_DATA_ENCRYPTION_KEY", None)
        if key is None:
            raise ImproperlyConfigured("FACE_DATA_ENCRYPTION_KEY is not configured.")
        if isinstance(key, str):
            key_bytes = key.encode()
        else:
            key_bytes = bytes(key)
        try:
            Fernet(key_bytes)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive programming
            raise ImproperlyConfigured("FACE_DATA_ENCRYPTION_KEY is invalid.") from exc
        return key_bytes

    def _get_cipher(self) -> Fernet:
        if self._cipher is None:
            self._cipher = Fernet(self._resolve_key())
        return self._cipher

    def encrypt(self, data: BytesLike) -> bytes:
        """Encrypt an arbitrary payload of bytes."""

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("encrypt expects a bytes-like object")
        return self._get_cipher().encrypt(bytes(data))

    def decrypt(self, token: BytesLike) -> bytes:
        """Decrypt an encrypted payload."""

        if not isinstance(token, (bytes, bytearray, memoryview)):
            raise TypeError("decrypt expects a bytes-like object")
        return self._get_cipher().decrypt(bytes(token))

    def encrypt_encoding(self, encoding: np.ndarray) -> bytes:
        """Encrypt a numpy facial encoding array."""

        if not isinstance(encoding, np.ndarray):
            raise TypeError("encrypt_encoding expects a numpy.ndarray")
        return self.encrypt(encoding.astype(np.float64).tobytes())

    def decrypt_encoding(
        self, encrypted_data: BytesLike, dtype: np.dtype = np.float64
    ) -> np.ndarray:
        """Decrypt an encrypted encoding back into a numpy array."""

        decrypted = self.decrypt(encrypted_data)
        return np.frombuffer(decrypted, dtype=dtype)


__all__ = ["FaceDataEncryption", "InvalidToken"]
