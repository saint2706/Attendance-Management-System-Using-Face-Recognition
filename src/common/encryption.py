"""Utility helpers for symmetric encryption used across the project."""

from __future__ import annotations

from typing import Union

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from cryptography.fernet import Fernet, InvalidToken

BytesLike = Union[bytes, bytearray, memoryview]


def _get_key_bytes() -> bytes:
    """Return the configured Fernet key as bytes."""

    key = getattr(settings, "DATA_ENCRYPTION_KEY", None)
    if key is None:
        raise ImproperlyConfigured("DATA_ENCRYPTION_KEY is not configured.")
    if isinstance(key, str):
        key_bytes = key.encode()
    else:
        key_bytes = bytes(key)
    try:
        Fernet(key_bytes)
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
        raise ImproperlyConfigured("DATA_ENCRYPTION_KEY is invalid.") from exc
    return key_bytes


def _get_cipher() -> Fernet:
    """Instantiate a Fernet cipher for the configured key."""

    return Fernet(_get_key_bytes())


def encrypt_bytes(data: BytesLike) -> bytes:
    """Encrypt the provided payload using Fernet symmetric encryption."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("encrypt_bytes expects a bytes-like object")
    cipher = _get_cipher()
    return cipher.encrypt(bytes(data))


def decrypt_bytes(token: BytesLike) -> bytes:
    """Decrypt an encrypted payload produced by :func:`encrypt_bytes`."""

    if not isinstance(token, (bytes, bytearray, memoryview)):
        raise TypeError("decrypt_bytes expects a bytes-like object")
    cipher = _get_cipher()
    return cipher.decrypt(bytes(token))


__all__ = ["encrypt_bytes", "decrypt_bytes", "InvalidToken"]
