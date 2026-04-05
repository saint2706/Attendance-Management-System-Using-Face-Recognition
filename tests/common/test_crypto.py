from django.core.exceptions import ImproperlyConfigured

import numpy as np
import pytest

from src.common.crypto import (
    FaceDataEncryption,
    _coerce_key_bytes,
    _FernetWrapper,
    _parse_key_material,
    decrypt_bytes,
    decrypt_face_bytes,
    decrypt_face_encoding,
    encrypt_bytes,
    encrypt_face_bytes,
    encrypt_face_encoding,
)


def test_coerce_key_bytes():
    assert _coerce_key_bytes("test") == b"test"
    assert _coerce_key_bytes(b"test") == b"test"
    assert _coerce_key_bytes(bytearray(b"test")) == b"test"


def test_parse_key_material():
    # Single key
    assert _parse_key_material("key1") == [b"key1"]

    # Comma-separated string
    assert _parse_key_material("key1, key2 ,key3") == [b"key1", b"key2", b"key3"]

    # Iterable
    assert _parse_key_material(["key1", b"key2"]) == [b"key1", b"key2"]

    # Empty comma string should raise ImproperlyConfigured
    with pytest.raises(ImproperlyConfigured):
        _parse_key_material(" , ")


class TestFernetWrapper:
    def test_missing_setting(self, settings):
        wrapper = _FernetWrapper("MISSING_KEY")
        with pytest.raises(ImproperlyConfigured, match="MISSING_KEY is not configured"):
            wrapper._resolve_keys()

    def test_encrypt_decrypt_bytes_like(self):
        # We need a valid fernet key for this
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()

        wrapper = _FernetWrapper("TEST_KEY", key_override=key)

        payload = b"test payload"
        token = wrapper.encrypt(payload)
        assert token != payload
        assert wrapper.decrypt(token) == payload

        # Test TypeError
        with pytest.raises(TypeError, match="encrypt expects a bytes-like object"):
            wrapper.encrypt("not bytes")

        with pytest.raises(TypeError, match="decrypt expects a bytes-like object"):
            wrapper.decrypt("not bytes")

    def test_encrypt_decrypt_encoding(self):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()

        wrapper = _FernetWrapper("TEST_KEY", key_override=key)

        encoding = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        token = wrapper.encrypt_encoding(encoding)
        assert token != encoding.tobytes()

        decrypted_encoding = wrapper.decrypt_encoding(token)
        np.testing.assert_array_equal(decrypted_encoding, encoding)

        with pytest.raises(TypeError, match="encrypt_encoding expects a numpy.ndarray"):
            wrapper.encrypt_encoding(b"not an array")


class TestFaceDataEncryption:
    def test_face_data_encryption_methods(self):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()

        fde = FaceDataEncryption(key=key)

        # Test bytes
        payload = b"face data"
        token = fde.encrypt(payload)
        assert fde.decrypt(token) == payload

        # Test encoding
        encoding = np.array([0.5, 0.6], dtype=np.float64)
        token_enc = fde.encrypt_encoding(encoding)
        decrypted_encoding = fde.decrypt_encoding(token_enc)
        np.testing.assert_array_equal(decrypted_encoding, encoding)


def test_global_crypto_functions(settings):
    from cryptography.fernet import Fernet

    key1 = Fernet.generate_key().decode()
    key2 = Fernet.generate_key().decode()

    settings.DATA_ENCRYPTION_KEY = key1
    settings.FACE_DATA_ENCRYPTION_KEY = key2

    # Test data encryption
    payload = b"global data"
    token = encrypt_bytes(payload)
    assert decrypt_bytes(token) == payload

    # Test face bytes encryption
    face_payload = b"global face data"
    face_token = encrypt_face_bytes(face_payload)
    assert decrypt_face_bytes(face_token) == face_payload

    # Test face encoding encryption
    encoding = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    enc_token = encrypt_face_encoding(encoding)
    decrypted_enc = decrypt_face_encoding(enc_token)
    np.testing.assert_array_equal(decrypted_enc, encoding)
