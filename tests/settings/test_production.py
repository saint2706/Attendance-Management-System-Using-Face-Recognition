"""Smoke tests for the production settings module."""

from __future__ import annotations

import importlib
import sys
import types


def _reload_production_settings():
    """Force a reload of the production settings module for isolation."""

    for module in [
        "attendance_system_facial_recognition.settings.production",
        "attendance_system_facial_recognition.settings.base",
        "attendance_system_facial_recognition.settings",
    ]:
        sys.modules.pop(module, None)
    return importlib.import_module("attendance_system_facial_recognition.settings.production")


def test_production_database_configuration(monkeypatch):
    fake_celery = types.ModuleType("celery")

    class _DummyCelery:
        def __init__(self, *args, **kwargs):
            pass

        def config_from_object(self, *args, **kwargs):
            return None

        def autodiscover_tasks(self, *args, **kwargs):
            return None

    fake_celery.Celery = _DummyCelery
    monkeypatch.setitem(sys.modules, "celery", fake_celery)

    fake_cryptography = types.ModuleType("cryptography")
    fake_fernet = types.ModuleType("cryptography.fernet")

    class _DummyFernet:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def generate_key() -> bytes:
            return b"0" * 32

    fake_fernet.Fernet = _DummyFernet
    fake_cryptography.fernet = fake_fernet
    monkeypatch.setitem(sys.modules, "cryptography", fake_cryptography)
    monkeypatch.setitem(sys.modules, "cryptography.fernet", fake_fernet)

    monkeypatch.setenv(
        "DJANGO_SETTINGS_MODULE",
        "attendance_system_facial_recognition.settings.production",
    )
    monkeypatch.setenv("DJANGO_ALLOWED_HOSTS", "example.com,localhost")
    monkeypatch.setenv("DJANGO_SECRET_KEY", "test-secret-key-for-tests-only-1234567890")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_NAME", "ci_db")
    monkeypatch.setenv("DB_USER", "ci_user")
    monkeypatch.setenv("DB_PASSWORD", "ci_password")
    monkeypatch.setenv("DB_HOST", "postgres")
    monkeypatch.setenv("DB_PORT", "6543")
    monkeypatch.setenv("DB_CONN_MAX_AGE", "120")

    settings = _reload_production_settings()

    database = settings.DATABASES["default"]
    assert database["ENGINE"] == "django.db.backends.postgresql"
    assert database["NAME"] == "ci_db"
    assert database["USER"] == "ci_user"
    assert database["PASSWORD"] == "ci_password"
    assert database["HOST"] == "postgres"
    assert database["PORT"] == "6543"
    assert database["CONN_MAX_AGE"] == 120


def test_security_headers_default_values(monkeypatch):
    """Test that security headers are set to secure defaults in production."""
    fake_celery = types.ModuleType("celery")

    class _DummyCelery:
        def __init__(self, *args, **kwargs):
            pass

        def config_from_object(self, *args, **kwargs):
            return None

        def autodiscover_tasks(self, *args, **kwargs):
            return None

    fake_celery.Celery = _DummyCelery
    monkeypatch.setitem(sys.modules, "celery", fake_celery)

    fake_cryptography = types.ModuleType("cryptography")
    fake_fernet = types.ModuleType("cryptography.fernet")

    class _DummyFernet:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def generate_key() -> bytes:
            return b"0" * 32

    fake_fernet.Fernet = _DummyFernet
    fake_cryptography.fernet = fake_fernet
    monkeypatch.setitem(sys.modules, "cryptography", fake_cryptography)
    monkeypatch.setitem(sys.modules, "cryptography.fernet", fake_fernet)

    monkeypatch.setenv(
        "DJANGO_SETTINGS_MODULE",
        "attendance_system_facial_recognition.settings.production",
    )
    monkeypatch.setenv("DJANGO_ALLOWED_HOSTS", "example.com")
    monkeypatch.setenv("DJANGO_SECRET_KEY", "test-secret-key-for-tests-only-1234567890")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_NAME", "ci_db")
    monkeypatch.setenv("DB_USER", "ci_user")
    monkeypatch.setenv("DB_PASSWORD", "ci_password")
    monkeypatch.setenv("DB_HOST", "postgres")
    monkeypatch.setenv("DB_PORT", "5432")

    settings = _reload_production_settings()

    # Test security headers are set to secure defaults
    assert settings.SECURE_CONTENT_TYPE_NOSNIFF is True
    assert settings.X_FRAME_OPTIONS == "DENY"
    assert settings.SECURE_REFERRER_POLICY == "strict-origin-when-cross-origin"


def test_secure_proxy_ssl_header_disabled_by_default(monkeypatch):
    """Test that SECURE_PROXY_SSL_HEADER is not set when environment variable is not enabled."""
    fake_celery = types.ModuleType("celery")

    class _DummyCelery:
        def __init__(self, *args, **kwargs):
            pass

        def config_from_object(self, *args, **kwargs):
            return None

        def autodiscover_tasks(self, *args, **kwargs):
            return None

    fake_celery.Celery = _DummyCelery
    monkeypatch.setitem(sys.modules, "celery", fake_celery)

    fake_cryptography = types.ModuleType("cryptography")
    fake_fernet = types.ModuleType("cryptography.fernet")

    class _DummyFernet:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def generate_key() -> bytes:
            return b"0" * 32

    fake_fernet.Fernet = _DummyFernet
    fake_cryptography.fernet = fake_fernet
    monkeypatch.setitem(sys.modules, "cryptography", fake_cryptography)
    monkeypatch.setitem(sys.modules, "cryptography.fernet", fake_fernet)

    monkeypatch.setenv(
        "DJANGO_SETTINGS_MODULE",
        "attendance_system_facial_recognition.settings.production",
    )
    monkeypatch.setenv("DJANGO_ALLOWED_HOSTS", "example.com")
    monkeypatch.setenv("DJANGO_SECRET_KEY", "test-secret-key-for-tests-only-1234567890")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_NAME", "ci_db")
    monkeypatch.setenv("DB_USER", "ci_user")
    monkeypatch.setenv("DB_PASSWORD", "ci_password")
    monkeypatch.setenv("DB_HOST", "postgres")
    monkeypatch.setenv("DB_PORT", "5432")
    # Explicitly ensure DJANGO_SECURE_PROXY_SSL_HEADER is not set
    monkeypatch.delenv("DJANGO_SECURE_PROXY_SSL_HEADER", raising=False)

    settings = _reload_production_settings()

    # SECURE_PROXY_SSL_HEADER should not be defined when disabled
    assert not hasattr(settings, "SECURE_PROXY_SSL_HEADER")


def test_secure_proxy_ssl_header_enabled_when_configured(monkeypatch):
    """Test that SECURE_PROXY_SSL_HEADER is set correctly when environment variable is enabled."""
    fake_celery = types.ModuleType("celery")

    class _DummyCelery:
        def __init__(self, *args, **kwargs):
            pass

        def config_from_object(self, *args, **kwargs):
            return None

        def autodiscover_tasks(self, *args, **kwargs):
            return None

    fake_celery.Celery = _DummyCelery
    monkeypatch.setitem(sys.modules, "celery", fake_celery)

    fake_cryptography = types.ModuleType("cryptography")
    fake_fernet = types.ModuleType("cryptography.fernet")

    class _DummyFernet:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def generate_key() -> bytes:
            return b"0" * 32

    fake_fernet.Fernet = _DummyFernet
    fake_cryptography.fernet = fake_fernet
    monkeypatch.setitem(sys.modules, "cryptography", fake_cryptography)
    monkeypatch.setitem(sys.modules, "cryptography.fernet", fake_fernet)

    monkeypatch.setenv(
        "DJANGO_SETTINGS_MODULE",
        "attendance_system_facial_recognition.settings.production",
    )
    monkeypatch.setenv("DJANGO_ALLOWED_HOSTS", "example.com")
    monkeypatch.setenv("DJANGO_SECRET_KEY", "test-secret-key-for-tests-only-1234567890")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_NAME", "ci_db")
    monkeypatch.setenv("DB_USER", "ci_user")
    monkeypatch.setenv("DB_PASSWORD", "ci_password")
    monkeypatch.setenv("DB_HOST", "postgres")
    monkeypatch.setenv("DB_PORT", "5432")
    # Enable SECURE_PROXY_SSL_HEADER
    monkeypatch.setenv("DJANGO_SECURE_PROXY_SSL_HEADER", "true")

    settings = _reload_production_settings()

    # SECURE_PROXY_SSL_HEADER should be set to trust X-Forwarded-Proto header
    assert hasattr(settings, "SECURE_PROXY_SSL_HEADER")
    assert settings.SECURE_PROXY_SSL_HEADER == ("HTTP_X_FORWARDED_PROTO", "https")


def test_secure_proxy_ssl_header_disabled_with_false_value(monkeypatch):
    """Test that SECURE_PROXY_SSL_HEADER is not set when environment variable is explicitly false."""
    fake_celery = types.ModuleType("celery")

    class _DummyCelery:
        def __init__(self, *args, **kwargs):
            pass

        def config_from_object(self, *args, **kwargs):
            return None

        def autodiscover_tasks(self, *args, **kwargs):
            return None

    fake_celery.Celery = _DummyCelery
    monkeypatch.setitem(sys.modules, "celery", fake_celery)

    fake_cryptography = types.ModuleType("cryptography")
    fake_fernet = types.ModuleType("cryptography.fernet")

    class _DummyFernet:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def generate_key() -> bytes:
            return b"0" * 32

    fake_fernet.Fernet = _DummyFernet
    fake_cryptography.fernet = fake_fernet
    monkeypatch.setitem(sys.modules, "cryptography", fake_cryptography)
    monkeypatch.setitem(sys.modules, "cryptography.fernet", fake_fernet)

    monkeypatch.setenv(
        "DJANGO_SETTINGS_MODULE",
        "attendance_system_facial_recognition.settings.production",
    )
    monkeypatch.setenv("DJANGO_ALLOWED_HOSTS", "example.com")
    monkeypatch.setenv("DJANGO_SECRET_KEY", "test-secret-key-for-tests-only-1234567890")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_NAME", "ci_db")
    monkeypatch.setenv("DB_USER", "ci_user")
    monkeypatch.setenv("DB_PASSWORD", "ci_password")
    monkeypatch.setenv("DB_HOST", "postgres")
    monkeypatch.setenv("DB_PORT", "5432")
    # Explicitly disable SECURE_PROXY_SSL_HEADER
    monkeypatch.setenv("DJANGO_SECURE_PROXY_SSL_HEADER", "false")

    settings = _reload_production_settings()

    # SECURE_PROXY_SSL_HEADER should not be defined when explicitly disabled
    assert not hasattr(settings, "SECURE_PROXY_SSL_HEADER")

