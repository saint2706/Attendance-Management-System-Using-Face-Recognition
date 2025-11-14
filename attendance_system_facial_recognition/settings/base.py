"""
Django settings for the Smart Attendance System project.

This file contains the configuration for the Django project, including database settings,
installed applications, middleware, and custom application-specific parameters.
It is configured to read sensitive values from environment variables for security.
"""

import os
import sys
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

import dj_database_url
from cryptography.fernet import Fernet

# Define the project's base directory.
# `BASE_DIR` points to the root of the Django project.
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# --- Security Settings ---


def _get_bool_env(var_name: str, default: bool = False) -> bool:
    """Return a boolean from an environment variable."""
    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


def _get_int_env(var_name: str, default: int) -> int:
    """Return a positive integer from an environment variable."""
    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ImproperlyConfigured(f"{var_name} must be an integer if provided.") from exc
    if value <= 0:
        raise ImproperlyConfigured(f"{var_name} must be a positive integer.")
    return value


def _parse_int_env(var_name: str, default: int, *, minimum: int | None = None) -> int:
    """Return an integer from the environment, enforcing an optional minimum."""
    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ImproperlyConfigured(f"{var_name} must be an integer if provided.") from exc

    if minimum is not None and value < minimum:
        raise ImproperlyConfigured(f"{var_name} must be >= {minimum} if provided.")

    return value


def _get_float_env(
    var_name: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    """Return a float from the environment with optional lower bound enforcement."""
    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ImproperlyConfigured(f"{var_name} must be a float if provided.") from exc

    if minimum is not None and value < minimum:
        raise ImproperlyConfigured(f"{var_name} must be >= {minimum} if provided.")

    return value


# Detect if we're running tests
TESTING = "test" in sys.argv or (len(sys.argv) > 0 and "pytest" in sys.argv[0])

DEFAULT_SECRET_KEY = "a-secure-default-key-for-development-only"

# DEBUG: A boolean that turns on/off debug mode.
# Never run with debug mode turned on in a production environment.
# The value is read from an environment variable, defaulting to False for safety.
# Automatically enable DEBUG mode when running tests.
DEBUG = _get_bool_env("DJANGO_DEBUG", default=TESTING)

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", DEFAULT_SECRET_KEY)
if SECRET_KEY == DEFAULT_SECRET_KEY and not DEBUG:
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY must be set to a secure value when DJANGO_DEBUG is not enabled."
    )


def _load_data_encryption_key() -> bytes:
    """Load the symmetric encryption key used for sensitive assets."""
    key = os.environ.get("DATA_ENCRYPTION_KEY")
    if key:
        key_bytes = key.encode()
        try:
            Fernet(key_bytes)
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive programming
            raise ImproperlyConfigured(
                "DATA_ENCRYPTION_KEY must be a valid 32-byte base64-encoded Fernet key."
            ) from exc
        return key_bytes

    if DEBUG or TESTING:
        return Fernet.generate_key()

    raise ImproperlyConfigured(
        "DATA_ENCRYPTION_KEY environment variable must be set in production environments."
    )


DATA_ENCRYPTION_KEY = _load_data_encryption_key()


def _load_face_data_encryption_key() -> bytes:
    """Load the Fernet key used to encrypt cached facial encodings."""
    key = os.environ.get("FACE_DATA_ENCRYPTION_KEY")
    if key:
        key_bytes = key.encode()
        try:
            Fernet(key_bytes)
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive programming
            raise ImproperlyConfigured(
                "FACE_DATA_ENCRYPTION_KEY must be a valid 32-byte base64-encoded Fernet key."
            ) from exc
        return key_bytes

    if DEBUG or TESTING:
        return Fernet.generate_key()

    raise ImproperlyConfigured(
        "FACE_DATA_ENCRYPTION_KEY environment variable must be set in production environments."
    )


FACE_DATA_ENCRYPTION_KEY = _load_face_data_encryption_key()

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

allowed_hosts_env = os.environ.get("DJANGO_ALLOWED_HOSTS")
if allowed_hosts_env:
    ALLOWED_HOSTS = [host.strip() for host in allowed_hosts_env.split(",") if host.strip()]
elif DEBUG:
    ALLOWED_HOSTS = ["localhost", "127.0.0.1", "[::1]"]
else:
    raise ImproperlyConfigured(
        "DJANGO_ALLOWED_HOSTS must be provided (comma separated) when DJANGO_DEBUG is not enabled."
    )


if DEBUG:
    SECURE_SSL_REDIRECT = False
    SECURE_HSTS_SECONDS = 0
    SECURE_HSTS_INCLUDE_SUBDOMAINS = False
    SECURE_HSTS_PRELOAD = False
    SESSION_COOKIE_SECURE = False
    CSRF_COOKIE_SECURE = False
else:
    SECURE_SSL_REDIRECT = _get_bool_env("SECURE_SSL_REDIRECT", default=True)
    SECURE_HSTS_SECONDS = _parse_int_env("SECURE_HSTS_SECONDS", 3600, minimum=0)
    SECURE_HSTS_INCLUDE_SUBDOMAINS = _get_bool_env("SECURE_HSTS_INCLUDE_SUBDOMAINS", default=True)
    SECURE_HSTS_PRELOAD = _get_bool_env("SECURE_HSTS_PRELOAD", default=True)
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True


INSTALLED_APPS = [
    "users.apps.UsersConfig",
    "recognition.apps.RecognitionConfig",
    "silk",
    "django_rq",
    "django_ratelimit",
    "crispy_forms",
    "crispy_bootstrap5",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

MIDDLEWARE = [
    "silk.middleware.SilkyMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "attendance_system_facial_recognition.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "attendance_system_facial_recognition.wsgi.application"

default_db_url = os.environ.get("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'db.sqlite3').as_posix()}")
conn_max_age = _get_int_env("DATABASE_CONN_MAX_AGE", 0)
database_config = dj_database_url.parse(default_db_url, conn_max_age=conn_max_age)

if _get_bool_env("DATABASE_SSL_REQUIRE", default=False):
    database_config.setdefault("OPTIONS", {})["sslmode"] = "require"

DATABASES = {"default": database_config}

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "unique-snowflake",
    }
}

RQ_QUEUES = {
    "default": {
        "HOST": os.environ.get("REDIS_HOST", "127.0.0.1"),
        "PORT": _get_int_env("REDIS_PORT", 6379),
        "DB": _get_int_env("REDIS_DB", 0),
        "PASSWORD": os.environ.get("REDIS_PASSWORD"),
        "DEFAULT_TIMEOUT": _get_int_env("RQ_DEFAULT_TIMEOUT", 360),
    }
}
RQ_SHOW_ADMIN = _get_bool_env("RQ_SHOW_ADMIN", default=False)

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
MEDIA_URL = os.environ.get("DJANGO_MEDIA_URL", "/media/")
MEDIA_ROOT = Path(os.environ.get("DJANGO_MEDIA_ROOT", BASE_DIR / "media"))
ATTENDANCE_GRAPHS_ROOT = MEDIA_ROOT / "attendance_graphs"

CRISPY_TEMPLATE_PACK = "bootstrap5"
LOGIN_URL = "login"
LOGOUT_REDIRECT_URL = "home"
LOGIN_REDIRECT_URL = "dashboard"

SILKY_AUTHENTICATION = True
SILKY_AUTHORISATION = True


def _silky_permissions(user):
    """Check if user has permission to access Silk profiler."""
    return user.is_staff


SILKY_PERMISSIONS = _silky_permissions

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

SILENCED_SYSTEM_CHECKS = ["django_ratelimit.E003", "django_ratelimit.W001"]

SESSION_COOKIE_SECURE = _get_bool_env("DJANGO_SESSION_COOKIE_SECURE", default=not DEBUG)
SESSION_COOKIE_HTTPONLY = _get_bool_env("DJANGO_SESSION_COOKIE_HTTPONLY", default=True)
CSRF_COOKIE_SECURE = _get_bool_env("DJANGO_CSRF_COOKIE_SECURE", default=not DEBUG)
SESSION_COOKIE_SAMESITE = os.environ.get("DJANGO_SESSION_COOKIE_SAMESITE", "Lax")
SESSION_COOKIE_AGE = _get_int_env("DJANGO_SESSION_COOKIE_AGE", 1800)
SESSION_EXPIRE_AT_BROWSER_CLOSE = _get_bool_env("DJANGO_SESSION_EXPIRE_AT_BROWSER_CLOSE", False)

RECOGNITION_DISTANCE_THRESHOLD = _get_float_env("RECOGNITION_DISTANCE_THRESHOLD", 0.4)

DEEPFACE_OPTIMIZATIONS = {
    "backend": os.environ.get("RECOGNITION_DEEPFACE_BACKEND", "opencv"),
    "model": os.environ.get("RECOGNITION_DEEPFACE_MODEL", "Facenet"),
    "detector_backend": os.environ.get("RECOGNITION_DEEPFACE_DETECTOR", "ssd"),
    "distance_metric": os.environ.get("RECOGNITION_DEEPFACE_DISTANCE_METRIC", "euclidean_l2"),
    "enforce_detection": _get_bool_env("RECOGNITION_DEEPFACE_ENFORCE_DETECTION", False),
    "anti_spoofing": _get_bool_env("RECOGNITION_DEEPFACE_ANTI_SPOOFING", True),
}

RATELIMIT_USE_CACHE = "default"
RECOGNITION_ATTENDANCE_RATE_LIMIT = os.environ.get("RECOGNITION_ATTENDANCE_RATE_LIMIT", "5/m")
RECOGNITION_ATTENDANCE_RATE_LIMIT_METHODS = tuple(
    m.strip().upper()
    for m in os.environ.get("RECOGNITION_ATTENDANCE_RATE_LIMIT_METHODS", "POST").split(",")
)

# Logging configuration
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {process:d} {thread:d} {message}",
            "style": "{",
        },
        "simple": {
            "format": "{levelname} {message}",
            "style": "{",
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(module)s %(lineno)d %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json" if not DEBUG else "simple",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": os.getenv("DJANGO_LOG_LEVEL", "INFO"),
            "propagate": False,
        },
        "recognition": {
            "handlers": ["console"],
            "level": os.getenv("APP_LOG_LEVEL", "DEBUG"),
            "propagate": False,
        },
    },
}
