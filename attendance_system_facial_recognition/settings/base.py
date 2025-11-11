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
        # During development and automated tests fall back to an ephemeral key
        # to avoid leaking plaintext assets to disk.
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

# ALLOWED_HOSTS: A list of strings representing the host/domain names that this Django site can serve.
# When DJANGO_DEBUG is False the value must be explicitly provided.
allowed_hosts_env = os.environ.get("DJANGO_ALLOWED_HOSTS")
if allowed_hosts_env:
    ALLOWED_HOSTS = [host.strip() for host in allowed_hosts_env.split(",") if host.strip()]
elif DEBUG:
    ALLOWED_HOSTS = ["localhost", "127.0.0.1", "[::1]"]
else:
    raise ImproperlyConfigured(
        "DJANGO_ALLOWED_HOSTS must be provided (comma separated) when DJANGO_DEBUG is not enabled."
    )


# --- Application Configuration ---

INSTALLED_APPS = [
    # Custom applications for this project
    "users.apps.UsersConfig",
    "recognition.apps.RecognitionConfig",
    # Third-party packages
    "silk",
    "django_rq",
    "django_ratelimit",
    "crispy_forms",
    "crispy_bootstrap5",
    # Core Django applications
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

# The root URL configuration module for the project.
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

# WSGI application entry point for production servers.
WSGI_APPLICATION = "attendance_system_facial_recognition.wsgi.application"


# --- Database Configuration ---
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

default_db_url = os.environ.get("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'db.sqlite3').as_posix()}")

conn_max_age_raw = os.environ.get("DATABASE_CONN_MAX_AGE")
if conn_max_age_raw is None:
    conn_max_age = 0
else:
    try:
        conn_max_age = int(conn_max_age_raw)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ImproperlyConfigured("DATABASE_CONN_MAX_AGE must be an integer if provided.") from exc
    if conn_max_age < 0:
        raise ImproperlyConfigured("DATABASE_CONN_MAX_AGE must be zero or positive.")

database_config = dj_database_url.parse(default_db_url, conn_max_age=conn_max_age)

if _get_bool_env("DATABASE_SSL_REQUIRE", default=False):
    database_config.setdefault("OPTIONS", {})["sslmode"] = "require"

DATABASES = {
    "default": database_config,
}


# --- Cache Configuration ---
# For django-ratelimit, we use LocMemCache in development/testing.
# While LocMemCache is not ideal for production (not shared across processes),
# it works for single-process deployments and CI/testing.
# For production multi-process deployments, configure Redis or Memcached.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "unique-snowflake",
    }
}


# --- RQ (Redis Queue) Configuration ---


def _build_rq_queue_settings() -> dict[str, dict[str, object]]:
    """Return the django-rq queue configuration sourced from the environment."""

    default_async = not TESTING and _get_bool_env("RQ_ASYNC", default=True)
    default_timeout = _parse_int_env("RQ_DEFAULT_TIMEOUT", 300, minimum=1)

    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        default_queue: dict[str, object] = {
            "URL": redis_url,
            "DEFAULT_TIMEOUT": default_timeout,
            "ASYNC": default_async,
        }
        return {"default": default_queue}

    host = os.environ.get("REDIS_HOST", "127.0.0.1")
    port = _parse_int_env("REDIS_PORT", 6379, minimum=1)
    db_index = _parse_int_env("REDIS_DB", 0, minimum=0)
    password = os.environ.get("REDIS_PASSWORD")

    default_queue = {
        "HOST": host,
        "PORT": port,
        "DB": db_index,
        "DEFAULT_TIMEOUT": default_timeout,
        "ASYNC": default_async,
    }

    if password:
        default_queue["PASSWORD"] = password

    return {"default": default_queue}


RQ_QUEUES = _build_rq_queue_settings()
RQ_SHOW_ADMIN = _get_bool_env("RQ_SHOW_ADMIN", default=False)


# --- Password Validation ---
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# --- Internationalization ---
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"  # Set to the appropriate timezone
USE_I18N = True
USE_TZ = True  # Enable timezone-aware datetimes


# --- Static & Media Files Configuration ---
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = "/static/"
# Django automatically collects static files from app-specific static directories
# (e.g., recognition/static). Additional directories can be added here if needed.

MEDIA_URL = os.environ.get("DJANGO_MEDIA_URL", "/media/")
MEDIA_ROOT = Path(os.environ.get("DJANGO_MEDIA_ROOT", BASE_DIR / "media"))

# Directory used by the reporting views to persist generated charts.
ATTENDANCE_GRAPHS_ROOT = MEDIA_ROOT / "attendance_graphs"

# --- Crispy Forms Configuration ---

# Specifies that django-crispy-forms should use Bootstrap 5 templates.
CRISPY_TEMPLATE_PACK = "bootstrap5"

# --- Authentication and Redirects ---

# The URL to redirect to for login when using the @login_required decorator.
LOGIN_URL = "login"

# The URL to redirect to after a user logs out.
LOGOUT_REDIRECT_URL = "home"

# The default URL to redirect to after a user logs in.
LOGIN_REDIRECT_URL = "dashboard"

# --- Performance Monitoring (django-silk) ---


def _silky_staff_only(user) -> bool:
    """Restrict Silk dashboards to authenticated staff members."""

    return bool(user and user.is_authenticated and user.is_staff)


SILKY_AUTHENTICATION = True
SILKY_AUTHORISATION = True
SILKY_PERMISSIONS = _silky_staff_only

# --- Model Field Configuration ---

# Specifies the default primary key field type for new models.
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Custom Application Settings ---

# --- System Check Silencing ---
# Silence django-ratelimit checks for LocMemCache in development/testing.
# LocMemCache works fine for single-process deployments and CI/testing environments.
# For production, configure a shared cache backend (Redis/Memcached).
SILENCED_SYSTEM_CHECKS = [
    "django_ratelimit.E003",  # LocMemCache not a shared cache
    "django_ratelimit.W001",  # LocMemCache not officially supported
]

# --- Session & Cookie Settings ---

SESSION_COOKIE_SECURE = _get_bool_env("DJANGO_SESSION_COOKIE_SECURE", default=not DEBUG)
SESSION_COOKIE_HTTPONLY = _get_bool_env("DJANGO_SESSION_COOKIE_HTTPONLY", default=True)
CSRF_COOKIE_SECURE = _get_bool_env("DJANGO_CSRF_COOKIE_SECURE", default=not DEBUG)

_session_cookie_samesite = os.environ.get("DJANGO_SESSION_COOKIE_SAMESITE")
if _session_cookie_samesite:
    _normalized_samesite = _session_cookie_samesite.strip().lower()
    _valid_samesite_values = {"lax": "Lax", "strict": "Strict", "none": "None"}
    if _normalized_samesite not in _valid_samesite_values:
        raise ImproperlyConfigured(
            "DJANGO_SESSION_COOKIE_SAMESITE must be one of: Lax, Strict, or None."
        )
    SESSION_COOKIE_SAMESITE = _valid_samesite_values[_normalized_samesite]
else:
    SESSION_COOKIE_SAMESITE = "Lax"

SESSION_COOKIE_AGE = _get_int_env("DJANGO_SESSION_COOKIE_AGE", default=1800)
SESSION_EXPIRE_AT_BROWSER_CLOSE = _get_bool_env(
    "DJANGO_SESSION_EXPIRE_AT_BROWSER_CLOSE", default=False
)

# Threshold for accepting DeepFace matches when marking attendance.
# Lower values (e.g., 0.3) mean stricter matching, while higher values (e.g., 0.5)
# are more permissive. This can be overridden via an environment variable.
RECOGNITION_DISTANCE_THRESHOLD = float(os.environ.get("RECOGNITION_DISTANCE_THRESHOLD", "0.4"))


def _build_deepface_optimizations() -> dict[str, object]:
    """Return DeepFace tuning parameters with environment overrides."""

    defaults: dict[str, object] = {
        "backend": "opencv",
        "model": "Facenet",
        "detector_backend": "ssd",
        "distance_metric": "euclidean_l2",
        "enforce_detection": False,
        "anti_spoofing": True,
    }

    env_overrides: dict[str, object] = {}

    backend = os.environ.get("RECOGNITION_DEEPFACE_BACKEND")
    if backend:
        env_overrides["backend"] = backend

    model = os.environ.get("RECOGNITION_DEEPFACE_MODEL")
    if model:
        env_overrides["model"] = model

    detector_backend = os.environ.get("RECOGNITION_DEEPFACE_DETECTOR")
    if detector_backend:
        env_overrides["detector_backend"] = detector_backend

    distance_metric = os.environ.get("RECOGNITION_DEEPFACE_DISTANCE_METRIC")
    if distance_metric:
        env_overrides["distance_metric"] = distance_metric

    env_overrides["enforce_detection"] = _get_bool_env(
        "RECOGNITION_DEEPFACE_ENFORCE_DETECTION",
        default=bool(defaults["enforce_detection"]),
    )
    env_overrides["anti_spoofing"] = _get_bool_env(
        "RECOGNITION_DEEPFACE_ANTI_SPOOFING",
        default=bool(defaults["anti_spoofing"]),
    )

    merged = defaults.copy()
    merged.update(env_overrides)
    return merged


DEEPFACE_OPTIMIZATIONS = _build_deepface_optimizations()

# Rate limiting configuration for attendance endpoints. Uses django-ratelimit's
# default cache to track request counts.
RATELIMIT_USE_CACHE = "default"
DEFAULT_ATTENDANCE_RATE_LIMIT = "5/m"
RECOGNITION_ATTENDANCE_RATE_LIMIT = os.environ.get(
    "RECOGNITION_ATTENDANCE_RATE_LIMIT", DEFAULT_ATTENDANCE_RATE_LIMIT
)
_attendance_methods = os.environ.get("RECOGNITION_ATTENDANCE_RATE_LIMIT_METHODS")
if _attendance_methods:
    RECOGNITION_ATTENDANCE_RATE_LIMIT_METHODS = tuple(
        method.strip().upper() for method in _attendance_methods.split(",") if method.strip()
    )
else:
    RECOGNITION_ATTENDANCE_RATE_LIMIT_METHODS = ("POST",)

RECOGNITION_CAMERA_START_ALERT_SECONDS = _get_float_env(
    "RECOGNITION_CAMERA_START_ALERT_SECONDS",
    default=3.0,
    minimum=0.0,
)
RECOGNITION_FRAME_DELAY_ALERT_SECONDS = _get_float_env(
    "RECOGNITION_FRAME_DELAY_ALERT_SECONDS",
    default=0.75,
    minimum=0.0,
)
RECOGNITION_MODEL_LOAD_ALERT_SECONDS = _get_float_env(
    "RECOGNITION_MODEL_LOAD_ALERT_SECONDS",
    default=4.0,
    minimum=0.0,
)
RECOGNITION_WARMUP_ALERT_SECONDS = _get_float_env(
    "RECOGNITION_WARMUP_ALERT_SECONDS",
    default=3.0,
    minimum=0.0,
)
RECOGNITION_LOOP_ALERT_SECONDS = _get_float_env(
    "RECOGNITION_LOOP_ALERT_SECONDS",
    default=1.5,
    minimum=0.0,
)
RECOGNITION_HEALTH_ALERT_HISTORY = _parse_int_env(
    "RECOGNITION_HEALTH_ALERT_HISTORY",
    default=50,
    minimum=1,
)
