"""
Django settings for the Smart Attendance System project.

This file contains the configuration for the Django project, including database settings,
installed applications, middleware, and custom application-specific parameters.
It is configured to read sensitive values from environment variables for security.
"""

import json
import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from django.core.exceptions import ImproperlyConfigured

import dj_database_url
from cryptography.fernet import Fernet

# Define the project's base directory.
# `BASE_DIR` points to the root of the Django project.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOCAL_ENV_PATH = Path(os.environ.get("LOCAL_ENV_PATH", BASE_DIR / ".env"))
DEV_KEY_CACHE_PATH = Path(
    os.environ.get("DEV_ENCRYPTION_KEY_FILE", BASE_DIR / ".dev_encryption_keys.json")
)


# --- Security Settings ---


def _get_bool_env(var_name: str, default: bool = False) -> bool:
    """Return a boolean from an environment variable."""

    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


def _get_bool_env_with_aliases(
    var_name: str,
    *aliases: str,
    default: bool,
) -> bool:
    """Return a boolean from a canonical environment variable or its aliases."""

    for candidate in (var_name, *aliases):
        raw_value = os.environ.get(candidate)
        if raw_value is not None:
            return raw_value.lower() in {"1", "true", "yes", "on"}
    return default


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


def _parse_int_env_with_aliases(
    var_name: str,
    *aliases: str,
    default: int,
    minimum: int | None = None,
) -> int:
    """Return an integer from an environment variable or its aliases."""

    for candidate in (var_name, *aliases):
        raw_value = os.environ.get(candidate)
        if raw_value is None:
            continue

        try:
            value = int(raw_value)
        except ValueError as exc:  # pragma: no cover - defensive programming
            raise ImproperlyConfigured(f"{candidate} must be an integer if provided.") from exc

        if minimum is not None and value < minimum:
            raise ImproperlyConfigured(f"{candidate} must be >= {minimum} if provided.")

        return value

    return default


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
DEBUG = _get_bool_env("DJANGO_DEBUG", default=not TESTING)

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", DEFAULT_SECRET_KEY)
if SECRET_KEY == DEFAULT_SECRET_KEY and not DEBUG:
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY must be set to a secure value when DJANGO_DEBUG is not enabled."
    )


def _validate_fernet_key(key: str | bytes, setting_name: str) -> bytes:
    """Ensure the provided key material is a valid Fernet key."""

    key_bytes = key.encode() if isinstance(key, str) else key
    try:
        Fernet(key_bytes)
    except (
        ValueError,
        TypeError,
    ) as exc:  # pragma: no cover - defensive programming
        raise ImproperlyConfigured(
            f"{setting_name} must be a valid 32-byte base64-encoded Fernet key."
        ) from exc
    return key_bytes


def _read_local_env_value(var_name: str) -> str | None:
    """Return a value from a local ``.env`` file if present."""

    if not LOCAL_ENV_PATH.exists():
        return None

    try:
        for raw_line in LOCAL_ENV_PATH.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() != var_name:
                continue
            return value.strip().strip("\"").strip("'")
    except OSError as exc:  # pragma: no cover - defensive programming
        warnings.warn(f"Unable to read {LOCAL_ENV_PATH}: {exc}")

    return None


def _read_configured_key(var_name: str, *, allow_dotenv: bool) -> str | None:
    """Resolve a key from the environment and, optionally, a ``.env`` file."""

    value = os.environ.get(var_name)
    if value:
        return value

    if allow_dotenv:
        return _read_local_env_value(var_name)

    return None


def _load_cached_dev_key(var_name: str) -> bytes | None:
    """Load a previously generated development key from disk."""

    if not DEV_KEY_CACHE_PATH.exists():
        return None

    try:
        cache = json.loads(DEV_KEY_CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive programming
        warnings.warn(f"Ignoring invalid dev key cache file: {exc}")
        return None

    cached_value = cache.get(var_name)
    if not cached_value:
        return None

    try:
        return _validate_fernet_key(cached_value, var_name)
    except ImproperlyConfigured:
        warnings.warn(f"Ignoring invalid cached {var_name}; regenerating.")
        return None


def _persist_dev_key(var_name: str, key: bytes) -> None:
    """Persist generated development keys so they survive restarts."""

    try:
        existing = (
            json.loads(DEV_KEY_CACHE_PATH.read_text())
            if DEV_KEY_CACHE_PATH.exists()
            else {}
        )
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive programming
        existing = {}

    existing[var_name] = key.decode()

    try:
        DEV_KEY_CACHE_PATH.write_text(json.dumps(existing, indent=2))
    except OSError as exc:  # pragma: no cover - defensive programming
        warnings.warn(f"Unable to persist dev encryption key cache: {exc}")


def _deterministic_dev_key(var_name: str) -> bytes:
    """Return a stable key for DEBUG/TESTING sessions, persisting when generated."""

    cached_key = _load_cached_dev_key(var_name)
    if cached_key:
        return cached_key

    key_bytes = Fernet.generate_key()
    _persist_dev_key(var_name, key_bytes)
    return key_bytes


def _load_data_encryption_key() -> bytes:
    """Load the symmetric encryption key used for sensitive assets."""

    key = _read_configured_key("DATA_ENCRYPTION_KEY", allow_dotenv=DEBUG or TESTING)
    if key:
        return _validate_fernet_key(key, "DATA_ENCRYPTION_KEY")

    if DEBUG or TESTING:
        return _deterministic_dev_key("DATA_ENCRYPTION_KEY")

    raise ImproperlyConfigured(
        "DATA_ENCRYPTION_KEY environment variable must be set in production environments."
    )


DATA_ENCRYPTION_KEY = _load_data_encryption_key()


def _load_face_data_encryption_key() -> bytes:
    """Load the Fernet key used to encrypt cached facial encodings."""

    key = _read_configured_key("FACE_DATA_ENCRYPTION_KEY", allow_dotenv=DEBUG or TESTING)
    if key:
        return _validate_fernet_key(key, "FACE_DATA_ENCRYPTION_KEY")

    if DEBUG or TESTING:
        return _deterministic_dev_key("FACE_DATA_ENCRYPTION_KEY")

    raise ImproperlyConfigured(
        "FACE_DATA_ENCRYPTION_KEY environment variable must be set in production environments."
    )


FACE_DATA_ENCRYPTION_KEY = _load_face_data_encryption_key()

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# Celery Beat scheduled tasks configuration
CELERY_BEAT_SCHEDULE = {
    "scheduled-nightly-evaluation": {
        "task": "recognition.scheduled_tasks.run_scheduled_evaluation",
        "schedule": 86400,  # Daily at midnight (configurable via env)
        "kwargs": {"evaluation_type": "nightly"},
        "options": {"queue": "evaluation"},
    },
    "scheduled-weekly-fairness-audit": {
        "task": "recognition.scheduled_tasks.run_fairness_audit",
        "schedule": 604800,  # Weekly (7 days)
        "kwargs": {},
        "options": {"queue": "evaluation"},
    },
    "scheduled-liveness-evaluation": {
        "task": "recognition.scheduled_tasks.run_liveness_evaluation",
        "schedule": 86400,  # Daily
        "kwargs": {},
        "options": {"queue": "evaluation"},
    },
}

# Allow customization of evaluation schedules via environment variables
_NIGHTLY_EVAL_SCHEDULE = os.environ.get("CELERY_NIGHTLY_EVAL_SCHEDULE")
if _NIGHTLY_EVAL_SCHEDULE:
    try:
        CELERY_BEAT_SCHEDULE["scheduled-nightly-evaluation"]["schedule"] = int(
            _NIGHTLY_EVAL_SCHEDULE
        )
    except ValueError:
        warnings.warn(
            f"Invalid CELERY_NIGHTLY_EVAL_SCHEDULE value: {_NIGHTLY_EVAL_SCHEDULE!r}. "
            "Expected an integer (seconds). Using default.",
            stacklevel=1,
        )

_WEEKLY_FAIRNESS_SCHEDULE = os.environ.get("CELERY_WEEKLY_FAIRNESS_SCHEDULE")
if _WEEKLY_FAIRNESS_SCHEDULE:
    try:
        CELERY_BEAT_SCHEDULE["scheduled-weekly-fairness-audit"]["schedule"] = int(
            _WEEKLY_FAIRNESS_SCHEDULE
        )
    except ValueError:
        warnings.warn(
            f"Invalid CELERY_WEEKLY_FAIRNESS_SCHEDULE value: {_WEEKLY_FAIRNESS_SCHEDULE!r}. "
            "Expected an integer (seconds). Using default.",
            stacklevel=1,
        )

_LIVENESS_EVAL_SCHEDULE = os.environ.get("CELERY_LIVENESS_EVAL_SCHEDULE")
if _LIVENESS_EVAL_SCHEDULE:
    try:
        CELERY_BEAT_SCHEDULE["scheduled-liveness-evaluation"]["schedule"] = int(
            _LIVENESS_EVAL_SCHEDULE
        )
    except ValueError:
        warnings.warn(
            f"Invalid CELERY_LIVENESS_EVAL_SCHEDULE value: {_LIVENESS_EVAL_SCHEDULE!r}. "
            "Expected an integer (seconds). Using default.",
            stacklevel=1,
        )

# Feature flag to enable/disable scheduled evaluations
CELERY_BEAT_ENABLED = _get_bool_env("CELERY_BEAT_ENABLED", default=True)

LOCALHOST_ALIASES: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]")


def _resolve_allowed_hosts(
    *,
    default_allowed_hosts: Sequence[str],
    require_explicit_hosts: bool,
) -> list[str]:
    """Return the allowed host list based on deployment defaults."""

    allowed_hosts_env = os.environ.get("DJANGO_ALLOWED_HOSTS")
    if allowed_hosts_env:
        return [host.strip() for host in allowed_hosts_env.split(",") if host.strip()]

    if require_explicit_hosts:
        raise ImproperlyConfigured(
            "DJANGO_ALLOWED_HOSTS must be provided (comma separated) when secure defaults are enforced."
        )

    return list(default_allowed_hosts)


def configure_environment(
    *,
    secure_defaults: bool,
    default_allowed_hosts: Sequence[str],
    require_allowed_hosts: bool,
) -> None:
    """Populate security-sensitive settings for the active environment."""

    global ALLOWED_HOSTS
    global SECURE_SSL_REDIRECT
    global SECURE_HSTS_SECONDS
    global SECURE_HSTS_INCLUDE_SUBDOMAINS
    global SECURE_HSTS_PRELOAD
    global SESSION_COOKIE_SECURE
    global SESSION_COOKIE_HTTPONLY
    global SESSION_COOKIE_SAMESITE
    global SESSION_COOKIE_AGE
    global SESSION_EXPIRE_AT_BROWSER_CLOSE
    global CSRF_COOKIE_SECURE

    ALLOWED_HOSTS = _resolve_allowed_hosts(
        default_allowed_hosts=default_allowed_hosts,
        require_explicit_hosts=require_allowed_hosts,
    )

    SECURE_SSL_REDIRECT = _get_bool_env_with_aliases(
        "DJANGO_SECURE_SSL_REDIRECT",
        "SECURE_SSL_REDIRECT",
        default=secure_defaults,
    )
    SECURE_HSTS_SECONDS = _parse_int_env_with_aliases(
        "DJANGO_SECURE_HSTS_SECONDS",
        "SECURE_HSTS_SECONDS",
        default=3600 if secure_defaults else 0,
        minimum=0,
    )
    SECURE_HSTS_INCLUDE_SUBDOMAINS = _get_bool_env_with_aliases(
        "DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS",
        "SECURE_HSTS_INCLUDE_SUBDOMAINS",
        default=secure_defaults,
    )
    SECURE_HSTS_PRELOAD = _get_bool_env_with_aliases(
        "DJANGO_SECURE_HSTS_PRELOAD",
        "SECURE_HSTS_PRELOAD",
        default=secure_defaults,
    )

    SESSION_COOKIE_SECURE = _get_bool_env_with_aliases(
        "DJANGO_SESSION_COOKIE_SECURE",
        "SESSION_COOKIE_SECURE",
        default=secure_defaults,
    )
    SESSION_COOKIE_HTTPONLY = _get_bool_env_with_aliases(
        "DJANGO_SESSION_COOKIE_HTTPONLY",
        "SESSION_COOKIE_HTTPONLY",
        default=True,
    )
    CSRF_COOKIE_SECURE = _get_bool_env_with_aliases(
        "DJANGO_CSRF_COOKIE_SECURE",
        "CSRF_COOKIE_SECURE",
        default=secure_defaults,
    )

    session_cookie_samesite_raw = os.environ.get("DJANGO_SESSION_COOKIE_SAMESITE")
    if session_cookie_samesite_raw is None:
        session_cookie_samesite_raw = os.environ.get("SESSION_COOKIE_SAMESITE")
    if session_cookie_samesite_raw:
        normalized_samesite = session_cookie_samesite_raw.strip().lower()
        valid_samesite_values = {"lax": "Lax", "strict": "Strict", "none": "None"}
        if normalized_samesite not in valid_samesite_values:
            raise ImproperlyConfigured(
                "DJANGO_SESSION_COOKIE_SAMESITE must be one of: Lax, Strict, or None."
            )
        SESSION_COOKIE_SAMESITE = valid_samesite_values[normalized_samesite]
    else:
        SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_AGE = _parse_int_env_with_aliases(
        "DJANGO_SESSION_COOKIE_AGE",
        "SESSION_COOKIE_AGE",
        default=1800,
        minimum=1,
    )
    SESSION_EXPIRE_AT_BROWSER_CLOSE = _get_bool_env_with_aliases(
        "DJANGO_SESSION_EXPIRE_AT_BROWSER_CLOSE",
        "SESSION_EXPIRE_AT_BROWSER_CLOSE",
        default=secure_defaults,
    )

    require_database_ssl = _get_bool_env_with_aliases(
        "DATABASE_SSL_REQUIRE",
        "DB_SSL_REQUIRE",
        default=secure_defaults,
    )
    db_options = DATABASES["default"].setdefault("OPTIONS", {})
    if require_database_ssl:
        db_options["sslmode"] = os.environ.get("DATABASE_SSLMODE", "require")
    else:
        db_options.pop("sslmode", None)


# --- Application Configuration ---

INSTALLED_APPS = [
    # Custom applications for this project
    "users.apps.UsersConfig",
    "recognition.apps.RecognitionConfig",
    # Third-party packages
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

    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

if not TESTING:
    INSTALLED_APPS.append("silk")
    MIDDLEWARE.insert(0, "silk.middleware.SilkyMiddleware")

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

DATABASES = {
    "default": database_config,
}


def build_postgres_database_config() -> dict[str, Any]:
    """Return a PostgreSQL configuration derived from discrete environment variables."""

    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME", "attendance"),
        "USER": os.environ.get("DB_USER", "attendance"),
        "PASSWORD": os.environ.get("DB_PASSWORD", "attendance"),
        "HOST": os.environ.get("DB_HOST", "localhost"),
        "PORT": os.environ.get("DB_PORT", "5432"),
        "CONN_MAX_AGE": _parse_int_env("DB_CONN_MAX_AGE", 600, minimum=0),
    }


configure_environment(
    secure_defaults=not DEBUG,
    default_allowed_hosts=LOCALHOST_ALIASES,
    require_allowed_hosts=not DEBUG,
)


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
STATIC_ROOT = Path(os.environ.get("DJANGO_STATIC_ROOT", BASE_DIR / "staticfiles"))
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

# Threshold for accepting DeepFace matches when marking attendance.
# Lower values (e.g., 0.3) mean stricter matching, while higher values (e.g., 0.5)
# are more permissive. This can be overridden via an environment variable.
RECOGNITION_DISTANCE_THRESHOLD = float(os.environ.get("RECOGNITION_DISTANCE_THRESHOLD", "0.4"))

RECOGNITION_LIGHTWEIGHT_LIVENESS_ENABLED = _get_bool_env(
    "RECOGNITION_LIGHTWEIGHT_LIVENESS_ENABLED",
    default=True,
)
RECOGNITION_LIVENESS_WINDOW = _parse_int_env(
    "RECOGNITION_LIVENESS_WINDOW",
    default=5,
    minimum=2,
)
RECOGNITION_LIVENESS_MIN_FRAMES = _parse_int_env(
    "RECOGNITION_LIVENESS_MIN_FRAMES",
    default=3,
    minimum=2,
)
RECOGNITION_LIVENESS_MOTION_THRESHOLD = _get_float_env(
    "RECOGNITION_LIVENESS_MOTION_THRESHOLD",
    default=1.1,
    minimum=0.0,
)


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

DEFAULT_FACE_API_RATE_LIMIT = "5/m"
RECOGNITION_FACE_API_RATE_LIMIT = os.environ.get(
    "RECOGNITION_FACE_API_RATE_LIMIT", DEFAULT_FACE_API_RATE_LIMIT
)

_raw_api_keys = os.environ.get("RECOGNITION_API_KEYS", "")
if _raw_api_keys:
    RECOGNITION_API_KEYS = tuple(
        key.strip() for key in _raw_api_keys.split(",") if key.strip()
    )
else:
    RECOGNITION_API_KEYS: tuple[str, ...] = ()

RECOGNITION_JWT_SECRET = os.environ.get("RECOGNITION_JWT_SECRET", "")
RECOGNITION_JWT_ISSUER = os.environ.get("RECOGNITION_JWT_ISSUER")
RECOGNITION_JWT_AUDIENCE = os.environ.get("RECOGNITION_JWT_AUDIENCE")

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
