"""Production settings overriding the defaults with hardened options."""

from __future__ import annotations

from django.core.exceptions import ImproperlyConfigured

from .base import *  # noqa: F401,F403
from .base import DATABASES, build_postgres_database_config, configure_environment
from .sentry import initialize_sentry

DEBUG = False


if DATABASES["default"].get("ENGINE") == "django.db.backends.sqlite3":
    DATABASES["default"] = build_postgres_database_config()

if DATABASES["default"].get("ENGINE") == "django.db.backends.sqlite3":
    raise ImproperlyConfigured(
        "Production deployments must configure a PostgreSQL database via DATABASE_URL or DB_* environment variables."
    )


configure_environment(
    secure_defaults=True,
    default_allowed_hosts=(),
    require_allowed_hosts=True,
)


initialize_sentry()
