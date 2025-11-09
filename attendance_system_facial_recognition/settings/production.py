"""Production settings overriding the defaults with hardened options."""

from __future__ import annotations

import os

from . import base as base_settings
from .base import *  # noqa: F401,F403


def _get_db_setting(var_name: str, *, default: str | None = None) -> str:
    """Return a database setting from the environment or the provided default."""

    value = os.environ.get(var_name)
    if value:
        return value
    if default is None:
        raise ImproperlyConfigured(
            f"{var_name} must be set when using the production settings module."
        )
    return default


DATABASES["default"] = {
    "ENGINE": "django.db.backends.postgresql",
    "NAME": _get_db_setting("DB_NAME", default="attendance"),
    "USER": _get_db_setting("DB_USER", default="attendance"),
    "PASSWORD": _get_db_setting("DB_PASSWORD", default="attendance"),
    "HOST": _get_db_setting("DB_HOST", default="localhost"),
    "PORT": _get_db_setting("DB_PORT", default="5432"),
    "CONN_MAX_AGE": base_settings._parse_int_env("DB_CONN_MAX_AGE", 600, minimum=0),
}

if base_settings._get_bool_env("DB_SSL_REQUIRE", default=False):
    DATABASES["default"].setdefault("OPTIONS", {})["sslmode"] = "require"
