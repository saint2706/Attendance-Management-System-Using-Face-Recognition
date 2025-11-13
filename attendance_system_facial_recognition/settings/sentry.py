"""Sentry configuration helpers used by production deployments."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from django.core.exceptions import ImproperlyConfigured

import sentry_sdk
from sentry_sdk.integrations import DidNotEnable
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.redis import RedisIntegration

from . import base as base_settings

__all__ = ["initialize_sentry"]

_SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie"}


def _get_sample_rate(var_name: str, default: float) -> float:
    """Return a tracing sample rate constrained between 0.0 and 1.0 inclusive."""
    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ImproperlyConfigured(
            f"{var_name} must be a floating point number between 0.0 and 1.0."
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise ImproperlyConfigured(f"{var_name} must be between 0.0 and 1.0 when provided.")
    return value


def _scrub_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
    """Remove sensitive headers from captured events."""
    scrubbed = dict(headers)
    for header in _SENSITIVE_HEADERS:
        if header in scrubbed:
            scrubbed[header] = "[Filtered]"
    return scrubbed


def _before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any] | None:
    """Scrub sensitive data from events before they are sent to Sentry."""
    if "request" in event and isinstance(event["request"], dict):
        if "headers" in event["request"] and isinstance(event["request"]["headers"], dict):
            event["request"]["headers"] = _scrub_headers(event["request"]["headers"])

    # Do not send user data to Sentry unless explicitly enabled
    if not base_settings._get_bool_env("SENTRY_SEND_DEFAULT_PII", default=False):
        if "user" in event:
            del event["user"]

    return event


def _before_breadcrumb(crumb: dict[str, Any], hint: dict[str, Any] | None) -> dict[str, Any] | None:
    """Filter out noisy or irrelevant breadcrumbs."""
    # Example: filter out SQL queries from a specific noisy logger
    if crumb.get("category") == "django.db.backends":
        if "some_noisy_logger" in crumb.get("message", ""):
            return None
    return crumb


def initialize_sentry() -> None:
    """Initialise Sentry SDK when a DSN is supplied via the environment."""
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return

    integrations = [
        DjangoIntegration(transaction_style="url"),
        LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        RedisIntegration(),
    ]

    try:
        integrations.append(CeleryIntegration())
    except DidNotEnable:
        logging.debug("CeleryIntegration for Sentry not enabled.")

    traces_sample_rate = _get_sample_rate("SENTRY_TRACES_SAMPLE_RATE", default=0.1)
    profiles_sample_rate = _get_sample_rate("SENTRY_PROFILES_SAMPLE_RATE", default=0.1)

    sentry_sdk.init(
        dsn=dsn,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
        release=os.environ.get("SENTRY_RELEASE"),
        integrations=integrations,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
        send_default_pii=base_settings._get_bool_env("SENTRY_SEND_DEFAULT_PII", default=False),
        before_send=_before_send,
        before_breadcrumb=_before_breadcrumb,
        enable_tracing=True,
    )
