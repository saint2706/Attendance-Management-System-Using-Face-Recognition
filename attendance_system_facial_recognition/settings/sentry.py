"""Sentry configuration helpers used by production deployments."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from django.core.exceptions import ImproperlyConfigured

import sentry_sdk
from sentry_sdk.integrations import DidNotEnable
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

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
    except ValueError as exc:  # pragma: no cover - defensive
        raise ImproperlyConfigured(
            f"{var_name} must be a floating point number between 0.0 and 1.0."
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise ImproperlyConfigured(
            f"{var_name} must be between 0.0 and 1.0 when provided."
        )
    return value


def _scrub_headers(headers: Mapping[str, Any]) -> None:
    """Remove sensitive headers from captured events in-place."""

    for header in _SENSITIVE_HEADERS:
        if header in headers:
            headers[header] = "[Filtered]"


def initialize_sentry() -> None:
    """Initialise Sentry SDK when a DSN is supplied via the environment."""

    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return

    send_default_pii = base_settings._get_bool_env(
        "SENTRY_SEND_DEFAULT_PII", default=False
    )

    def _before_send(
        event: dict[str, Any], _hint: object | None
    ) -> dict[str, Any] | None:
        request = event.get("request")
        if isinstance(request, dict):
            headers = request.get("headers")
            if isinstance(headers, Mapping):
                _scrub_headers(headers)
        if not send_default_pii:
            event.pop("user", None)
        return event

    integrations = [
        DjangoIntegration(transaction_style="url"),
        LoggingIntegration(level=None, event_level=logging.ERROR),
    ]

    try:
        from sentry_sdk.integrations.celery import CeleryIntegration
    except (ImportError, DidNotEnable):  # pragma: no cover - optional dependency
        CeleryIntegration = None
    if CeleryIntegration is not None:
        try:
            integrations.append(CeleryIntegration())
        except DidNotEnable:  # pragma: no cover - optional dependency
            pass

    try:
        from sentry_sdk.integrations.redis import RedisIntegration
    except ImportError:  # pragma: no cover - optional dependency
        RedisIntegration = None
    if RedisIntegration is not None:
        integrations.append(RedisIntegration())

    traces_sample_rate = _get_sample_rate("SENTRY_TRACES_SAMPLE_RATE", default=0.0)
    profiles_sample_rate = _get_sample_rate("SENTRY_PROFILES_SAMPLE_RATE", default=0.0)

    sentry_sdk.init(
        dsn=dsn,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
        release=os.environ.get("SENTRY_RELEASE"),
        integrations=integrations,
        enable_tracing=True,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
        send_default_pii=send_default_pii,
        before_send=_before_send,
    )
