"""
Shared utilities and helpers for recognition views.

This module contains utilities used across different recognition view modules,
including Sentry integration, recognition attempt logging, rate limiting,
and async task management.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Dict, Mapping, Optional, Sequence

from django.conf import settings
from django.contrib.auth.models import User
from django.http import HttpResponse

from celery.result import AsyncResult
from django_ratelimit.core import is_ratelimited
from sentry_sdk import Hub

from users.models import RecognitionAttempt

# Initialize logger for the module
logger = logging.getLogger(__name__)


def _record_sentry_breadcrumb(
    *,
    message: str,
    category: str,
    level: str = "info",
    data: Mapping[str, object] | None = None,
) -> None:
    """Add a breadcrumb to the active Sentry hub, swallowing integration errors."""

    try:
        Hub.current.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=dict(data or {}),
        )
    except Exception:  # pragma: no cover - telemetry is best-effort
        logger.debug("Unable to add Sentry breadcrumb", exc_info=True)


def _bind_request_to_sentry_scope(
    request,
    *,
    flow: str,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Attach request metadata to the Sentry scope for easier triage."""

    try:
        with Hub.current.configure_scope() as scope:
            context: dict[str, object] = {
                "path": getattr(request, "path", ""),
                "method": getattr(request, "method", ""),
                "flow": flow,
            }
            if extra:
                context.update(extra)
            scope.set_context("attendance_flow", context)

            user = getattr(request, "user", None)
            if getattr(user, "is_authenticated", False):
                username: str | None = None
                get_username = getattr(user, "get_username", None)
                if callable(get_username):
                    try:
                        username = get_username()
                    except Exception:  # pragma: no cover - user object may be lazy
                        username = None
                if not username:
                    username = getattr(user, "username", None)

                scope.set_user(
                    {
                        "id": getattr(user, "id", None),
                        "username": username,
                        "email": getattr(user, "email", None) or None,
                    }
                )

            direction = context.get("direction")
            if direction:
                scope.set_tag("attendance.direction", str(direction))
    except Exception:  # pragma: no cover - telemetry must not break request handling
        logger.debug("Unable to bind request metadata to Sentry scope", exc_info=True)


def _monotonic_seconds() -> float:
    """Return a best-effort float value from :func:`time.monotonic`."""

    try:
        value = time.monotonic()
    except Exception:  # pragma: no cover - defensive guard around patched time
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class _RecognitionAttemptLogger:
    """Utility for creating recognition attempt records with shared metadata."""

    __slots__ = (
        "_direction",
        "_site",
        "_source",
        "_start_time",
        "_successes",
        "_recorded_keys",
        "records",
    )

    def __init__(self, direction: str, site: str, source: str) -> None:
        self._direction = direction
        self._site = site
        self._source = source
        self._start_time = _monotonic_seconds()
        self._successes: dict[str, RecognitionAttempt] = {}
        self._recorded_keys: set[str] = set()
        self.records: list[RecognitionAttempt] = []

    def _latency_ms(self) -> float:
        now = _monotonic_seconds()
        try:
            start = float(self._start_time)
        except (TypeError, ValueError):
            start = 0.0
            self._start_time = start
        return max(0.0, (now - start) * 1000.0)

    def _log(
        self, *, username: Optional[str], success: bool, spoofed: bool, error: str = ""
    ) -> RecognitionAttempt:
        attempt = RecognitionAttempt(
            username=username or "",
            direction=self._direction,
            site=self._site,
            source=self._source,
            successful=success,
            spoof_detected=spoofed,
            latency_ms=self._latency_ms(),
            error_message=error,
        )
        attempt.save()
        self.records.append(attempt)
        return attempt

    def log_success(self, username: str) -> RecognitionAttempt:
        if not username:
            raise ValueError("Username must be provided for successful attempts.")
        if username in self._successes:
            return self._successes[username]
        attempt = self._log(username=username, success=True, spoofed=False)
        self._successes[username] = attempt
        self._recorded_keys.add(username)
        return attempt

    def log_failure(
        self,
        username: Optional[str],
        *,
        spoofed: bool,
        error: str,
    ) -> Optional[RecognitionAttempt]:
        key = username or "__generic__"
        if key in self._recorded_keys:
            return None
        attempt = self._log(username=username, success=False, spoofed=spoofed, error=error)
        self._recorded_keys.add(key)
        return attempt

    def ensure_generic_failure(self, message: str) -> None:
        if not self.records:
            self.log_failure(None, spoofed=False, error=message)

    @property
    def success_attempt_ids(self) -> Dict[str, int]:
        return {username: attempt.id for username, attempt in self._successes.items()}


def _resolve_recognition_site(request) -> str:
    """Best-effort lookup for the site identifier associated with a request."""

    header_keys = ("HTTP_X_RECOGNITION_SITE", "HTTP_X_SITE_CODE", "HTTP_X_SITE")
    for header in header_keys:
        value = request.META.get(header)
        if value:
            return str(value)

    site_setting = getattr(settings, "RECOGNITION_SITE_CODE", "")
    if site_setting:
        return str(site_setting)

    try:
        return request.get_host()
    except Exception:  # pragma: no cover - request may not have host in tests
        return ""


def _attach_attempt_user(attempt: RecognitionAttempt | None, username: Optional[str]) -> None:
    """Persist a user relation on the attempt when a username is known."""

    if not attempt or not username:
        return

    user = User.objects.filter(username=username).first()
    if not user:
        return

    attempt.user = user
    attempt.save(update_fields=["user"])


# Rate limit helper utilities -------------------------------------------------


def _attendance_rate_limit_methods() -> tuple[str, ...]:
    """Return the normalized HTTP methods subject to attendance rate limiting."""

    methods = getattr(settings, "RECOGNITION_ATTENDANCE_RATE_LIMIT_METHODS", ("POST",))
    if isinstance(methods, str):
        methods = (methods,)
    return tuple(method.upper() for method in methods)


def attendance_rate_limited(view_func):
    """Apply django-ratelimit protection to attendance endpoints."""

    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        rate = getattr(settings, "RECOGNITION_ATTENDANCE_RATE_LIMIT", "5/m")
        if not rate:
            return view_func(request, *args, **kwargs)

        methods = _attendance_rate_limit_methods()
        request_method = request.method.upper()
        if methods and request_method not in methods:
            return view_func(request, *args, **kwargs)

        was_limited = is_ratelimited(
            request=request,
            group="recognition.attendance",
            key="user_or_ip",
            rate=rate,
            method=methods or None,
            increment=True,
        )

        if was_limited:
            logger.warning(
                "Attendance rate limit triggered for %s via %s",
                (
                    request.user
                    if getattr(request, "user", None) and request.user.is_authenticated
                    else request.META.get("REMOTE_ADDR", "unknown")
                ),
                request_method,
            )
            return HttpResponse("Too many attendance attempts. Please wait.", status=429)

        return view_func(request, *args, **kwargs)

    return _wrapped


def _enqueue_attendance_records(records: Sequence[Dict[str, Any]]) -> AsyncResult:
    """Submit attendance records for asynchronous processing via Celery."""

    from recognition.tasks import process_attendance_batch

    return process_attendance_batch.delay(records)


def _describe_async_result(task_id: str) -> Dict[str, Any]:
    """Return a normalized snapshot of the Celery task identified by ``task_id``."""

    result = AsyncResult(task_id)
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "successful": result.successful(),
    }

    info = result.info
    if isinstance(info, Mapping):
        payload["meta"] = dict(info)
    elif info is not None and not isinstance(info, Exception):
        payload["meta"] = info

    if result.failed():
        try:
            payload["error"] = str(result.result)
        except Exception:  # pragma: no cover - defensive programming
            payload["error"] = "Task failed"
    elif result.successful():
        try:
            payload["result"] = result.result
        except Exception:  # pragma: no cover - defensive programming
            payload["result"] = info if info is not None else {}

    return payload


def username_present(username: str) -> bool:
    """
    Return whether the given username exists in the system.

    Args:
        username: The username to check.

    Returns:
        True if the user exists, False otherwise.
    """
    return User.objects.filter(username=username).exists()


def not_authorised(request):
    """Render the 'not authorised' page for users without permissions."""
    from django.shortcuts import render

    return render(request, "recognition/not_authorised.html")
