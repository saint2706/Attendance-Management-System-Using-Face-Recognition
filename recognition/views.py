"""
Views for the recognition app.

This module contains the view functions for the face recognition-based attendance system.
It handles requests for rendering pages, processing forms, capturing images,
marking attendance, and displaying attendance data.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render

import cv2
from celery.result import AsyncResult
from django_ratelimit.core import is_ratelimited

from users.models import RecognitionAttempt

from .forms import usernameForm
from .tasks import recognize_face
from .webcam_manager import get_webcam_manager

logger = logging.getLogger(__name__)


def _monotonic_seconds() -> float:
    """Return a best-effort float value from :func:`time.monotonic`."""
    try:
        return float(time.monotonic())
    except (TypeError, ValueError, AttributeError):
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
        return max(0.0, (_monotonic_seconds() - self._start_time) * 1000.0)

    def _log(
        self, *, username: str | None, success: bool, spoofed: bool, error: str = ""
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
        self, username: str | None, *, spoofed: bool, error: str
    ) -> RecognitionAttempt | None:
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
    def success_attempt_ids(self) -> dict[str, int]:
        return {username: attempt.id for username, attempt in self._successes.items()}


def _resolve_recognition_site(request: HttpRequest) -> str:
    """Best-effort lookup for the site identifier associated with a request."""
    for header in ("HTTP_X_RECOGNITION_SITE", "HTTP_X_SITE_CODE", "HTTP_X_SITE"):
        if value := request.META.get(header):
            return str(value)
    return getattr(settings, "RECOGNITION_SITE_CODE", request.get_host())


def _attach_attempt_user(attempt: RecognitionAttempt | None, username: str | None) -> None:
    """Persist a user relation on the attempt when a username is known."""
    if not attempt or not username:
        return
    if user := User.objects.filter(username=username).first():
        attempt.user = user
        attempt.save(update_fields=["user"])


def attendance_rate_limited(view_func: Callable) -> Callable:
    """Apply django-ratelimit protection to attendance endpoints."""

    def _wrapped(request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        rate = getattr(settings, "RECOGNITION_ATTENDANCE_RATE_LIMIT", "5/m")
        methods = tuple(
            m.upper()
            for m in getattr(settings, "RECOGNITION_ATTENDANCE_RATE_LIMIT_METHODS", ("POST",))
        )
        if not rate or (request.method not in methods):
            return view_func(request, *args, **kwargs)

        if is_ratelimited(
            request=request,
            group="recognition.attendance",
            key="user_or_ip",
            rate=rate,
            method=request.method,
            increment=True,
        ):
            user_identity = (
                request.user.username
                if request.user.is_authenticated
                else request.META.get("REMOTE_ADDR", "unknown")
            )
            logger.warning(
                "Attendance rate limit triggered for %s via %s", user_identity, request.method
            )
            return HttpResponse("Too many attendance attempts. Please wait.", status=429)

        return view_func(request, *args, **kwargs)

    return _wrapped


@login_required
def task_status(request: HttpRequest, task_id: str) -> JsonResponse:
    """Poll for the status of a Celery task."""
    task = AsyncResult(task_id)
    response_data = {"task_id": task_id, "status": task.status, "result": task.result}
    return JsonResponse(response_data)


def _mark_attendance(request: HttpRequest, check_in: bool) -> JsonResponse:
    """
    Core logic for marking attendance using face recognition.
    This view now captures an image and enqueues a Celery task for recognition.
    """
    manager = get_webcam_manager()
    direction = "in" if check_in else "out"

    with manager.frame_consumer() as consumer:
        frame = consumer.read(timeout=5.0)

    if frame is None:
        return JsonResponse({"error": "Could not capture an image from the webcam."}, status=500)

    _, buffer = cv2.imencode(".jpg", frame)
    image_bytes = buffer.tobytes()

    task = recognize_face.delay(list(image_bytes), direction)

    return JsonResponse({"task_id": task.id}, status=202)


@login_required
@attendance_rate_limited
def mark_your_attendance(request: HttpRequest) -> JsonResponse:
    """View to handle marking time-in."""
    return _mark_attendance(request, check_in=True)


@login_required
@attendance_rate_limited
def mark_your_attendance_out(request: HttpRequest) -> JsonResponse:
    """View to handle marking time-out."""
    return _mark_attendance(request, check_in=False)


def home(request: HttpRequest) -> HttpResponse:
    """Render the home page."""
    return render(request, "recognition/home.html")


def dashboard(request: HttpRequest) -> HttpResponse:
    """Render the dashboard, which differs for admins and regular employees."""
    if request.user.is_staff or request.user.is_superuser:
        return render(request, "recognition/admin_dashboard.html")
    return render(request, "recognition/employee_dashboard.html")


@login_required
def add_photos(request: HttpRequest) -> HttpResponse:
    """Handle the 'Add Photos' functionality for admins to create face datasets for users."""
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    if request.method == "POST":
        form = usernameForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            # The create_dataset function has been removed, as it's now handled by the Celery task
            messages.success(
                request, f"Dataset creation for {username} is now handled asynchronously."
            )
            return redirect("add-photos")
    else:
        form = usernameForm()

    return render(request, "recognition/add_photos.html", {"form": form})


@login_required
def train(request: HttpRequest) -> HttpResponse:
    """
    This view is now obsolete, as DeepFace handles recognition implicitly
    by searching through image folders. It redirects to the dashboard with an
    informational message.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    messages.info(
        request,
        "The training process is now automatic. Just add photos for new users.",
    )
    return redirect("dashboard")
