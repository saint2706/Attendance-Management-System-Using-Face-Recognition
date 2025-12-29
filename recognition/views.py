"""
Views for the recognition app.

This module contains the view functions for the face recognition-based attendance system.
It handles requests for rendering pages, processing forms, capturing images,
marking attendance, and displaying attendance data.
"""

from __future__ import annotations

import base64
import binascii
import datetime
import hashlib
import io
import json
import logging
import math
import os
import pickle
import sys
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urljoin

from django.conf import settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.cache import cache
from django.db.models import Count, QuerySet
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View

import cv2
import imutils
import jwt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from celery.result import AsyncResult
from deepface import DeepFace
from django_pandas.io import read_frame
from django_ratelimit.core import is_ratelimited
from django_ratelimit.decorators import ratelimit
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
from sentry_sdk import Hub

from src.common import FaceDataEncryption, InvalidToken, decrypt_bytes
from users.models import Direction, Present, RecognitionAttempt, Time

from . import health, monitoring
from .forms import DateForm, DateForm_2, UsernameAndDateForm, usernameForm
from .liveness import LivenessBuffer, is_live_face
from .metrics_store import log_recognition_outcome
from .models import RecognitionOutcome
from .pipeline import extract_embedding, find_closest_dataset_match, is_within_distance_threshold
from .webcam_manager import get_webcam_manager

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


def _face_api_rate_limit_key(group, request):  # pragma: no cover - wrapper delegates logic
    """Return a stable rate-limit key for the face recognition API."""

    user = getattr(request, "user", None)
    if getattr(user, "is_authenticated", False):
        return f"user:{getattr(user, 'pk', '') or getattr(user, 'username', '')}"

    api_key = request.META.get("HTTP_X_API_KEY")
    if api_key:
        return f"api-key:{hashlib.sha256(api_key.encode()).hexdigest()}"

    auth_header = request.META.get("HTTP_AUTHORIZATION", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.partition(" ")[2].strip()
        if token:
            return f"bearer:{hashlib.sha256(token.encode()).hexdigest()}"

    return request.META.get("REMOTE_ADDR")


def _enqueue_attendance_records(records: Sequence[Dict[str, Any]]) -> AsyncResult:
    """Submit attendance records for asynchronous processing via Celery."""

    from .tasks import process_attendance_batch

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


@method_decorator(
    ratelimit(
        key=_face_api_rate_limit_key,
        rate=getattr(settings, "RECOGNITION_FACE_API_RATE_LIMIT", "5/m"),
        method="POST",
        block=False,
    ),
    name="post",
)
class FaceRecognitionAPI(View):
    """Handle face recognition requests submitted via HTTP POST."""

    http_method_names = ["post", "options"]

    def _authenticate_request(self, request) -> tuple[bool, Optional[str], Optional[str]]:
        """Validate session, API key, or JWT credentials for API access."""

        user = getattr(request, "user", None)
        if getattr(user, "is_authenticated", False):
            request.face_api_principal = getattr(user, "username", None) or str(
                getattr(user, "pk", "")
            )
            return True, request.face_api_principal, None

        api_keys: tuple[str, ...] = tuple(
            key.strip() for key in getattr(settings, "RECOGNITION_API_KEYS", ()) if key.strip()
        )
        api_key = request.META.get("HTTP_X_API_KEY")
        if api_key:
            if api_keys and api_key in api_keys:
                masked_key = hashlib.sha256(api_key.encode()).hexdigest()
                request.face_api_principal = f"api-key:{masked_key}"
                return True, request.face_api_principal, None
            return False, None, "Invalid API key provided."

        auth_header = request.META.get("HTTP_AUTHORIZATION", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.partition(" ")[2].strip()
            if not token:
                return False, None, "Authorization header is malformed."

            secret = getattr(settings, "RECOGNITION_JWT_SECRET", "")
            if not secret:
                return False, None, "JWT authentication is not configured."

            audience = getattr(settings, "RECOGNITION_JWT_AUDIENCE", None) or None
            issuer = getattr(settings, "RECOGNITION_JWT_ISSUER", None) or None
            options = {"verify_aud": audience is not None}

            try:
                claims = jwt.decode(
                    token,
                    secret,
                    algorithms=["HS256"],
                    audience=audience,
                    issuer=issuer,
                    options=options,
                )
            except jwt.ExpiredSignatureError:
                return False, None, "Authentication token has expired."
            except jwt.InvalidTokenError:
                return False, None, "Invalid authentication token supplied."

            subject = claims.get("sub") or claims.get("username")
            request.face_api_principal = subject or "jwt-client"
            request.face_api_claims = claims
            return True, request.face_api_principal, None

        return False, None, "Authentication is required for this endpoint."

    def _parse_payload(self, request) -> Dict[str, Any]:
        """Return the JSON or form payload supplied with the request."""

        content_type = request.META.get("CONTENT_TYPE", "")
        if "application/json" in content_type:
            try:
                raw_body = request.body.decode(request.encoding or "utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("Request body must be UTF-8 encoded JSON.") from exc

            try:
                return json.loads(raw_body or "{}")
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive programming
                raise ValueError("Invalid JSON payload supplied.") from exc

        if request.POST:
            return request.POST.dict()

        if not request.body:
            return {}

        try:
            raw_body = request.body.decode(request.encoding or "utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - defensive programming
            raise ValueError("Unable to decode request body.") from exc

        try:
            return json.loads(raw_body or "{}")
        except json.JSONDecodeError:
            return {}

    def _coerce_embedding(self, raw_embedding) -> Optional[np.ndarray]:
        """Convert the submitted embedding payload into a NumPy array."""

        if raw_embedding is None:
            return None

        if isinstance(raw_embedding, str):
            stripped = raw_embedding.strip()
            if not stripped:
                return None
            try:
                raw_embedding = json.loads(stripped)
            except json.JSONDecodeError:
                parts = [segment for segment in stripped.split(",") if segment.strip()]
                try:
                    raw_embedding = [float(segment) for segment in parts]
                except ValueError as exc:
                    raise ValueError("'embedding' must contain numeric values.") from exc

        if not isinstance(raw_embedding, (list, tuple)):
            raise ValueError("'embedding' must be provided as a list of numbers.")

        try:
            vector = np.array([float(value) for value in raw_embedding], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("'embedding' must contain only numeric values.") from exc

        if vector.size == 0:
            raise ValueError("'embedding' must contain at least one value.")

        return vector

    def _extract_image_bytes(self, request, payload: Dict[str, Any]) -> Optional[bytes]:
        """Return raw image bytes from either an upload or base64 string."""
        # 🛡️ Sentinel: Enforce max upload size to prevent Memory Exhaustion DoS
        max_size = int(getattr(settings, "RECOGNITION_MAX_UPLOAD_SIZE", 5 * 1024 * 1024))  # 5MB

        uploaded = request.FILES.get("image") if hasattr(request, "FILES") else None
        if uploaded is not None:
            if uploaded.size > max_size:
                raise ValueError(
                    f"Image size {uploaded.size} bytes exceeds maximum allowed size of {max_size} bytes."
                )
            return uploaded.read()

        raw_image = payload.get("image")
        if not raw_image:
            return None

        if isinstance(raw_image, (bytes, bytearray)):
            if len(raw_image) > max_size:
                raise ValueError("Image payload exceeds maximum allowed size.")
            return bytes(raw_image)

        if not isinstance(raw_image, str):
            raise ValueError("Unsupported image payload supplied.")

        image_data = raw_image.strip()
        if not image_data:
            return None

        if image_data.startswith("data:"):
            _, _, image_data = image_data.partition(",")

        # Check base64 string length approximation (base64 is ~4/3 larger)
        # Using 1.4 multiplier for base64 overhead
        if len(image_data) > max_size * 1.4:
            raise ValueError("Image payload exceeds maximum allowed size.")

        try:
            decoded = base64.b64decode(image_data, validate=True)
            if len(decoded) > max_size:
                raise ValueError(
                    f"Decoded image size {len(decoded)} bytes exceeds maximum allowed size of {max_size} bytes."
                )
            return decoded
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64-encoded image payload supplied.") from exc

    def _extract_liveness_frames(self, payload: Dict[str, Any]) -> list[np.ndarray]:
        """Decode optional liveness frame bursts embedded in the payload."""

        raw_frames = payload.get("liveness_frames")
        if raw_frames in (None, ""):
            return []

        if not isinstance(raw_frames, list):
            raise ValueError("'liveness_frames' must be provided as a list of images.")

        decoded_frames: list[np.ndarray] = []
        for index, raw_frame in enumerate(raw_frames):
            if raw_frame in (None, ""):
                continue

            if isinstance(raw_frame, (bytes, bytearray)):
                frame_bytes = bytes(raw_frame)
            elif isinstance(raw_frame, str):
                frame_data = raw_frame.strip()
                if not frame_data:
                    continue
                if frame_data.startswith("data:"):
                    _, _, frame_data = frame_data.partition(",")
                try:
                    frame_bytes = base64.b64decode(frame_data, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError(f"'liveness_frames[{index}]' must be base64-encoded.") from exc
            else:
                raise ValueError(
                    "Each entry in 'liveness_frames' must be base64 data or raw bytes."
                )

            frame = _decode_image_bytes(frame_bytes)
            if frame is not None:
                decoded_frames.append(frame)

        return decoded_frames

    def post(self, request, *args, **kwargs):  # pylint: disable=unused-argument
        site_code = _resolve_recognition_site(request)
        default_direction = RecognitionAttempt.Direction.IN.value
        request_user = getattr(request, "user", None)
        request_username = getattr(request_user, "username", None)

        auth_ok, principal, auth_error = self._authenticate_request(request)
        if principal and not request_username:
            request_username = principal

        if not auth_ok:
            attempt_logger = _RecognitionAttemptLogger(
                default_direction,
                site_code,
                source="api",
            )
            attempt_logger.log_failure(
                request_username,
                spoofed=False,
                error=auth_error or "Authentication failed.",
            )
            return JsonResponse({"error": auth_error or "Authentication failed."}, status=401)

        if getattr(request, "limited", False):
            attempt_logger = _RecognitionAttemptLogger(
                default_direction,
                site_code,
                source="api",
            )
            attempt_logger.log_failure(
                request_username,
                spoofed=False,
                error="Too many requests.",
            )
            return JsonResponse({"error": "Too many requests."}, status=429)

        try:
            payload = self._parse_payload(request)
        except ValueError as exc:
            attempt_logger = _RecognitionAttemptLogger(
                default_direction,
                site_code,
                source="api",
            )
            attempt_logger.log_failure(
                request_username,
                spoofed=False,
                error=str(exc),
            )
            return JsonResponse({"error": str(exc)}, status=400)

        raw_direction = payload.get("direction")
        if isinstance(raw_direction, str):
            direction = raw_direction.lower()
        else:
            direction = default_direction

        if direction not in {
            RecognitionAttempt.Direction.IN.value,
            RecognitionAttempt.Direction.OUT.value,
        }:
            direction = default_direction

        attempt_logger = _RecognitionAttemptLogger(direction, site_code, source="api")

        model_name = _get_face_recognition_model()
        detector_backend = _get_face_detection_backend()
        enforce_detection = _should_enforce_detection()

        raw_username = payload.get("username")
        submitted_username = (
            raw_username.strip() if isinstance(raw_username, str) else request_username
        )

        embedding_vector = None
        frame = None
        facial_area: Optional[Dict[str, int]] = None
        liveness_frames: list[np.ndarray] = []

        try:
            liveness_frames = self._extract_liveness_frames(payload)
        except ValueError as exc:
            attempt_logger.log_failure(submitted_username, spoofed=False, error=str(exc))
            return JsonResponse({"error": str(exc)}, status=400)

        try:
            embedding_vector = self._coerce_embedding(payload.get("embedding"))
        except ValueError as exc:
            attempt_logger.log_failure(submitted_username, spoofed=False, error=str(exc))
            return JsonResponse({"error": str(exc)}, status=400)

        if embedding_vector is None:
            try:
                image_bytes = self._extract_image_bytes(request, payload)
            except ValueError as exc:
                attempt_logger.log_failure(submitted_username, spoofed=False, error=str(exc))
                return JsonResponse({"error": str(exc)}, status=400)

            if not image_bytes:
                attempt_logger.log_failure(
                    submitted_username,
                    spoofed=False,
                    error="Provide either an 'embedding' array or an 'image' payload.",
                )
                return JsonResponse(
                    {"error": "Provide either an 'embedding' array or an 'image' payload."},
                    status=400,
                )

            frame = _decode_image_bytes(image_bytes)
            if frame is None:
                attempt_logger.log_failure(
                    submitted_username,
                    spoofed=False,
                    error="Unable to decode the supplied image.",
                )
                return JsonResponse({"error": "Unable to decode the supplied image."}, status=400)

            try:
                representations = DeepFace.represent(
                    img_path=frame,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=enforce_detection,
                )
            except ValueError as exc:
                # DeepFace raises ValueError when no face is detected or invalid input provided
                logger.warning("DeepFace failed to detect face in API request: %s", exc)
                attempt_logger.log_failure(
                    submitted_username,
                    spoofed=False,
                    error="No face detected in the provided image.",
                )
                return JsonResponse(
                    {"error": "No face detected in the provided image."},
                    status=400,
                )
            except AttributeError as exc:
                # DeepFace raises AttributeError for library/dependency issues
                logger.error("DeepFace dependency or module error: %s", exc)
                attempt_logger.log_failure(
                    submitted_username,
                    spoofed=False,
                    error="Face recognition service misconfiguration.",
                )
                return JsonResponse(
                    {"error": "Face recognition service misconfiguration."},
                    status=500,
                )
            except OSError as exc:
                # DeepFace raises OSError for file system or model loading issues
                logger.error("DeepFace file system or model loading error: %s", exc)
                attempt_logger.log_failure(
                    submitted_username,
                    spoofed=False,
                    error="Failed to load face recognition models.",
                )
                return JsonResponse(
                    {"error": "Failed to load face recognition models."},
                    status=500,
                )
            except Exception as exc:  # pragma: no cover - catch truly unexpected errors
                # Fallback for any other unexpected exceptions from DeepFace
                logger.error("Unexpected error during face analysis: %s", exc, exc_info=True)
                attempt_logger.log_failure(
                    submitted_username,
                    spoofed=False,
                    error="Failed to analyse the provided image.",
                )
                return JsonResponse(
                    {"error": "Failed to analyse the provided image."},
                    status=500,
                )

            extracted_embedding, facial_area = extract_embedding(representations)
            if extracted_embedding is None:
                attempt_logger.log_failure(
                    submitted_username,
                    spoofed=False,
                    error="No face embedding could be extracted from the image.",
                )
                return JsonResponse(
                    {"error": "No face embedding could be extracted from the image."},
                    status=400,
                )

            try:
                embedding_vector = np.array(extracted_embedding, dtype=float)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.warning("Received invalid embedding from DeepFace: %s", exc)
                attempt_logger.log_failure(
                    submitted_username,
                    spoofed=False,
                    error="Invalid embedding generated.",
                )
                return JsonResponse({"error": "Invalid embedding generated."}, status=500)

        dataset_index = _load_dataset_embeddings_for_matching(
            model_name, detector_backend, enforce_detection
        )
        normalized_index = []
        for entry in dataset_index:
            candidate = entry.get("embedding") if isinstance(entry, dict) else None
            if candidate is None:
                continue
            if not isinstance(candidate, np.ndarray):
                try:
                    candidate_array = np.array(candidate, dtype=float)
                except Exception:  # pragma: no cover - defensive conversion
                    continue
            else:
                candidate_array = candidate
            normalized_entry = dict(entry)
            normalized_entry["embedding"] = candidate_array
            normalized_index.append(normalized_entry)

        if not normalized_index:
            attempt_logger.log_failure(
                submitted_username,
                spoofed=False,
                error="No enrolled face embeddings are available for comparison.",
            )
            return JsonResponse(
                {"error": "No enrolled face embeddings are available for comparison."},
                status=503,
            )

        distance_metric = _get_deepface_distance_metric()
        match = find_closest_dataset_match(embedding_vector, normalized_index, distance_metric)

        distance_threshold = getattr(
            settings, "RECOGNITION_DISTANCE_THRESHOLD", DEFAULT_DISTANCE_THRESHOLD
        )

        normalized_region = _normalize_face_region(facial_area)
        spoofed = False
        reference_frame = frame or (liveness_frames[-1] if liveness_frames else None)
        if reference_frame is not None:
            history = list(liveness_frames)
            if frame is not None:
                history.append(frame)
            elif not history:
                history = [reference_frame]
            spoofed = not _passes_liveness_check(
                reference_frame,
                normalized_region,
                frame_history=history,
            )

        response_payload: Dict[str, Any] = {
            "recognized": False,
            "threshold": float(distance_threshold),
            "distance_metric": distance_metric,
        }

        if match is None:
            if spoofed:
                response_payload["spoofed"] = True
            attempt = attempt_logger.log_failure(
                submitted_username,
                spoofed=spoofed,
                error="No matching identity found for the provided embedding.",
            )
            _attach_attempt_user(attempt, submitted_username)
            return JsonResponse(response_payload)

        username, distance_value, identity_path = match
        response_payload.update(
            {
                "distance": float(distance_value),
                "identity": identity_path,
            }
        )
        if username:
            response_payload["username"] = username

        resolved_username = username or submitted_username
        if not resolved_username and identity_path:
            resolved_username = Path(identity_path).stem

        if spoofed:
            response_payload["spoofed"] = True
            attempt = attempt_logger.log_failure(
                resolved_username,
                spoofed=True,
                error="Liveness check failed.",
            )
            _attach_attempt_user(attempt, resolved_username)
            return JsonResponse(response_payload)
        else:
            recognized = is_within_distance_threshold(distance_value, distance_threshold)
            response_payload["recognized"] = recognized

            if recognized:
                attempt = attempt_logger.log_success(resolved_username or "unknown")
                _attach_attempt_user(attempt, resolved_username)
            else:
                formatted_distance = (
                    float(distance_value) if distance_value is not None else float("nan")
                )
                attempt = attempt_logger.log_failure(
                    resolved_username,
                    spoofed=False,
                    error=(
                        "Distance %.4f exceeded threshold %.4f"
                        % (formatted_distance, float(distance_threshold))
                    ),
                )
                _attach_attempt_user(attempt, resolved_username)

        return JsonResponse(response_payload)


@login_required
@attendance_rate_limited
def enqueue_attendance_batch(request):
    """Accept a batch of attendance records and enqueue them for Celery processing."""

    if request.method.upper() != "POST":
        return JsonResponse({"detail": "Method not allowed."}, status=405)

    try:
        raw_body = request.body.decode(request.encoding or "utf-8") if request.body else "{}"
    except UnicodeDecodeError:
        return JsonResponse({"detail": "Request body must be UTF-8 encoded."}, status=400)

    try:
        payload = json.loads(raw_body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON payload."}, status=400)

    records = payload.get("records")
    if not isinstance(records, list):
        return JsonResponse({"detail": "'records' must be a list."}, status=400)

    normalized_records: list[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return JsonResponse(
                {"detail": f"Record at index {index} must be a JSON object."},
                status=400,
            )
        normalized_records.append(record)

    try:
        async_result = _enqueue_attendance_records(normalized_records)
    except Exception:  # pragma: no cover - defensive programming
        logger.exception("Failed to enqueue attendance batch via API.")
        return JsonResponse(
            {"detail": "Unable to enqueue attendance batch at this time."}, status=503
        )

    return JsonResponse(
        {
            "task_id": async_result.id,
            "status": async_result.status,
            "total": len(normalized_records),
        },
        status=202,
    )


# Define root directories for data storage
DATA_ROOT = Path(settings.BASE_DIR) / "face_recognition_data"
TRAINING_DATASET_ROOT = DATA_ROOT / "training_dataset"

# Define paths for saving generated attendance graphs within MEDIA_ROOT
ATTENDANCE_GRAPHS_ROOT = Path(
    getattr(
        settings,
        "ATTENDANCE_GRAPHS_ROOT",
        Path(settings.MEDIA_ROOT) / "attendance_graphs",
    )
)
HOURS_VS_DATE_PATH = ATTENDANCE_GRAPHS_ROOT / "hours_vs_date" / "1.png"
EMPLOYEE_LOGIN_PATH = ATTENDANCE_GRAPHS_ROOT / "employee_login" / "1.png"
HOURS_VS_EMPLOYEE_PATH = ATTENDANCE_GRAPHS_ROOT / "hours_vs_employee" / "1.png"
THIS_WEEK_PATH = ATTENDANCE_GRAPHS_ROOT / "this_week" / "1.png"
LAST_WEEK_PATH = ATTENDANCE_GRAPHS_ROOT / "last_week" / "1.png"


class DatasetEmbeddingCache:
    """Cache DeepFace embeddings for the encrypted training dataset."""

    def __init__(
        self,
        dataset_root: Path,
        cache_root: Path,
        *,
        encryption: FaceDataEncryption | None = None,
    ) -> None:
        self._dataset_root = dataset_root
        self._cache_root = cache_root
        self._lock = threading.RLock()
        self._memory_cache: Dict[
            Tuple[str, str, bool], Tuple[Tuple[Tuple[str, int, int], ...], list[dict]]
        ]
        self._memory_cache = {}
        self._encryption = encryption or FaceDataEncryption()

    def _cache_file_path(
        self, model_name: str, detector_backend: str, enforce_detection: bool
    ) -> Path:
        safe_model = model_name.replace("/", "_").replace(" ", "_")
        safe_detector = detector_backend.replace("/", "_").replace(" ", "_")
        enforcement_suffix = "enf" if enforce_detection else "noenf"
        filename = (
            f"representations_{safe_model.lower()}_{safe_detector.lower()}_"
            f"{enforcement_suffix}.pkl"
        )
        return self._cache_root / filename

    def _current_dataset_state(self) -> Tuple[Tuple[str, int, int], ...]:
        """Return the current state of the dataset, cached for performance."""
        cache_key = "recognition:dataset_state"
        cached_state = cache.get(cache_key)
        if cached_state is not None:
            return cached_state

        state = self._compute_dataset_state()
        # Cache for configured timeout to reduce IO on high-traffic endpoints
        timeout = getattr(settings, "RECOGNITION_DATASET_STATE_CACHE_TIMEOUT", 60)
        cache.set(cache_key, state, timeout=timeout)
        return state

    def _compute_dataset_state(self) -> Tuple[Tuple[str, int, int], ...]:
        """Compute the dataset state by scanning the filesystem."""
        entries: list[Tuple[str, int, int]] = []
        if not self._dataset_root.exists():
            return tuple()

        for image_path in sorted(self._dataset_root.glob("*/*.jpg")):
            try:
                stat_result = image_path.stat()
            except FileNotFoundError:
                continue
            relative = image_path.relative_to(self._dataset_root).as_posix()
            entries.append((relative, int(stat_result.st_mtime_ns), int(stat_result.st_size)))
        return tuple(entries)

    def _load_from_disk(
        self, model_name: str, detector_backend: str, enforce_detection: bool
    ) -> Optional[Tuple[Tuple[Tuple[str, int, int], ...], list[dict]]]:
        cache_file = self._cache_file_path(model_name, detector_backend, enforce_detection)
        if not cache_file.exists():
            return None

        try:
            encrypted_bytes = cache_file.read_bytes()
        except OSError as exc:  # pragma: no cover - defensive file I/O
            logger.warning("Failed to read cached embeddings %s: %s", cache_file, exc)
            return None

        try:
            decrypted_bytes = self._encryption.decrypt(encrypted_bytes)
        except InvalidToken:
            logger.warning("Invalid cache encryption token for %s", cache_file)
            return None
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.warning("Failed to decrypt cached embeddings %s: %s", cache_file, exc)
            return None

        try:
            payload = pickle.loads(decrypted_bytes)
        except pickle.UnpicklingError as exc:  # pragma: no cover - corrupted cache data
            logger.warning("Failed to deserialize cached embeddings %s: %s", cache_file, exc)
            return None
        except (
            EOFError,
            AttributeError,
            ImportError,
        ) as exc:  # pragma: no cover - pickle edge cases
            logger.warning("Cache deserialization error %s: %s", cache_file, exc)
            return None

        dataset_state = payload.get("dataset_state")
        dataset_index = payload.get("dataset_index")
        if not isinstance(dataset_state, tuple) or not isinstance(dataset_index, list):
            return None

        normalized_index: list[dict] = []
        for entry in dataset_index:
            if not isinstance(entry, dict):
                continue
            normalized = dict(entry)
            embedding = normalized.get("embedding")
            if embedding is not None and not isinstance(embedding, np.ndarray):
                try:
                    normalized["embedding"] = np.array(embedding, dtype=float)
                except Exception:  # pragma: no cover - defensive programming
                    logger.debug("Failed to coerce cached embedding to ndarray: %r", embedding)
                    continue
            normalized_index.append(normalized)

        if len(normalized_index) != len(dataset_index):
            return None

        return dataset_state, normalized_index

    def _save_to_disk(
        self,
        model_name: str,
        detector_backend: str,
        enforce_detection: bool,
        dataset_state: Tuple[Tuple[str, int, int], ...],
        dataset_index: list[dict],
    ) -> None:
        cache_file = self._cache_file_path(model_name, detector_backend, enforce_detection)
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload_index: list[dict] = []
            for entry in dataset_index:
                normalized = dict(entry)
                embedding = normalized.get("embedding")
                if isinstance(embedding, np.ndarray):
                    normalized["embedding"] = embedding.astype(float).tolist()
                elif isinstance(embedding, (list, tuple)):
                    normalized["embedding"] = [float(value) for value in embedding]
                payload_index.append(normalized)

            payload = {
                "dataset_state": dataset_state,
                "dataset_index": payload_index,
            }
            serialized = pickle.dumps(payload)
            encrypted = self._encryption.encrypt(serialized)
            cache_file.write_bytes(encrypted)
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.warning("Failed to persist cached embeddings %s: %s", cache_file, exc)

    def get_dataset_index(
        self,
        model_name: str,
        detector_backend: str,
        enforce_detection: bool,
        builder: Callable[[], list[dict]],
    ) -> list[dict]:
        """Return cached embeddings, building them if the dataset has changed."""

        key = (model_name, detector_backend, enforce_detection)
        current_state = self._current_dataset_state()

        with self._lock:
            memory_entry = self._memory_cache.get(key)
            if memory_entry and memory_entry[0] == current_state:
                return memory_entry[1]

            disk_entry = self._load_from_disk(model_name, detector_backend, enforce_detection)
            if disk_entry and disk_entry[0] == current_state:
                self._memory_cache[key] = disk_entry
                return disk_entry[1]

        dataset_index = builder()
        refreshed_state = self._current_dataset_state()

        with self._lock:
            entry = (refreshed_state, dataset_index)
            self._memory_cache[key] = entry
            self._save_to_disk(
                model_name,
                detector_backend,
                enforce_detection,
                refreshed_state,
                dataset_index,
            )

        return dataset_index

    def invalidate(self) -> None:
        """Clear the in-memory cache and remove cached files from disk."""

        with self._lock:
            self._memory_cache.clear()

        cache.delete("recognition:dataset_state")

        for cache_file in self._cache_root.glob("representations_*.pkl"):
            try:
                cache_file.unlink()
            except FileNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover - defensive programming
                logger.warning("Failed to delete cache file %s: %s", cache_file, exc)


_dataset_embedding_cache = DatasetEmbeddingCache(TRAINING_DATASET_ROOT, DATA_ROOT)


def _ensure_directory(path: Path) -> None:
    """
    Ensure the parent directory for the given path exists.

    If the directory does not exist, it is created. This is useful for
    ensuring that file paths for saving graphs are valid.

    Args:
        path: The file path whose parent directory needs to exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _media_url_for(path: Path) -> str:
    """Return the MEDIA_URL-relative URL for the provided file path."""

    media_root = Path(settings.MEDIA_ROOT)
    try:
        relative_path = path.resolve().relative_to(media_root.resolve())
    except ValueError:
        relative_path = path
    return urljoin(settings.MEDIA_URL, relative_path.as_posix())


def _save_plot_to_media(path: Path) -> str:
    """Persist the current Matplotlib plot and return its media URL."""

    _ensure_directory(path)
    try:
        plt.savefig(path)
    finally:
        plt.close()
    return _media_url_for(path)


def _decrypt_image_bytes(image_path: Path) -> Optional[bytes]:
    """Return decrypted bytes for the encrypted image stored on disk."""
    try:
        encrypted_bytes = image_path.read_bytes()
    except FileNotFoundError:
        logger.warning("Image path %s is missing while building dataset index.", image_path)
        return None
    except OSError as exc:
        logger.error("Failed to read encrypted image %s: %s", image_path, exc)
        return None

    try:
        decrypted_bytes = decrypt_bytes(encrypted_bytes)
    except InvalidToken:
        logger.error("Invalid encryption token encountered for %s", image_path)
        return None
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.error("Unexpected error decrypting %s: %s", image_path, exc)
        return None

    return decrypted_bytes


def _decode_image_bytes(
    decrypted_bytes: bytes, *, source: Optional[Path] = None
) -> Optional[np.ndarray]:
    """Decode decrypted image bytes into a numpy array."""

    frame_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
    if frame_array.size == 0:
        if source is not None:
            logger.warning("Decrypted image %s is empty.", source)
        else:
            logger.warning("Encountered empty decrypted image payload.")
        return None

    imread_flag = getattr(cv2, "IMREAD_COLOR", 1)
    image = cv2.imdecode(frame_array, imread_flag)
    if image is None:
        if source is not None:
            logger.warning("Failed to decode decrypted image %s", source)
        else:
            logger.warning("Failed to decode decrypted image payload.")
        return None
    return image


def _load_encrypted_image(image_path: Path) -> Optional[np.ndarray]:
    """Decrypt and load an encrypted image stored on disk."""

    decrypted_bytes = _decrypt_image_bytes(image_path)
    if decrypted_bytes is None:
        return None

    return _decode_image_bytes(decrypted_bytes, source=image_path)


def _get_or_compute_cached_embedding(
    image_path: Path,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool = False,
) -> Optional[np.ndarray]:
    """Return the embedding for an encrypted image using Django's cache."""

    decrypted_bytes = _decrypt_image_bytes(image_path)
    if decrypted_bytes is None:
        return None

    payload_hash = hashlib.sha256(decrypted_bytes).hexdigest()
    cache_key = f"recognition:embedding:{model_name}:{detector_backend}:{payload_hash}"

    cached_embedding = cache.get(cache_key)
    if cached_embedding is not None:
        try:
            return np.array(cached_embedding, dtype=float)
        except Exception:  # pragma: no cover - defensive programming
            logger.debug("Failed to coerce cached embedding for %s into an array", image_path)

    image = _decode_image_bytes(decrypted_bytes, source=image_path)
    if image is None:
        return None

    try:
        representations = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
        )
    except Exception as exc:
        logger.debug("Failed to generate embedding for %s: %s", image_path, exc)
        return None

    embedding_vector, _ = extract_embedding(representations)
    if embedding_vector is None:
        logger.debug("No embedding produced for %s", image_path)
        return None

    embedding_array = np.array(embedding_vector, dtype=float)

    try:
        cache.set(cache_key, embedding_array.tolist(), timeout=None)
    except Exception:  # pragma: no cover - defensive programming
        logger.debug("Failed to store embedding for %s in cache", image_path)

    return embedding_array


def _build_dataset_embeddings_for_matching(
    model_name: str, detector_backend: str, enforce_detection: bool = False
):
    """Build embeddings for the encrypted training dataset."""

    dataset_index = []
    image_paths = sorted(TRAINING_DATASET_ROOT.glob("*/*.jpg"))

    for image_path in image_paths:
        embedding_array = _get_or_compute_cached_embedding(
            image_path, model_name, detector_backend, enforce_detection
        )
        if embedding_array is None:
            continue

        dataset_index.append(
            {
                "identity": str(image_path),
                "embedding": embedding_array,
                "username": image_path.parent.name,
            }
        )

    return dataset_index


def _load_dataset_embeddings_for_matching(
    model_name: str, detector_backend: str, enforce_detection: bool
):
    """Return embeddings from the cache, computing them if necessary."""

    return _dataset_embedding_cache.get_dataset_index(
        model_name,
        detector_backend,
        enforce_detection,
        lambda: _build_dataset_embeddings_for_matching(
            model_name, detector_backend, enforce_detection
        ),
    )


def username_present(username: str) -> bool:
    """
    Return whether the given username exists in the system.

    Args:
        username: The username to check.

    Returns:
        True if the user exists, False otherwise.
    """
    return User.objects.filter(username=username).exists()


def update_attendance_in_db_in(
    present: Dict[str, bool],
    *,
    attempt_ids: Mapping[str, int] | None = None,
) -> None:
    """
    Persist check-in attendance information for the provided users.

    This function records the time-in for recognized users. It creates a `Present`
    record if one doesn't exist for the day, and adds a `Time` entry for the check-in.

    Args:
        present: A dictionary mapping usernames to their presence status (True).
    """
    _record_sentry_breadcrumb(
        message="Persisting check-in attendance",
        category="attendance.persistence",
        data={
            "direction": "in",
            "user_count": len(present),
            "attempt_ids": bool(attempt_ids),
        },
    )

    today = timezone.localdate()
    current_time = timezone.now()
    attempt_ids = attempt_ids or {}
    for person, is_present in present.items():
        user = User.objects.filter(username=person).first()
        if user is None:
            logger.warning(
                "Skipping check-in attendance update for unknown user '%s'. "
                "Training data may be stale.",
                person,
                extra={
                    "flow": "attendance_persistence",
                    "direction": "in",
                    "username": person,
                },
            )
            _record_sentry_breadcrumb(
                message="Unknown user during check-in persistence",
                category="attendance.persistence",
                level="warning",
                data={"direction": "in", "username": person},
            )
            attempt_id = attempt_ids.get(person)
            if attempt_id:
                RecognitionAttempt.objects.filter(id=attempt_id).update(
                    successful=False,
                    error_message="User not found while persisting check-in.",
                )
            continue

        # Find or create the attendance record for the day
        qs, created = Present.objects.get_or_create(
            user=user, date=today, defaults={"present": is_present}
        )

        if not created and is_present and not qs.present:
            # Mark as present if they were previously marked absent
            qs.present = True
            qs.save(update_fields=["present"])

        if is_present:
            # Record the check-in time
            time_record = Time.objects.create(
                user=user, date=today, time=current_time, direction=Direction.IN
            )
        else:
            time_record = None

        attempt_id = attempt_ids.get(person)
        if attempt_id:
            attempt_updates: Dict[str, Any] = {
                "user": user,
                "present_record": qs,
            }
            if time_record is not None:
                attempt_updates["time_record"] = time_record
            RecognitionAttempt.objects.filter(id=attempt_id).update(**attempt_updates)


def update_attendance_in_db_out(
    present: Dict[str, bool],
    *,
    attempt_ids: Mapping[str, int] | None = None,
) -> None:
    """
    Persist check-out attendance information for the provided users.

    This function records the time-out for recognized users by creating a `Time`
    entry with the `out` flag set to True.

    Args:
        present: A dictionary mapping usernames to their presence status (True).
    """
    _record_sentry_breadcrumb(
        message="Persisting check-out attendance",
        category="attendance.persistence",
        data={
            "direction": "out",
            "user_count": len(present),
            "attempt_ids": bool(attempt_ids),
        },
    )

    today = timezone.localdate()
    current_time = timezone.now()
    attempt_ids = attempt_ids or {}
    for person, is_present in present.items():
        if not is_present:
            continue

        user = User.objects.filter(username=person).first()
        if user is None:
            logger.warning(
                "Skipping check-out attendance update for unknown user '%s'. "
                "Training data may be stale.",
                person,
                extra={
                    "flow": "attendance_persistence",
                    "direction": "out",
                    "username": person,
                },
            )
            _record_sentry_breadcrumb(
                message="Unknown user during check-out persistence",
                category="attendance.persistence",
                level="warning",
                data={"direction": "out", "username": person},
            )
            attempt_id = attempt_ids.get(person)
            if attempt_id:
                RecognitionAttempt.objects.filter(id=attempt_id).update(
                    successful=False,
                    error_message="User not found while persisting check-out.",
                )
            continue
        # Record the check-out time
        time_record = Time.objects.create(
            user=user, date=today, time=current_time, direction=Direction.OUT
        )

        attempt_id = attempt_ids.get(person)
        if attempt_id:
            present_record = Present.objects.filter(user=user, date=today).first()
            attempt_updates: Dict[str, Any] = {
                "user": user,
                "time_record": time_record,
            }
            if present_record is not None:
                attempt_updates["present_record"] = present_record
            RecognitionAttempt.objects.filter(id=attempt_id).update(**attempt_updates)


def check_validity_times(times_all: QuerySet[Time]) -> tuple[bool, float]:
    """
    Validate and calculate break hours from a sequence of time entries.

    This function checks if the time entries follow a valid in-out sequence and
    calculates the total break time in hours.

    Args:
        times_all: A queryset of `Time` objects for a user on a given day,
                   ordered by time.

    Returns:
        A tuple containing:
        - A boolean indicating if the time sequence is valid.
        - The total break hours as a float.
    """
    if not times_all:
        return True, 0  # No records, so no invalidity

    first_entry = times_all.first()
    if first_entry is None or first_entry.time is None:
        return True, 0

    # The first entry must be a check-in
    if first_entry.direction == Direction.OUT:
        return False, 0

    # The number of check-ins must equal the number of check-outs
    if (
        times_all.filter(direction=Direction.IN).count()
        != times_all.filter(direction=Direction.OUT).count()
    ):
        return False, 0

    break_hours = 0
    prev_time = None
    is_break = False

    for entry in times_all:
        if entry.direction == Direction.IN:  # This is a check-in
            if is_break and prev_time is not None and entry.time is not None:
                # Calculate time since the last check-out
                break_duration = (entry.time - prev_time).total_seconds() / 3600
                break_hours += break_duration
            is_break = False
        else:  # This is a check-out
            is_break = True
        prev_time = entry.time

    return True, break_hours


def convert_hours_to_hours_mins(hours: float) -> str:
    """
    Convert a float representing hours into a 'h hrs m mins' format.

    Args:
        hours: The total hours as a float.

    Returns:
        A formatted string (e.g., "8 hrs 30 mins").
    """
    h = int(hours)
    minutes = (hours - h) * 60
    m = math.ceil(minutes)
    return f"{h} hrs {m} mins"


def hours_vs_date_given_employee(
    present_qs: QuerySet[Present], time_qs: QuerySet[Time], admin: bool = True
) -> Tuple[QuerySet[Present], str]:
    """
    Calculate work and break hours for an employee over a date range and generate a plot.

    Args:
        present_qs: A queryset of `Present` objects for the employee.
        time_qs: A queryset of `Time` objects for the employee.
        admin: A boolean indicating if the view is for an admin (affects save path).

    Returns:
        A tuple containing the annotated queryset and the media URL of the generated plot.
    """
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []

    for obj in present_qs:
        date = obj.date
        times_all = time_qs.filter(date=date).order_by("time")
        times_in = times_all.filter(direction=Direction.IN)
        times_out = times_all.filter(direction=Direction.OUT)

        # Use intermediate variables to avoid calling .time on None
        first_in = times_in.first()
        last_out = times_out.last()
        obj.time_in = first_in.time if first_in else None
        obj.time_out = last_out.time if last_out else None

        hours_val = 0.0
        if obj.time_in and obj.time_out:
            hours_val = (obj.time_out - obj.time_in).total_seconds() / 3600

        is_valid, break_hours_val = check_validity_times(times_all)
        if not is_valid:
            break_hours_val = 0.0

        df_hours.append(hours_val)
        df_break_hours.append(break_hours_val)

        # Format for display
        obj.hours = convert_hours_to_hours_mins(hours_val)
        obj.break_hours = convert_hours_to_hours_mins(break_hours_val)

    # Generate and save the plot
    df = read_frame(present_qs, fieldnames=["date"])
    df["hours"] = df_hours
    df["break_hours"] = df_break_hours
    logger.debug("Attendance dataframe for employee: %s", df)

    sns.barplot(data=df, x="date", y="hours")
    plt.xticks(rotation="vertical")
    rcParams.update({"figure.autolayout": True})
    plt.tight_layout()

    target_path = HOURS_VS_DATE_PATH if admin else EMPLOYEE_LOGIN_PATH
    chart_url = _save_plot_to_media(target_path)

    return present_qs, chart_url


def hours_vs_employee_given_date(
    present_qs: QuerySet[Present], time_qs: QuerySet[Time]
) -> Tuple[QuerySet[Present], str]:
    """
    Calculate work and break hours for all employees on a given date and generate a plot.

    Args:
        present_qs: A queryset of `Present` objects for the date.
        time_qs: A queryset of `Time` objects for the date.

    Returns:
        A tuple containing the annotated queryset and the media URL of the generated plot.
    """
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    df_username = []

    for obj in present_qs:
        user = obj.user
        times_all = time_qs.filter(user=user).order_by("time")
        times_in = times_all.filter(direction=Direction.IN)
        times_out = times_all.filter(direction=Direction.OUT)

        # Use intermediate variables to avoid calling .time on None
        first_in = times_in.first()
        last_out = times_out.last()
        obj.time_in = first_in.time if first_in else None
        obj.time_out = last_out.time if last_out else None

        hours_val = 0.0
        if obj.time_in and obj.time_out:
            hours_val = (obj.time_out - obj.time_in).total_seconds() / 3600

        is_valid, break_hours_val = check_validity_times(times_all)
        if not is_valid:
            break_hours_val = 0.0

        df_hours.append(hours_val)
        df_username.append(user.username)
        df_break_hours.append(break_hours_val)

        # Format for display
        obj.hours = convert_hours_to_hours_mins(hours_val)
        obj.break_hours = convert_hours_to_hours_mins(break_hours_val)

    # Generate and save the plot
    df = read_frame(present_qs, fieldnames=["user"])
    df["hours"] = df_hours
    df["username"] = df_username
    df["break_hours"] = df_break_hours

    sns.barplot(data=df, x="username", y="hours")
    plt.xticks(rotation="vertical")
    rcParams.update({"figure.autolayout": True})
    plt.tight_layout()
    chart_url = _save_plot_to_media(HOURS_VS_EMPLOYEE_PATH)

    return present_qs, chart_url


def total_number_employees() -> int:
    """
    Return the total count of non-staff, non-superuser employees.
    """
    return User.objects.filter(is_staff=False, is_superuser=False).count()


def employees_present_today() -> int:
    """
    Return the count of employees marked as present today.
    """
    today = timezone.localdate()
    return Present.objects.filter(date=today, present=True).count()


def this_week_emp_count_vs_date() -> Optional[str]:
    """
    Generate and save a line plot of employee presence for the current week.

    Returns:
        The media URL of the generated plot, or ``None`` when no data is available.
    """
    today = timezone.localdate()
    start_of_week = today - datetime.timedelta(days=today.weekday())

    # Get attendance data for the current week
    qs = (
        Present.objects.filter(date__gte=start_of_week, date__lte=today, present=True)
        .values("date")
        .annotate(emp_count=Count("user"))
        .order_by("date")
    )

    # Create a dictionary for quick lookup
    attendance_by_date = {item["date"]: item["emp_count"] for item in qs}

    # Prepare data for the plot (Monday to Friday)
    str_dates_all = []
    emp_cnt_all = []
    for i in range(5):
        current_date = start_of_week + datetime.timedelta(days=i)
        if current_date > today:
            break
        str_dates_all.append(current_date.strftime("%Y-%m-%d"))
        emp_cnt_all.append(attendance_by_date.get(current_date, 0))

    if not str_dates_all:
        return None  # Avoid plotting if there's no data

    df = pd.DataFrame({"date": str_dates_all, "Number of employees": emp_cnt_all})

    sns.lineplot(data=df, x="date", y="Number of employees")
    return _save_plot_to_media(THIS_WEEK_PATH)


def last_week_emp_count_vs_date() -> Optional[str]:
    """
    Generate and save a line plot of employee presence for the last week.

    Returns:
        The media URL of the generated plot, or ``None`` when no data is available.
    """
    today = timezone.localdate()
    start_of_last_week = today - datetime.timedelta(days=today.weekday() + 7)
    end_of_last_week = start_of_last_week + datetime.timedelta(days=4)

    # Get attendance data for the last week
    qs = (
        Present.objects.filter(
            date__gte=start_of_last_week, date__lte=end_of_last_week, present=True
        )
        .values("date")
        .annotate(emp_count=Count("user"))
        .order_by("date")
    )

    attendance_by_date = {item["date"]: item["emp_count"] for item in qs}

    # Prepare data for the plot (Monday to Friday of last week)
    str_dates_all = []
    emp_cnt_all = []
    for i in range(5):
        current_date = start_of_last_week + datetime.timedelta(days=i)
        str_dates_all.append(current_date.strftime("%Y-%m-%d"))
        emp_cnt_all.append(attendance_by_date.get(current_date, 0))

    df = pd.DataFrame({"date": str_dates_all, "emp_count": emp_cnt_all})

    sns.lineplot(data=df, x="date", y="emp_count")
    return _save_plot_to_media(LAST_WEEK_PATH)


# ========== Main Views ==========


def home(request):
    """
    Render the home page.
    """
    return render(request, "recognition/home.html")


@login_required
def dashboard(request):
    """
    Render the dashboard, which differs for admins and regular employees.
    """
    if request.user.is_staff or request.user.is_superuser:
        logger.debug("Rendering admin dashboard for %s", request.user)
        dataset_snapshot = health.dataset_health()
        model_snapshot = health.model_health(
            dataset_last_updated=dataset_snapshot.get("last_updated")
        )
        onboarding_state = _build_onboarding_state(
            dataset_snapshot=dataset_snapshot, model_snapshot=model_snapshot
        )
        return render(
            request,
            "recognition/admin_dashboard.html",
            {
                "dataset_snapshot": dataset_snapshot,
                "model_snapshot": model_snapshot,
                "onboarding_state": onboarding_state,
            },
        )

    logger.debug("Rendering employee dashboard for %s", request.user)
    return render(request, "recognition/employee_dashboard.html")


def _build_onboarding_state(
    *, dataset_snapshot: Dict[str, Any], model_snapshot: Dict[str, Any]
) -> Dict[str, Any]:
    """Summarize first-run readiness and suggested actions for admins.

    The checklist covers all prerequisites for end-to-end recognition:
    1. Encryption keys configured (FACE_DATA_ENCRYPTION_KEY)
    2. Webcam connected and accessible
    3. At least one employee registered
    4. At least one employee with photos captured
    5. Recognition model trained

    See USER_GUIDE.md for detailed setup instructions.
    """
    # Base URL for documentation links - use the actual repo
    docs_base_url = (
        "https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/blob/main"
    )

    user_count = get_user_model().objects.filter(is_staff=False, is_superuser=False).count()
    image_count = dataset_snapshot.get("image_count", 0)
    identity_count = dataset_snapshot.get("identity_count", 0)
    onboarding_steps = []

    # Step 1: Encryption keys must be configured for secure biometric storage
    if not os.environ.get("FACE_DATA_ENCRYPTION_KEY"):
        onboarding_steps.append(
            {
                "title": "Configure encryption keys",
                "description": (
                    "Set FACE_DATA_ENCRYPTION_KEY in your environment to encrypt "
                    "biometric data at rest. See the Deployment Guide for details."
                ),
                "cta": {
                    "url": f"{docs_base_url}/DEPLOYMENT.md#3-configuration",
                    "label": "View Config Guide",
                    "icon": "fa-key",
                },
            },
        )

    # Step 2: Webcam availability check (shown when no employees or photos exist)
    if user_count == 0 and image_count == 0:
        onboarding_steps.append(
            {
                "title": "Connect a webcam",
                "description": (
                    "Ensure a webcam is connected and accessible for capturing photos. "
                    "Grant browser camera permissions when prompted."
                ),
                "cta": {
                    "url": f"{docs_base_url}/USER_GUIDE.md#troubleshooting",
                    "label": "Troubleshooting Guide",
                    "icon": "fa-video",
                },
            },
        )

    # Step 3: At least one employee must be registered
    if user_count == 0:
        onboarding_steps.append(
            {
                "title": "Register your first employee",
                "description": (
                    "Add at least one employee account. Each employee needs a username "
                    "and profile before you can capture their photos."
                ),
                "cta": {
                    "url": reverse("register"),
                    "label": "Register Employee",
                    "icon": "fa-user-plus",
                },
            }
        )

    # Step 4: At least one employee must have photos captured
    if image_count == 0:
        onboarding_steps.append(
            {
                "title": "Capture employee photos",
                "description": (
                    "Use the webcam to capture face photos for each employee. "
                    "Multiple photos improve recognition accuracy."
                ),
                "cta": {
                    "url": reverse("add-photos"),
                    "label": "Add Photos",
                    "icon": "fa-camera",
                },
            }
        )
    elif user_count > 0 and identity_count < user_count:
        # Some employees exist but not all have photos
        onboarding_steps.append(
            {
                "title": "Add photos for remaining employees",
                "description": (
                    f"Only {identity_count} of {user_count} employees have photos. "
                    "Capture photos for all employees to enable recognition."
                ),
                "cta": {
                    "url": reverse("add-photos"),
                    "label": "Add Photos",
                    "icon": "fa-camera",
                },
            }
        )

    # Step 5: Model must be trained
    if not model_snapshot.get("model_present"):
        onboarding_steps.append(
            {
                "title": "Train the recognition model",
                "description": (
                    "Start training once photos are captured. The model learns to "
                    "recognize employees from their face embeddings."
                ),
                "cta": {
                    "url": reverse("train"),
                    "label": "Start Training",
                    "icon": "fa-brain",
                },
            }
        )
    elif model_snapshot.get("stale"):
        # Model exists but is outdated
        onboarding_steps.append(
            {
                "title": "Retrain the model",
                "description": (
                    "New photos were added since the last training. "
                    "Retrain to include the latest employee data."
                ),
                "cta": {
                    "url": reverse("train"),
                    "label": "Retrain Now",
                    "icon": "fa-rotate-right",
                },
            }
        )

    readiness = {
        "has_encryption_key": bool(os.environ.get("FACE_DATA_ENCRYPTION_KEY")),
        "has_users": user_count > 0,
        "has_dataset": image_count > 0,
        "all_users_have_photos": user_count > 0 and identity_count >= user_count,
        "model_ready": bool(model_snapshot.get("model_present")),
        "model_fresh": not model_snapshot.get("stale", False),
    }

    return {
        "user_count": user_count,
        "identity_count": identity_count,
        "readiness": readiness,
        "steps": onboarding_steps,
        "model_stale": model_snapshot.get("stale", False),
    }


@login_required
def add_photos(request):
    """
    Handle the 'Add Photos' functionality for admins to create face datasets for users.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    task_context: Dict[str, Any] | None = None
    task_id = request.GET.get("task_id")
    if task_id:
        task_context = _describe_async_result(task_id)

    if request.method == "POST":
        form = usernameForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            if username_present(username):
                try:
                    from .tasks import capture_dataset

                    async_result = capture_dataset.delay(username)
                except Exception:
                    logger.exception("Failed to enqueue dataset capture for %s", username)
                    messages.error(
                        request,
                        "Unable to start dataset capture at this time. Please try again shortly.",
                    )
                else:
                    messages.success(
                        request,
                        (
                            f"Dataset capture for {username} started in the background. "
                            f"Track progress with task ID {async_result.id}."
                        ),
                    )
                    return redirect(f"{reverse('add-photos')}?task_id={async_result.id}")

            messages.warning(request, "No such username found. Please register employee first.")
            return redirect("add-photos")
    else:
        form = usernameForm()

    context: Dict[str, Any] = {"form": form}
    if task_context:
        context["task"] = task_context

    return render(request, "recognition/add_photos.html", context)


# Default distance threshold for face recognition confidence
DEFAULT_DISTANCE_THRESHOLD = 0.4


LIVENESS_FAILURE_MESSAGE = (
    "Liveness check failed. Please blink or move your head slightly and try again "
    "before marking attendance."
)


def _normalize_face_region(
    region: Optional[Dict[str, int]],
) -> Optional[Dict[str, int]]:
    """Normalize various facial area representations into ``x, y, w, h`` form."""

    if not isinstance(region, dict):
        return None

    keys = {key.lower(): value for key, value in region.items() if value is not None}

    def _get_value(*aliases: str) -> Optional[int]:
        for alias in aliases:
            if alias in keys:
                try:
                    return int(keys[alias])
                except (TypeError, ValueError):
                    return None
        return None

    x = _get_value("x", "left")
    y = _get_value("y", "top")
    w = _get_value("w", "width")
    h = _get_value("h", "height")

    if None not in (x, y, w, h):
        return {"x": x, "y": y, "w": w, "h": h}

    right = _get_value("right")
    bottom = _get_value("bottom")
    if None not in (x, right, y, bottom):
        width = right - x
        height = bottom - y
        if width > 0 and height > 0:
            return {"x": x, "y": y, "w": width, "h": height}

    return None


def _crop_face_region(frame: np.ndarray, region: Optional[Dict[str, int]]) -> np.ndarray:
    """Return a cropped face image given a bounding box region."""

    if frame is None or not hasattr(frame, "shape"):
        return frame

    normalized = _normalize_face_region(region)
    if not normalized:
        return frame

    height, width = frame.shape[:2]
    x = max(normalized.get("x", 0), 0)
    y = max(normalized.get("y", 0), 0)
    w = max(normalized.get("w", 0), 0)
    h = max(normalized.get("h", 0), 0)

    if w <= 0 or h <= 0:
        return frame

    x2 = min(x + w, width)
    y2 = min(y + h, height)
    if x >= x2 or y >= y2:
        return frame

    return frame[y:y2, x:x2]


def _interpret_deepface_liveness_result(result: object) -> Optional[bool]:
    """Best-effort interpretation of DeepFace anti-spoofing output."""

    if isinstance(result, (list, tuple)):
        if not result:
            return None
        return _interpret_deepface_liveness_result(result[0])

    if not isinstance(result, dict):
        return None

    bool_keys = (
        "is_real",
        "is_real_face",
        "is_live",
        "is_genuine",
        "real",
        "live",
        "genuine",
    )
    for key in bool_keys:
        value = result.get(key)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)

    score_keys = ("real", "live", "genuine")
    threshold = float(getattr(settings, "RECOGNITION_LIVENESS_SCORE_THRESHOLD", 0.5))
    for key in score_keys:
        value = result.get(key)
        if isinstance(value, (int, float)):
            return value >= threshold

    face_type = result.get("face_type")
    if isinstance(face_type, str):
        return face_type.lower() in {"real", "live", "genuine"}
    if isinstance(face_type, dict):
        label = face_type.get("type") or face_type.get("value")
        if isinstance(label, str):
            return label.lower() in {"real", "live", "genuine"}

    return None


def _deepface_liveness_check(frame: np.ndarray) -> Optional[bool]:
    """Run DeepFace anti-spoofing on the provided frame."""

    try:
        analysis = DeepFace.analyze(  # type: ignore[call-arg]
            img_path=frame,
            actions=("anti-spoof",),
            enforce_detection=False,
            detector_backend=_get_face_detection_backend(),
            silent=True,
        )
    except Exception as exc:
        logger.warning("Liveness check failed: %s", exc)
        return None

    decision = _interpret_deepface_liveness_result(analysis)
    if decision is None:
        logger.warning("Could not interpret DeepFace anti-spoofing result: %s", analysis)
    return decision


def _passes_liveness_check(
    frame: Optional[np.ndarray],
    face_region: Optional[Dict[str, int]] = None,
    *,
    frame_history: Optional[Sequence[np.ndarray]] = None,
) -> bool:
    """Return ``True`` when the supplied frame passes all configured liveness gates."""

    lightweight_enabled = _is_lightweight_liveness_enabled()
    deepface_enabled = _is_liveness_enabled()
    enhanced_enabled = _is_enhanced_liveness_enabled()

    if not lightweight_enabled and not deepface_enabled and not enhanced_enabled:
        return True

    if lightweight_enabled:
        history_frames = list(frame_history or [])
        if frame is not None and not history_frames:
            history_frames = [frame]
        min_frames = _get_lightweight_liveness_min_frames()
        threshold = _get_lightweight_liveness_threshold()
        score = is_live_face(
            history_frames,
            face_region=face_region,
            min_frames=min_frames,
            return_score=True,
        )
        if score is not None:
            logger.debug(
                "Lightweight liveness score %.4f (threshold %.4f)",
                score,
                threshold,
            )
            if score < threshold:
                logger.info(
                    "Motion-based liveness rejected frame (score %.4f < %.4f)",
                    score,
                    threshold,
                )
                return False
        else:
            logger.debug("Insufficient data for lightweight liveness evaluation.")

    # Enhanced liveness verification (CNN + depth + frame consistency)
    if enhanced_enabled and frame_history and len(frame_history) >= 5:
        try:
            from recognition.liveness import run_enhanced_liveness_verification
            from recognition.views.config import (
                get_cnn_antispoof_threshold,
                get_depth_variance_threshold,
            )

            enhanced_result = run_enhanced_liveness_verification(
                list(frame_history),
                face_region=face_region,
                cnn_threshold=get_cnn_antispoof_threshold(),
                depth_variance_threshold=get_depth_variance_threshold(),
            )
            if not enhanced_result.passed:
                logger.info(
                    "Enhanced liveness rejected: %s",
                    enhanced_result.failure_reasons,
                )
                return False
            logger.debug(
                "Enhanced liveness passed with confidence %.4f",
                enhanced_result.confidence,
            )
        except ImportError:
            logger.debug("Enhanced liveness modules not available, skipping.")
        except Exception as exc:
            logger.warning("Enhanced liveness check failed: %s", exc)

    if not deepface_enabled:
        return True

    if frame is None:
        logger.debug("Skipping DeepFace liveness check because no reference frame was provided.")
        return True

    custom_checker = getattr(settings, "RECOGNITION_LIVENESS_CHECKER", None)
    if callable(custom_checker):
        try:
            decision = custom_checker(frame=frame, face_region=face_region)
            if decision is not None:
                return bool(decision)
        except Exception as exc:
            logger.warning("Custom liveness checker raised an exception: %s", exc)

    cropped = _crop_face_region(frame, face_region)
    decision = _deepface_liveness_check(cropped)
    return bool(decision)


def _evaluate_recognition_match(
    frame: np.ndarray,
    match: pd.Series,
    distance_threshold: float,
    *,
    frame_history: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[Optional[str], bool, Optional[Dict[str, int]]]:
    """Evaluate a DeepFace match result and run the liveness gate."""

    try:
        distance = float(match.get("distance", math.inf))
    except (TypeError, ValueError):
        distance = math.inf

    if not is_within_distance_threshold(distance, distance_threshold):
        return None, False, None

    identity_value = match.get("identity")
    username: Optional[str] = None
    if identity_value:
        try:
            identity_path = Path(str(identity_value))
            parent_name = identity_path.parent.name
            if parent_name:
                username = parent_name
        except Exception as exc:
            logger.debug("Could not parse identity path '%s': %s", identity_value, exc)

    face_region = {
        "x": match.get("source_x"),
        "y": match.get("source_y"),
        "w": match.get("source_w"),
        "h": match.get("source_h"),
    }
    normalized_region = _normalize_face_region(face_region)

    if not _passes_liveness_check(
        frame,
        normalized_region,
        frame_history=frame_history,
    ):
        return username, True, normalized_region

    return username, False, normalized_region


def _predict_identity_from_embedding(
    frame: np.ndarray,
    embedding_vector: Sequence[float],
    facial_area: Optional[Dict[str, int]],
    model,
    class_names: Sequence[str],
    attendance_type: str,
    *,
    frame_history: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[Optional[str], bool, Optional[Dict[str, int]]]:
    """Run liveness verification and model prediction for an embedding."""

    normalized_region = _normalize_face_region(facial_area)
    if not _passes_liveness_check(
        frame,
        normalized_region,
        frame_history=frame_history,
    ):
        logger.warning("Spoofing attempt blocked during '%s' attendance.", attendance_type)
        return None, True, normalized_region

    try:
        prediction = model.predict(np.array([embedding_vector]))
    except Exception as exc:
        logger.warning("Failed to classify embedding: %s", exc)
        return None, False, normalized_region

    if isinstance(prediction, (list, tuple, np.ndarray)) and len(prediction) > 0:
        predicted_index = int(prediction[0])
    else:
        try:
            predicted_index = int(prediction)
        except (TypeError, ValueError):
            logger.warning("Received unexpected prediction output: %s", prediction)
            return None, False, normalized_region

    if 0 <= predicted_index < len(class_names):
        predicted_name = str(class_names[predicted_index])
        return predicted_name, False, normalized_region

    logger.warning("Predicted index %s out of range", predicted_index)
    return None, False, normalized_region


def _is_headless_environment() -> bool:
    """Return ``True`` when no graphical display is available."""

    override = getattr(settings, "RECOGNITION_HEADLESS", None)
    if override is not None:
        return bool(override)

    # Windows typically has a display available when running interactively
    if sys.platform.startswith("win"):
        return False

    display_vars = ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")
    if any(os.environ.get(var) for var in display_vars):
        return False

    return True


@login_required
def attendance_session(request):
    """Render a structured attendance session view with live recognition activity."""

    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    dataset_snapshot = health.dataset_health()
    model_snapshot = health.model_health(dataset_last_updated=dataset_snapshot.get("last_updated"))
    onboarding_state = _build_onboarding_state(
        dataset_snapshot=dataset_snapshot, model_snapshot=model_snapshot
    )

    return render(
        request,
        "recognition/attendance_session.html",
        {
            "dataset_snapshot": dataset_snapshot,
            "model_snapshot": model_snapshot,
            "onboarding_state": onboarding_state,
            "recent_activity": health.recognition_activity(),
        },
    )


@login_required
def attendance_session_feed(request) -> JsonResponse:
    """Return a live feed of recent recognition attempts and outcomes for the UI log."""

    if not (request.user.is_staff or request.user.is_superuser):
        return JsonResponse({"detail": "Not authorised"}, status=403)

    try:
        minutes = int(request.GET.get("minutes", "60"))
    except (TypeError, ValueError):
        minutes = 60

    since = timezone.now() - datetime.timedelta(minutes=max(minutes, 1))
    outcome_records = RecognitionOutcome.objects.filter(created_at__gte=since)[:50]
    attempt_records = RecognitionAttempt.objects.filter(created_at__gte=since)[:50]

    events: list[dict[str, Any]] = []
    for outcome in outcome_records:
        events.append(
            {
                "event_type": "outcome",
                "timestamp": outcome.created_at.isoformat(),
                "username": outcome.username,
                "direction": outcome.direction,
                "accepted": outcome.accepted,
                "confidence": outcome.confidence,
                "distance": outcome.distance,
                "threshold": outcome.threshold,
                "source": outcome.source,
            }
        )

    for attempt in attempt_records:
        liveness_status = "failed" if attempt.spoof_detected else "passed"
        events.append(
            {
                "event_type": "attempt",
                "timestamp": attempt.created_at.isoformat(),
                "username": attempt.username or (attempt.user.username if attempt.user else ""),
                "direction": attempt.direction,
                "successful": attempt.successful,
                "liveness": liveness_status,
                "error": attempt.error_message,
                "source": attempt.source,
            }
        )

    events.sort(key=lambda item: item["timestamp"], reverse=True)
    return JsonResponse({"events": events})


def _mark_attendance(request, check_in: bool):
    """
    Core logic for marking attendance (both in and out) using face recognition.

    Args:
        request: The Django HttpRequest object.
        check_in: True for marking time-in, False for time-out.
    """
    manager = get_webcam_manager()
    present = {}
    recorded_outcomes: set[tuple[str, bool]] = set()
    direction = "in" if check_in else "out"

    request_metadata = {
        "direction": direction,
        "path": getattr(request, "path", ""),
        "user_id": getattr(getattr(request, "user", None), "id", None),
        "username": getattr(getattr(request, "user", None), "username", None),
        "remote_addr": request.META.get("REMOTE_ADDR") if hasattr(request, "META") else None,
    }
    _bind_request_to_sentry_scope(request, flow="webcam_attendance", extra=request_metadata)
    _record_sentry_breadcrumb(
        message="Attendance capture started",
        category="attendance.flow",
        data=request_metadata,
    )

    def _log_outcome(candidate: Optional[str], *, accepted: bool, distance: float | None) -> None:
        key = (candidate or "", accepted)
        if key in recorded_outcomes:
            return
        recorded_outcomes.add(key)
        log_recognition_outcome(
            username=candidate,
            accepted=accepted,
            direction=direction,
            distance=distance,
            threshold=distance_threshold,
            source="webcam",
        )

    headless = _is_headless_environment()
    max_frames = int(getattr(settings, "RECOGNITION_HEADLESS_ATTENDANCE_FRAMES", 30))
    frame_pause = float(getattr(settings, "RECOGNITION_HEADLESS_FRAME_SLEEP", 0.01))
    frames_processed = 0

    direction_choice = (
        RecognitionAttempt.Direction.IN if check_in else RecognitionAttempt.Direction.OUT
    )
    attempt_logger = _RecognitionAttemptLogger(
        direction_choice.value,
        _resolve_recognition_site(request),
        source="webcam",
    )

    # Configure DeepFace settings
    model_name = _get_face_recognition_model()
    detector_backend = _get_face_detection_backend()
    enforce_detection = _should_enforce_detection()
    distance_threshold = getattr(
        settings, "RECOGNITION_DISTANCE_THRESHOLD", DEFAULT_DISTANCE_THRESHOLD
    )
    distance_metric = _get_deepface_distance_metric()

    window_title = (
        "Mark Attendance - In - Press q to exit"
        if check_in
        else "Mark Attendance- Out - Press q to exit"
    )

    dataset_load_start = time.perf_counter()
    dataset_index = _load_dataset_embeddings_for_matching(
        model_name, detector_backend, enforce_detection
    )
    dataset_load_duration = time.perf_counter() - dataset_load_start
    monitoring.observe_stage_duration(
        "dataset_index_load",
        dataset_load_duration,
        threshold_key="model_load",
    )
    dataset_size: int | None
    try:
        dataset_size = len(dataset_index)
    except TypeError:
        dataset_size = None
    _record_sentry_breadcrumb(
        message="Attendance dataset loaded",
        category="attendance.dataset",
        data={
            "direction": direction,
            "dataset_size": dataset_size,
            "load_seconds": round(dataset_load_duration, 6),
        },
    )
    if not dataset_index:
        dataset_snapshot = health.dataset_health()
        model_snapshot = health.model_health(
            dataset_last_updated=dataset_snapshot.get("last_updated")
        )
        attempt_logger.log_failure(
            getattr(request.user, "username", None),
            spoofed=False,
            error="No encrypted training data available for matching.",
        )
        logger.warning(
            "No encrypted training data available for attendance matching.",
            extra={
                "flow": "webcam_attendance",
                "direction": direction,
                "dataset_images": dataset_snapshot.get("image_count"),
                "model_present": model_snapshot.get("model_present"),
            },
        )
        _record_sentry_breadcrumb(
            message="Attendance dataset empty",
            category="attendance.dataset",
            level="warning",
            data={
                "direction": direction,
                "load_seconds": round(dataset_load_duration, 6),
                "dataset_images": dataset_snapshot.get("image_count"),
                "model_present": model_snapshot.get("model_present"),
            },
        )
        messages.error(
            request,
            "No encrypted training data available for matching. Please recreate the dataset.",
        )
        return redirect("home")

    spoof_detected = False
    model_warmed_up = False
    liveness_buffer = LivenessBuffer(maxlen=_get_lightweight_liveness_window())

    try:
        with manager.frame_consumer() as consumer:
            while True:
                frame = consumer.read(timeout=1.0)
                if frame is None:
                    continue
                iteration_start = time.perf_counter()
                frame = imutils.resize(frame, width=800)
                liveness_buffer.append(frame)
                frames_processed += 1

                try:
                    # Use DeepFace to find matching faces in the database
                    inference_start = time.perf_counter()
                    representations = DeepFace.represent(
                        img_path=frame,
                        model_name=model_name,
                        detector_backend=detector_backend,
                        enforce_detection=enforce_detection,
                    )
                    inference_duration = time.perf_counter() - inference_start
                    if not model_warmed_up:
                        monitoring.observe_stage_duration(
                            "deepface_warmup",
                            inference_duration,
                            threshold_key="warmup",
                        )
                        model_warmed_up = True
                    else:
                        monitoring.observe_stage_duration(
                            "deepface_inference",
                            inference_duration,
                        )

                    embedding_vector, facial_area = extract_embedding(representations)
                    if embedding_vector is None:
                        continue

                    frame_embedding = np.array(embedding_vector, dtype=float)
                    match = find_closest_dataset_match(
                        frame_embedding, dataset_index, distance_metric
                    )
                    if match is None:
                        continue

                    username, distance_value, identity_path = match
                    candidate_name: Optional[str] = None
                    if username:
                        candidate_name = username
                    elif identity_path:
                        try:
                            candidate_name = Path(identity_path).parent.name
                        except Exception:  # pragma: no cover - defensive parsing
                            candidate_name = None
                    normalized_region = _normalize_face_region(facial_area)
                    spoofed = not _passes_liveness_check(
                        frame,
                        normalized_region,
                        frame_history=liveness_buffer.snapshot(),
                    )

                    if not is_within_distance_threshold(distance_value, distance_threshold):
                        logger.info(
                            "Ignoring potential match for '%s' due to high distance %.4f",
                            Path(identity_path).parent.name,
                            distance_value,
                            extra={
                                "flow": "webcam_attendance",
                                "direction": direction,
                                "threshold": distance_threshold,
                            },
                        )
                        attempt_logger.log_failure(
                            candidate_name,
                            spoofed=False,
                            error=(
                                f"Distance {distance_value:.4f} exceeds threshold"
                                f" {distance_threshold:.4f}"
                            ),
                        )
                        _log_outcome(candidate_name, accepted=False, distance=distance_value)
                        continue

                    if spoofed:
                        spoof_detected = True
                        _log_outcome(candidate_name, accepted=False, distance=distance_value)
                        logger.warning(
                            "Spoofing attempt blocked while marking attendance for '%s'",
                            username or Path(identity_path).parent.name,
                        )
                        fallback_username = username or Path(identity_path).parent.name
                        attempt_logger.log_failure(
                            fallback_username,
                            spoofed=True,
                            error="Spoofing detected by liveness gate.",
                        )

                        if normalized_region:
                            x = int(normalized_region["x"])
                            y = int(normalized_region["y"])
                            w = int(normalized_region["w"])
                            h = int(normalized_region["h"])
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(
                                frame,
                                "Spoof detected",
                                (x, max(y - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )
                        continue

                    if username:
                        present[username] = True
                        _log_outcome(username, accepted=True, distance=distance_value)
                        logger.info("Recognized %s with distance %.4f", username, distance_value)
                        attempt_logger.log_success(username)

                        if normalized_region:
                            x = int(normalized_region["x"])
                            y = int(normalized_region["y"])
                            w = int(normalized_region["w"])
                            h = int(normalized_region["h"])
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                username,
                                (x, max(y - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 255, 0),
                                2,
                            )

                except Exception:
                    logger.exception(
                        "Error during face recognition loop",
                        extra={
                            "flow": "webcam_attendance",
                            "direction": direction,
                        },
                    )
                finally:
                    iteration_duration = time.perf_counter() - iteration_start
                    monitoring.observe_stage_duration(
                        "recognition_iteration",
                        iteration_duration,
                        threshold_key="recognition_iteration",
                    )

                    if not headless:
                        cv2.imshow(window_title, frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    else:
                        if frame_pause:
                            time.sleep(frame_pause)
                        if frames_processed >= max_frames:
                            logger.info(
                                "Headless mode reached frame limit of %d frames",
                                max_frames,
                                extra={
                                    "flow": "webcam_attendance",
                                    "direction": direction,
                                    "frames_processed": frames_processed,
                                },
                            )
                            break
    finally:
        if not headless:
            cv2.destroyAllWindows()

    attempt_logger.ensure_generic_failure("No faces were recognized during the attendance session.")

    if spoof_detected:
        messages.error(request, LIVENESS_FAILURE_MESSAGE)

    if present:
        record: Dict[str, Any] = {
            "direction": direction_choice.value,
            "present": present,
        }
        attempt_ids = attempt_logger.success_attempt_ids
        if attempt_ids:
            record["attempt_ids"] = attempt_ids
        try:
            async_result = _enqueue_attendance_records([record])
            logger.info(
                "Queued attendance batch %s for %s event.",
                async_result.id,
                record["direction"],
                extra={
                    "flow": "webcam_attendance",
                    "direction": direction,
                    "task_id": async_result.id,
                    "recognized_users": sorted(present.keys()),
                },
            )
            _record_sentry_breadcrumb(
                message="Attendance batch enqueued",
                category="attendance.queue",
                data={
                    "direction": direction,
                    "task_id": async_result.id,
                    "user_count": len(present),
                },
            )
        except Exception:  # pragma: no cover - defensive programming
            logger.exception(
                "Failed to enqueue attendance processing; falling back to synchronous update.",
                extra={
                    "flow": "webcam_attendance",
                    "direction": direction,
                },
            )
            _record_sentry_breadcrumb(
                message="Attendance batch enqueue failed",
                category="attendance.queue",
                level="error",
                data={
                    "direction": direction,
                    "user_count": len(present),
                },
            )
            if check_in:
                update_attendance_in_db_in(present, attempt_ids=attempt_ids)
            else:
                update_attendance_in_db_out(present, attempt_ids=attempt_ids)

    return redirect("home")


@login_required
@attendance_rate_limited
def mark_your_attendance(request):
    """View to handle marking time-in."""
    return _mark_attendance(request, check_in=True)


@login_required
@attendance_rate_limited
def mark_your_attendance_out(request):
    """View to handle marking time-out."""
    return _mark_attendance(request, check_in=False)


@staff_member_required
def monitoring_metrics(request):
    """Expose Prometheus metrics for the recognition system."""

    payload = monitoring.export_metrics()
    return HttpResponse(payload, content_type=monitoring.prometheus_content_type())


@login_required
def train(request):
    """Allow administrators to trigger and monitor background model training."""

    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    task_context: Dict[str, Any] | None = None
    task_id = request.GET.get("task_id")
    if task_id:
        task_context = _describe_async_result(task_id)

    if request.method == "POST":
        try:
            from .tasks import train_recognition_model

            async_result = train_recognition_model.delay(
                initiated_by=(
                    getattr(request.user, "get_username", lambda: None)()
                    if hasattr(request.user, "get_username")
                    else getattr(request.user, "username", None)
                )
            )
        except Exception:
            logger.exception("Failed to enqueue training job")
            messages.error(
                request,
                "Unable to start model training. Please retry once the worker is available.",
            )
        else:
            messages.success(
                request,
                (
                    "Model training has started in the background. "
                    f"Track progress with task ID {async_result.id}."
                ),
            )
            return redirect(f"{reverse('train')}?task_id={async_result.id}")

    context: Dict[str, Any] = {}
    if task_context:
        context["task"] = task_context

    return render(request, "recognition/train.html", context)


@login_required
def task_status(request, task_id: str) -> JsonResponse:
    """Return JSON describing the state of a Celery task."""

    if not (request.user.is_staff or request.user.is_superuser):
        return JsonResponse({"detail": "Not authorised"}, status=403)

    payload = _describe_async_result(task_id)
    return JsonResponse(payload)


@login_required
def not_authorised(request):
    """Render a page for users trying to access unauthorized areas."""
    return render(request, "recognition/not_authorised.html")


@login_required
def view_attendance_home(request):
    """
    Render the main attendance viewing page for admins.

    This view displays summary statistics and generates weekly attendance graphs.
    """
    total_num_of_emp = total_number_employees()
    emp_present_today = employees_present_today()
    this_week_graph_url = this_week_emp_count_vs_date()
    last_week_graph_url = last_week_emp_count_vs_date()
    context = {
        "total_num_of_emp": total_num_of_emp,
        "emp_present_today": emp_present_today,
        "this_week_graph_url": this_week_graph_url,
        "last_week_graph_url": last_week_graph_url,
    }
    return render(request, "recognition/view_attendance_home.html", context)


@login_required
def view_attendance_date(request):
    """
    Admin view to see attendance for all employees on a specific date.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    qs: Optional[QuerySet[Present]] = None
    chart_url: Optional[str] = None
    if request.method == "POST":
        form = DateForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data["date"]
            logger.debug("Admin %s viewing attendance for date %s", request.user, date)

            time_qs = Time.objects.filter(date=date)
            present_qs = Present.objects.filter(date=date)

            if present_qs.exists():
                qs, chart_url = hours_vs_employee_given_date(present_qs, time_qs)
            else:
                messages.warning(request, "No records for the selected date.")
    else:
        form = DateForm()

    context = {
        "form": form,
        "qs": qs,
        "hours_vs_employee_chart_url": chart_url,
    }
    return render(request, "recognition/view_attendance_date.html", context)


@login_required
def view_attendance_employee(request):
    """
    Admin view to see attendance for a specific employee over a date range.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    qs: Optional[QuerySet[Present]] = None
    chart_url: Optional[str] = None
    if request.method == "POST":
        form = UsernameAndDateForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            date_from = form.cleaned_data["date_from"]
            date_to = form.cleaned_data["date_to"]

            if date_to < date_from:
                messages.warning(
                    request,
                    "Invalid date selection: 'To' date cannot be before 'From' date.",
                )
                return redirect("view-attendance-employee")

            user = User.objects.filter(username=username).first()
            if user:
                time_qs = Time.objects.filter(
                    user=user, date__gte=date_from, date__lte=date_to
                ).order_by("-date")
                present_qs = Present.objects.filter(
                    user=user, date__gte=date_from, date__lte=date_to
                ).order_by("-date")

                if present_qs.exists():
                    qs, chart_url = hours_vs_date_given_employee(present_qs, time_qs, admin=True)
                else:
                    messages.warning(request, "No records for the selected duration.")
            else:
                messages.warning(request, "Username not found.")
    else:
        form = UsernameAndDateForm()

    context = {
        "form": form,
        "qs": qs,
        "hours_vs_date_chart_url": chart_url,
    }
    return render(request, "recognition/view_attendance_employee.html", context)


@login_required
def view_my_attendance_employee_login(request):
    """
    Employee-specific view to see their own attendance over a date range.
    """
    if request.user.is_staff or request.user.is_superuser:
        return redirect("not-authorised")

    qs: Optional[QuerySet[Present]] = None
    chart_url: Optional[str] = None
    if request.method == "POST":
        form = DateForm_2(request.POST)
        if form.is_valid():
            user = request.user
            date_from = form.cleaned_data["date_from"]
            date_to = form.cleaned_data["date_to"]

            if date_to < date_from:
                messages.warning(request, "Invalid date selection.")
                return redirect("view-my-attendance-employee-login")

            time_qs = Time.objects.filter(
                user=user, date__gte=date_from, date__lte=date_to
            ).order_by("-date")
            present_qs = Present.objects.filter(
                user=user, date__gte=date_from, date__lte=date_to
            ).order_by("-date")

            if present_qs.exists():
                qs, chart_url = hours_vs_date_given_employee(present_qs, time_qs, admin=False)
            else:
                messages.warning(request, "No records for the selected duration.")

    else:
        form = DateForm_2()

    context = {
        "form": form,
        "qs": qs,
        "hours_vs_date_chart_url": chart_url,
    }
    return render(request, "recognition/view_my_attendance_employee_login.html", context)


def _get_deepface_options() -> Dict[str, Any]:
    """Return the configured DeepFace options merged with defaults."""

    defaults: Dict[str, Any] = {
        "backend": "opencv",
        "model": "Facenet",
        "detector_backend": "ssd",
        "distance_metric": "euclidean_l2",
        "enforce_detection": False,
        "anti_spoofing": True,
    }

    configured = getattr(settings, "DEEPFACE_OPTIMIZATIONS", {})
    if not isinstance(configured, dict):
        configured = {}

    options = defaults | configured
    distance_metric = str(options.get("distance_metric", "euclidean_l2")).lower()
    options["distance_metric"] = distance_metric

    enforce_value = options.get("enforce_detection", False)
    if isinstance(enforce_value, str):
        enforce_value = enforce_value.lower() in {"1", "true", "yes", "on"}
    else:
        enforce_value = bool(enforce_value)
    options["enforce_detection"] = enforce_value

    spoof_value = options.get("anti_spoofing", True)
    if isinstance(spoof_value, str):
        spoof_value = spoof_value.lower() in {"1", "true", "yes", "on"}
    else:
        spoof_value = bool(spoof_value)
    options["anti_spoofing"] = spoof_value
    return options


def _get_face_detection_backend() -> str:
    """
    Return the configured face detection backend for DeepFace.

    Defaults to 'opencv' if not specified in Django settings.

    Returns:
        The name of the backend (e.g., 'opencv', 'ssd', 'dlib').
    """
    return str(_get_deepface_options().get("detector_backend", "opencv"))


def _get_face_recognition_model() -> str:
    """
    Return the face recognition model name configured in Django settings.

    This value is used to select the pre-trained model for face recognition.

    Returns:
        The name of the face recognition model.
    """
    return str(_get_deepface_options().get("model", "VGG-Face"))


def _should_enforce_detection() -> bool:
    """Return whether DeepFace should enforce detection failures."""

    return bool(_get_deepface_options().get("enforce_detection", False))


def _get_deepface_distance_metric() -> str:
    """Return the configured embedding distance metric."""

    return str(_get_deepface_options().get("distance_metric", "euclidean_l2"))


def _is_lightweight_liveness_enabled() -> bool:
    """Return whether the motion-based liveness gate is enabled."""

    return bool(getattr(settings, "RECOGNITION_LIGHTWEIGHT_LIVENESS_ENABLED", True))


def _get_lightweight_liveness_window() -> int:
    """Return the configured number of frames to keep for motion analysis."""

    try:
        window = int(getattr(settings, "RECOGNITION_LIVENESS_WINDOW", 5))
    except (TypeError, ValueError):
        return 5
    return max(2, window)


def _get_lightweight_liveness_min_frames() -> int:
    """Return the minimum frames required to attempt motion analysis."""

    try:
        minimum = int(getattr(settings, "RECOGNITION_LIVENESS_MIN_FRAMES", 3))
    except (TypeError, ValueError):
        return 3
    return max(2, minimum)


def _get_lightweight_liveness_threshold() -> float:
    """Return the acceptance threshold for the motion-based liveness score."""

    try:
        threshold = float(getattr(settings, "RECOGNITION_LIVENESS_MOTION_THRESHOLD", 1.1))
    except (TypeError, ValueError):
        return 1.1
    return max(0.0, threshold)


def _is_liveness_enabled() -> bool:
    """Return whether anti-spoofing checks are enabled."""

    return bool(_get_deepface_options().get("anti_spoofing", True))


def _is_enhanced_liveness_enabled() -> bool:
    """Return whether enhanced liveness (CNN + depth + consistency) is enabled."""

    return bool(getattr(settings, "RECOGNITION_ENHANCED_LIVENESS_ENABLED", False))


def _get_recognition_training_test_split_ratio() -> float:
    """
    Return the train-test split ratio for model validation.

    Defaults to 0.25 if not specified in Django settings.

    Returns:
        The proportion of the dataset to be used for testing.
    """
    return float(getattr(settings, "RECOGNITION_TRAINING_TEST_SPLIT_RATIO", 0.25))


def _get_recognition_training_seed() -> int:
    """
    Return the fixed seed for reproducible train-test splits.

    Defaults to 42 if not specified in Django settings.

    Returns:
        An integer seed for random operations.
    """
    return int(getattr(settings, "RECOGNITION_TRAINING_SEED", 42))


# ========== Views ==========


@login_required
def mark_attendance_view(request, attendance_type):
    """
    View to handle marking attendance (check-in or check-out) using the trained model.

    Args:
        request: The Django HttpRequest object.
        attendance_type: A string indicating the attendance type ('in' or 'out').
    """
    if not (request.user.is_staff or request.user.is_superuser):
        messages.error(request, "Not authorised to perform this action")
        return redirect("not_authorised")

    metadata = {
        "direction": attendance_type,
        "path": getattr(request, "path", ""),
        "user_id": getattr(request.user, "id", None),
        "username": getattr(request.user, "username", None),
        "remote_addr": request.META.get("REMOTE_ADDR") if hasattr(request, "META") else None,
    }
    _bind_request_to_sentry_scope(request, flow="svc_attendance", extra=metadata)
    _record_sentry_breadcrumb(
        message="Model-based attendance flow started",
        category="attendance.flow",
        data=metadata,
    )

    logger.info(
        "Attendance marking process ('%s') initiated by %s.",
        attendance_type,
        request.user,
        extra={
            "flow": "svc_attendance",
            "direction": attendance_type,
            "user_id": metadata["user_id"],
        },
    )

    direction_choice = (
        RecognitionAttempt.Direction.IN
        if attendance_type == "in"
        else RecognitionAttempt.Direction.OUT
    )
    attempt_logger = _RecognitionAttemptLogger(
        direction_choice.value,
        _resolve_recognition_site(request),
        source="svc",
    )

    # --- Load cached embeddings to avoid rebuilding them per frame ---
    model_name = _get_face_recognition_model()
    detector_backend = _get_face_detection_backend()
    enforce_detection = _should_enforce_detection()
    dataset_index = _load_dataset_embeddings_for_matching(
        model_name, detector_backend, enforce_detection
    )
    dataset_size: int | None
    try:
        dataset_size = len(dataset_index)
    except TypeError:
        dataset_size = None
    _record_sentry_breadcrumb(
        message="Model-based dataset loaded",
        category="attendance.dataset",
        data={
            "direction": attendance_type,
            "dataset_size": dataset_size,
        },
    )
    if not dataset_index:
        logger.warning(
            "Cached embeddings are empty while marking attendance via SVC; continuing with model predictions.",
            extra={
                "flow": "svc_attendance",
                "direction": attendance_type,
            },
        )
        attempt_logger.log_failure(
            getattr(request.user, "username", None),
            spoofed=False,
            error="Cached embeddings unavailable while marking attendance.",
        )
        _record_sentry_breadcrumb(
            message="Model-based dataset empty",
            category="attendance.dataset",
            level="warning",
            data={
                "direction": attendance_type,
            },
        )

    # --- Load Model ---
    try:
        model_path = DATA_ROOT / "svc.sav"
        classes_path = DATA_ROOT / "classes.npy"
        encrypted_model = model_path.read_bytes()
        model = pickle.loads(decrypt_bytes(encrypted_model))  # noqa: F841

        encrypted_classes = classes_path.read_bytes()
        class_names = np.load(io.BytesIO(decrypt_bytes(encrypted_classes)), allow_pickle=True)
    except FileNotFoundError:
        messages.error(
            request,
            "Model not found. Please train the model before marking attendance.",
        )
        return redirect("train")
    except Exception:
        logger.exception(
            "Failed to load the recognition model",
            extra={
                "flow": "svc_attendance",
                "direction": attendance_type,
            },
        )
        messages.error(request, "Failed to load the model. Check logs for details.")
        return redirect("train")

    # --- Video Stream ---
    manager = get_webcam_manager()
    present = {name: False for name in class_names}
    recorded_outcomes: set[tuple[str, bool, bool]] = set()
    spoof_detected = False
    headless = _is_headless_environment()
    max_frames = int(getattr(settings, "RECOGNITION_HEADLESS_ATTENDANCE_FRAMES", 100))
    frame_pause = float(getattr(settings, "RECOGNITION_HEADLESS_FRAME_SLEEP", 0.01))

    def _log_outcome(
        candidate: Optional[str],
        *,
        accepted: bool,
        distance: float | None,
        spoofed: bool = False,
    ) -> None:
        key = (candidate or "", accepted, spoofed)
        if key in recorded_outcomes:
            return
        recorded_outcomes.add(key)
        log_recognition_outcome(
            username=candidate,
            accepted=accepted,
            direction=attendance_type,
            distance=distance,
            threshold=None,
            source="svc",
        )
        if accepted and candidate:
            attempt_logger.log_success(candidate)
        else:
            error_message = (
                "Spoofing detected by liveness gate."
                if spoofed
                else "Face not recognized during SVC attendance session."
            )
            attempt_logger.log_failure(candidate, spoofed=spoofed, error=error_message)

    frame_count = 0
    liveness_buffer = LivenessBuffer(maxlen=_get_lightweight_liveness_window())
    try:
        with manager.frame_consumer() as consumer:
            while True:
                frame = consumer.read(timeout=1.0)
                if frame is None:
                    continue
                frame = imutils.resize(frame, width=800)
                liveness_buffer.append(frame)
                frame_count += 1

                try:
                    # --- Face Recognition ---
                    embeddings = DeepFace.represent(
                        img_path=frame,
                        model_name=model_name,
                        detector_backend=detector_backend,
                        enforce_detection=enforce_detection,
                    )

                    # Normalize and safely extract the first embedding + facial_area
                    embedding_vector = None
                    facial_area = None
                    if (
                        isinstance(embeddings, np.ndarray)
                        and embeddings.ndim == 2
                        and len(embeddings) > 0
                    ):
                        embedding_vector = list(embeddings[0])
                    elif isinstance(embeddings, list) and len(embeddings) > 0:
                        first = embeddings[0]
                        if isinstance(first, dict):
                            embedding_vector = first.get("embedding")
                            facial_area = first.get("facial_area")  # noqa: F841
                        elif isinstance(first, (list, tuple, np.ndarray)):
                            embedding_vector = list(first)
                        else:
                            try:
                                embedding_vector = list(first)
                            except Exception:
                                embedding_vector = None

                    predicted_name: Optional[str] = None
                    spoofed = False
                    normalized_region: Optional[Dict[str, int]] = None
                    if embedding_vector is not None:
                        predicted_name, spoofed, normalized_region = (
                            _predict_identity_from_embedding(
                                frame,
                                embedding_vector,
                                facial_area if isinstance(facial_area, dict) else None,
                                model,
                                class_names,
                                attendance_type,
                                frame_history=liveness_buffer.snapshot(),
                            )
                        )

                        if spoofed:
                            spoof_detected = True
                            _log_outcome(
                                predicted_name,
                                accepted=False,
                                distance=None,
                                spoofed=True,
                            )
                            if normalized_region:
                                x = int(normalized_region["x"])
                                y = int(normalized_region["y"])
                                w = int(normalized_region["w"])
                                h = int(normalized_region["h"])
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                cv2.putText(
                                    frame,
                                    "Spoof detected",
                                    (x, max(y - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2,
                                )
                            continue

                        if predicted_name:
                            present[predicted_name] = True
                            _log_outcome(predicted_name, accepted=True, distance=None)

                            if normalized_region:
                                x = int(normalized_region["x"])
                                y = int(normalized_region["y"])
                                w = int(normalized_region["w"])
                                h = int(normalized_region["h"])
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(
                                    frame,
                                    predicted_name,
                                    (x, max(y - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                )
                        else:
                            _log_outcome(None, accepted=False, distance=None)

                except Exception:
                    logger.warning(
                        "Could not process frame for recognition",
                        exc_info=True,
                        extra={
                            "flow": "svc_attendance",
                            "direction": attendance_type,
                        },
                    )

                if not headless:
                    cv2.imshow(f"Marking Attendance ({attendance_type})", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    if frame_pause:
                        time.sleep(frame_pause)

                if frame_count >= max_frames:
                    break
    finally:
        if not headless:
            cv2.destroyAllWindows()

    if spoof_detected:
        messages.error(request, LIVENESS_FAILURE_MESSAGE)

    attempt_logger.ensure_generic_failure("No faces were recognized during the attendance session.")

    recognized_present = {name: True for name, value in present.items() if value}

    if recognized_present:
        record: Dict[str, Any] = {
            "direction": attendance_type,
            "present": recognized_present,
        }
        attempt_ids = {
            username: attempt_id
            for username, attempt_id in attempt_logger.success_attempt_ids.items()
            if recognized_present.get(username)
        }
        if attempt_ids:
            record["attempt_ids"] = attempt_ids
        try:
            async_result = _enqueue_attendance_records([record])
            logger.info(
                "Queued attendance batch %s for %s event via API.",
                async_result.id,
                attendance_type,
                extra={
                    "flow": "svc_attendance",
                    "direction": attendance_type,
                    "task_id": async_result.id,
                    "recognized_users": sorted(recognized_present.keys()),
                },
            )
            _record_sentry_breadcrumb(
                message="API attendance batch enqueued",
                category="attendance.queue",
                data={
                    "direction": attendance_type,
                    "task_id": async_result.id,
                    "user_count": len(recognized_present),
                },
            )
            if attendance_type == "in":
                messages.success(
                    request,
                    "Checked-in users are being processed in the background.",
                )
            elif attendance_type == "out":
                messages.success(
                    request,
                    "Checked-out users are being processed in the background.",
                )
        except Exception:  # pragma: no cover - defensive programming
            logger.exception(
                "Failed to enqueue attendance processing via API; applying updates synchronously.",
                extra={
                    "flow": "svc_attendance",
                    "direction": attendance_type,
                },
            )
            _record_sentry_breadcrumb(
                message="API attendance batch enqueue failed",
                category="attendance.queue",
                level="error",
                data={
                    "direction": attendance_type,
                    "user_count": len(recognized_present),
                },
            )
            if attendance_type == "in":
                update_attendance_in_db_in(recognized_present, attempt_ids=attempt_ids)
                messages.success(request, "Checked-in users have been marked present.")
            elif attendance_type == "out":
                update_attendance_in_db_out(recognized_present, attempt_ids=attempt_ids)
                messages.success(request, "Checked-out users have been marked.")
    else:
        messages.info(request, "No recognized users to update attendance for.")

    return redirect("home")
