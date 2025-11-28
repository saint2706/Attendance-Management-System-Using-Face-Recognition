from __future__ import annotations

"""Helpers for surfacing system health and artifact freshness to admins."""

import datetime as dt
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from django.conf import settings
from django.utils import timezone

from attendance_system_facial_recognition.celery import app as celery_app
from recognition.models import RecognitionOutcome
from users.models import RecognitionAttempt

logger = logging.getLogger(__name__)

DATA_ROOT = Path(settings.BASE_DIR) / "face_recognition_data"
TRAINING_DATASET_ROOT = DATA_ROOT / "training_dataset"
MODEL_PATH = DATA_ROOT / "svc.sav"
CLASSES_PATH = DATA_ROOT / "classes.npy"
REPORT_PATH = DATA_ROOT / "classification_report.txt"


def _safe_mtime(path: Path) -> Optional[dt.datetime]:
    """Return the path modification time as a timezone-aware datetime."""

    try:
        timestamp = path.stat().st_mtime
    except FileNotFoundError:
        return None
    except OSError:  # pragma: no cover - defensive guard
        logger.debug("Unable to stat %s", path, exc_info=True)
        return None

    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)


def _isoformat_or_none(value: Optional[dt.datetime]) -> Optional[str]:
    """Return an ISO-8601 string for the datetime if provided."""

    if value is None:
        return None
    if timezone.is_naive(value):
        value = timezone.make_aware(value, timezone=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat()


def dataset_health() -> Dict[str, Any]:
    """Summarize the encrypted training dataset on disk."""

    dataset_files = [path for path in TRAINING_DATASET_ROOT.glob("*/*.jpg") if path.is_file()]
    identities = {path.parent.name for path in dataset_files}
    last_updated = max((_safe_mtime(path) for path in dataset_files), default=None)

    return {
        "exists": TRAINING_DATASET_ROOT.exists(),
        "image_count": len(dataset_files),
        "identity_count": len(identities),
        "last_updated": last_updated,
        "last_updated_display": _isoformat_or_none(last_updated),
    }


def model_health(*, dataset_last_updated: Optional[dt.datetime]) -> Dict[str, Any]:
    """Report the presence and staleness of trained model artifacts."""

    model_mtime = _safe_mtime(MODEL_PATH)
    classes_mtime = _safe_mtime(CLASSES_PATH)
    report_mtime = _safe_mtime(REPORT_PATH)
    artifact_times = [ts for ts in (model_mtime, classes_mtime, report_mtime) if ts]
    last_trained = max(artifact_times) if artifact_times else None

    stale = bool(dataset_last_updated and last_trained and dataset_last_updated > last_trained)

    return {
        "model_present": MODEL_PATH.exists(),
        "classes_present": CLASSES_PATH.exists(),
        "report_present": REPORT_PATH.exists(),
        "last_trained": last_trained,
        "last_trained_display": _isoformat_or_none(last_trained),
        "stale": stale,
    }


def recognition_activity() -> Dict[str, Any]:
    """Provide the most recent recognition attempt and outcome snapshots."""

    def _serialize_attempt(attempt: RecognitionAttempt | None) -> Optional[Dict[str, Any]]:
        if attempt is None:
            return None
        return {
            "username": attempt.username or (attempt.user.username if attempt.user else ""),
            "direction": attempt.get_direction_display() if hasattr(attempt, "get_direction_display") else attempt.direction,
            "timestamp": _isoformat_or_none(attempt.created_at),
            "successful": attempt.successful,
            "spoof_detected": attempt.spoof_detected,
            "error": attempt.error_message,
        }

    def _serialize_outcome(outcome: RecognitionOutcome | None) -> Optional[Dict[str, Any]]:
        if outcome is None:
            return None
        return {
            "username": outcome.username,
            "direction": outcome.direction,
            "timestamp": _isoformat_or_none(outcome.created_at),
            "accepted": outcome.accepted,
            "distance": outcome.distance,
            "threshold": outcome.threshold,
            "confidence": outcome.confidence,
        }

    last_attempt = RecognitionAttempt.objects.order_by("-created_at").first()
    last_spoof = (
        RecognitionAttempt.objects.filter(spoof_detected=True).order_by("-created_at").first()
    )
    last_success = (
        RecognitionAttempt.objects.filter(successful=True).order_by("-created_at").first()
    )
    last_failure = (
        RecognitionAttempt.objects.filter(successful=False).order_by("-created_at").first()
    )
    last_outcome = RecognitionOutcome.objects.order_by("-created_at").first()

    return {
        "last_attempt": _serialize_attempt(last_attempt),
        "last_spoof": _serialize_attempt(last_spoof),
        "last_success": _serialize_attempt(last_success),
        "last_failure": _serialize_attempt(last_failure),
        "last_outcome": _serialize_outcome(last_outcome),
    }


def worker_health() -> Dict[str, Any]:
    """Ping the Celery worker to surface reachability in the dashboard."""

    broker_configured = bool(getattr(settings, "CELERY_BROKER_URL", None))
    if not broker_configured:
        return {"status": "not-configured", "workers": 0}

    try:
        responses = celery_app.control.ping(timeout=0.5)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Celery ping failed: %s", exc)
        return {"status": "unreachable", "workers": 0, "error": str(exc)}

    return {"status": "online" if responses else "unreachable", "workers": len(responses)}


__all__ = [
    "dataset_health",
    "model_health",
    "recognition_activity",
    "worker_health",
]
