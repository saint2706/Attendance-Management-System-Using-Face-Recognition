"""Helpers for persisting lightweight recognition analytics."""

from __future__ import annotations

import logging
from typing import Optional

from django.db import transaction

from .models import LivenessResult, RecognitionOutcome

logger = logging.getLogger(__name__)


def _normalize_confidence(distance: Optional[float], threshold: Optional[float]) -> Optional[float]:
    """Return a bounded confidence value derived from a distance/threshold pair."""

    if distance is None or threshold in (None, 0):
        return None

    try:
        ratio = max(0.0, min(float(distance) / float(threshold), 1.0))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None

    return round(1.0 - ratio, 6)


def log_recognition_outcome(
    *,
    username: str | None,
    accepted: bool,
    direction: str | None,
    distance: float | None = None,
    threshold: float | None = None,
    source: str | None = None,
    liveness_confidence: float | None = None,
    liveness_passed: bool | None = None,
    profile_name: str | None = None,
) -> None:
    """Persist a recognition outcome without impacting the calling flow."""

    confidence = _normalize_confidence(distance, threshold)

    try:
        with transaction.atomic():
            RecognitionOutcome.objects.create(
                username=username or "",
                accepted=bool(accepted),
                direction=direction or "",
                distance=distance,
                threshold=threshold,
                confidence=confidence,
                source=source or "",
                liveness_confidence=liveness_confidence,
                liveness_passed=liveness_passed,
                profile_name=profile_name or "",
            )
    except Exception:  # pragma: no cover - defensive persistence
        logger.debug("Unable to persist recognition outcome", exc_info=True)
        return

    try:
        RecognitionOutcome.prune_expired()
    except Exception:  # pragma: no cover - defensive retention
        logger.debug("Failed to apply outcome retention policy", exc_info=True)


def log_liveness_result(
    *,
    username: str | None = None,
    site: str | None = None,
    source: str | None = None,
    challenge_type: str = "motion",
    challenge_status: str = "passed",
    liveness_confidence: float | None = None,
    motion_score: float | None = None,
    threshold_used: float | None = None,
    frames_analyzed: int = 0,
) -> None:
    """Persist a liveness check result for auditing and analytics."""

    try:
        with transaction.atomic():
            LivenessResult.objects.create(
                username=username or "",
                site=site or "",
                source=source or "",
                challenge_type=challenge_type,
                challenge_status=challenge_status,
                liveness_confidence=liveness_confidence,
                motion_score=motion_score,
                threshold_used=threshold_used,
                frames_analyzed=frames_analyzed,
            )
    except Exception:  # pragma: no cover - defensive persistence
        logger.debug("Unable to persist liveness result", exc_info=True)


__all__ = ["log_recognition_outcome", "log_liveness_result"]
