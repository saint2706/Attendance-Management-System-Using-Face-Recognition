"""Helpers for persisting lightweight recognition analytics."""

from __future__ import annotations

import logging
from typing import Optional

from django.db import transaction

from .models import RecognitionOutcome

logger = logging.getLogger(__name__)


def _normalize_confidence(
    distance: Optional[float], threshold: Optional[float]
) -> Optional[float]:
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
            )
    except Exception:  # pragma: no cover - defensive persistence
        logger.debug("Unable to persist recognition outcome", exc_info=True)
        return

    try:
        RecognitionOutcome.prune_expired()
    except Exception:  # pragma: no cover - defensive retention
        logger.debug("Failed to apply outcome retention policy", exc_info=True)


__all__ = ["log_recognition_outcome"]
