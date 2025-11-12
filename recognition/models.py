"""Database models for the recognition app."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from django.conf import settings
from django.db import models
from django.utils import timezone

logger = logging.getLogger(__name__)


class RecognitionOutcomeQuerySet(models.QuerySet["RecognitionOutcome"]):
    """Custom queryset with helpers for outcome analytics."""

    def accepted(self) -> "RecognitionOutcomeQuerySet":
        """Return only accepted recognition outcomes."""

        return self.filter(accepted=True)

    def rejected(self) -> "RecognitionOutcomeQuerySet":
        """Return only rejected recognition outcomes."""

        return self.filter(accepted=False)


class RecognitionOutcome(models.Model):
    """Persisted snapshot of a recognition decision made during attendance flows."""

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    username = models.CharField(max_length=150, blank=True)
    direction = models.CharField(max_length=12, blank=True)
    source = models.CharField(max_length=32, blank=True)
    accepted = models.BooleanField()
    confidence = models.FloatField(null=True, blank=True)
    distance = models.FloatField(null=True, blank=True)
    threshold = models.FloatField(null=True, blank=True)

    objects: RecognitionOutcomeQuerySet = RecognitionOutcomeQuerySet.as_manager()

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["created_at", "direction"]),
            models.Index(fields=["accepted", "created_at"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - human readability
        status = "accepted" if self.accepted else "rejected"
        username = self.username or "unknown"
        return f"{username} {status} @ {self.created_at:%Y-%m-%d %H:%M:%S}"

    @classmethod
    def prune_expired(cls) -> None:
        """Delete outcomes older than the configured retention window."""

        retention_days: Optional[int] = getattr(settings, "RECOGNITION_OUTCOME_RETENTION_DAYS", 30)
        if retention_days in (None, "none", ""):
            return

        try:
            days = int(retention_days)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            logger.debug(
                "Invalid RECOGNITION_OUTCOME_RETENTION_DAYS=%r; skipping prune.",
                retention_days,
            )
            return

        if days <= 0:
            logger.debug("Retention set to %s days; skipping prune.", days)
            return

        cutoff = timezone.now() - timedelta(days=days)
        cls.objects.filter(created_at__lt=cutoff).delete()
