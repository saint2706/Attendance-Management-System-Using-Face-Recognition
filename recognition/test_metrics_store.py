"""Tests for the lightweight recognition metrics store."""

from __future__ import annotations

from datetime import timedelta

from django.test import TestCase, override_settings
from django.utils import timezone

from recognition.metrics_store import log_recognition_outcome
from recognition.models import RecognitionOutcome


class MetricsStoreTests(TestCase):
    """Validate persistence and retention behaviour for recognition outcomes."""

    def test_log_outcome_persists_record(self) -> None:
        """A recognition outcome should be persisted with normalized confidence."""

        log_recognition_outcome(
            username="alice",
            accepted=True,
            direction="in",
            distance=0.2,
            threshold=0.5,
            source="webcam",
        )

        outcome = RecognitionOutcome.objects.get()
        self.assertEqual(outcome.username, "alice")
        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.direction, "in")
        self.assertAlmostEqual(outcome.confidence, 0.6)

    @override_settings(RECOGNITION_OUTCOME_RETENTION_DAYS=1)
    def test_retention_prunes_old_records(self) -> None:
        """Outcomes older than the retention window should be removed."""

        old = RecognitionOutcome.objects.create(
            username="bob",
            accepted=False,
            direction="out",
            confidence=0.1,
        )
        RecognitionOutcome.objects.filter(pk=old.pk).update(
            created_at=timezone.now() - timedelta(days=3)
        )

        log_recognition_outcome(
            username="carol",
            accepted=True,
            direction="in",
            distance=0.1,
            threshold=0.5,
            source="api",
        )

        self.assertFalse(RecognitionOutcome.objects.filter(pk=old.pk).exists())
        self.assertEqual(RecognitionOutcome.objects.count(), 1)
