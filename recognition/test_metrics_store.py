"""Tests for the lightweight recognition metrics store."""

from __future__ import annotations

from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import Client, TestCase, override_settings
from django.urls import reverse
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


class RecognitionTrendViewTests(TestCase):
    """Ensure the admin dashboard aggregates accuracy information."""

    def setUp(self) -> None:
        self.client = Client()
        self.staff_user = get_user_model().objects.create_user(
            username="staff",
            email="staff@example.com",
            password="password",
            is_staff=True,
        )

    def test_trends_view_returns_aggregated_data(self) -> None:
        """The trend view should expose aggregated accuracy data."""

        base_time = timezone.now().replace(hour=10, minute=0, second=0, microsecond=0)
        outcome = RecognitionOutcome.objects.create(
            username="alice",
            accepted=True,
            direction="in",
            confidence=0.8,
        )
        RecognitionOutcome.objects.filter(pk=outcome.pk).update(created_at=base_time)

        outcome = RecognitionOutcome.objects.create(
            username="bob",
            accepted=False,
            direction="in",
            confidence=0.2,
        )
        RecognitionOutcome.objects.filter(pk=outcome.pk).update(created_at=base_time)

        self.client.force_login(self.staff_user)
        response = self.client.get(reverse("admin_recognition_trends"))

        self.assertEqual(response.status_code, 200)
        daily = response.context["daily_trend"]
        self.assertTrue(daily)
        day_entry = daily[0]
        self.assertEqual(day_entry["total"], 2)
        self.assertEqual(day_entry["accepted"], 1)
        self.assertEqual(day_entry["rejected"], 1)
        self.assertIn("weekly_trend", response.context)
