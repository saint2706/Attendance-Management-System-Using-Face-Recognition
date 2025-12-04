"""Tests for scheduled evaluation tasks and model health tracking."""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pytest

from recognition import health
from recognition.models import LivenessResult, ModelEvaluationResult

pytestmark = pytest.mark.django_db


class TestModelEvaluationResult:
    """Tests for the ModelEvaluationResult model."""

    def test_create_evaluation_result(self):
        """Should create an evaluation result with metrics."""
        result = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.95,
            precision=0.94,
            recall=0.93,
            f1_score=0.935,
            far=0.02,
            frr=0.05,
            samples_evaluated=100,
            threshold_used=0.4,
            identities_evaluated=10,
            success=True,
        )

        assert result.id is not None
        assert result.accuracy == 0.95
        assert result.evaluation_type == "nightly"
        assert result.success is True

    def test_get_latest_returns_most_recent(self):
        """Should return the most recent evaluation result."""
        # Create older result
        older = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.90,
            success=True,
        )
        # Ensure the older result is strictly older
        from django.utils import timezone
        older.created_at = timezone.now() - dt.timedelta(seconds=1)
        older.save()

        # Create newer result
        newer = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.95,
            success=True,
        )

        latest = ModelEvaluationResult.get_latest(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY
        )
        assert latest is not None
        assert latest.id == newer.id
        assert latest.accuracy == 0.95

    def test_get_latest_filters_by_type(self):
        """Should filter by evaluation type when specified."""
        ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.90,
            success=True,
        )

        fairness = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.FAIRNESS_AUDIT,
            accuracy=0.92,
            success=True,
        )

        latest = ModelEvaluationResult.get_latest(
            evaluation_type=ModelEvaluationResult.EvaluationType.FAIRNESS_AUDIT
        )
        assert latest is not None
        assert latest.id == fairness.id

    def test_get_latest_excludes_failures_by_default(self):
        """Should exclude failed evaluations by default."""
        successful = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.90,
            success=True,
        )

        # Create a failed evaluation (more recent)
        ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=False,
            error_message="Test error",
        )

        latest = ModelEvaluationResult.get_latest()
        assert latest is not None
        assert latest.id == successful.id

    def test_compute_trend_with_no_previous(self):
        """Should return no trend data when there's no previous evaluation."""
        result = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.95,
            success=True,
        )

        trend = result.compute_trend()
        assert trend["has_previous"] is False
        assert trend["trends"] == {}

    def test_compute_trend_with_previous(self):
        """Should compute trends correctly when previous evaluation exists."""
        # Create older result
        older = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.90,
            precision=0.89,
            f1_score=0.88,
            far=0.05,
            frr=0.10,
            success=True,
        )
        # Ensure the older result is strictly older
        from django.utils import timezone
        older.created_at = timezone.now() - dt.timedelta(seconds=1)
        older.save()

        # Create newer result with improved metrics
        newer = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.95,
            precision=0.94,
            f1_score=0.93,
            far=0.03,  # Lower is better for FAR
            frr=0.08,  # Lower is better for FRR
            success=True,
        )

        trend = newer.compute_trend()
        assert trend["has_previous"] is True
        assert "accuracy" in trend["trends"]
        assert trend["trends"]["accuracy"]["direction"] == "improved"
        assert trend["trends"]["far"]["direction"] == "improved"
        assert trend["trends"]["frr"]["direction"] == "improved"

    def test_compute_trend_degraded_metrics(self):
        """Should detect degraded metrics correctly."""
        # Create older result with good metrics
        older = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.95,
            far=0.02,
            success=True,
        )
        # Ensure the older result is strictly older
        from django.utils import timezone
        older.created_at = timezone.now() - dt.timedelta(seconds=1)
        older.save()

        # Create newer result with worse metrics
        newer = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.85,
            far=0.08,
            success=True,
        )

        trend = newer.compute_trend()
        assert trend["trends"]["accuracy"]["direction"] == "degraded"
        assert trend["trends"]["far"]["direction"] == "degraded"

    def test_string_representation(self):
        """Should have a readable string representation."""
        result = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
        )

        str_repr = str(result)
        assert "✓" in str_repr
        assert "Scheduled Nightly" in str_repr


class TestEvaluationHealth:
    """Tests for the evaluation_health function."""

    def test_evaluation_health_no_evaluations(self):
        """Should handle the case when no evaluations exist."""
        result = health.evaluation_health()

        assert result["latest_evaluation"] is None
        assert result["latest_nightly"] is None
        assert result["latest_fairness"] is None
        assert result["latest_liveness"] is None
        assert result["total_evaluations"] == 0
        assert result["trends"] == {}

    def test_evaluation_health_with_evaluations(self):
        """Should return evaluation data when evaluations exist."""
        ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.95,
            f1_score=0.93,
            far=0.02,
            frr=0.05,
            samples_evaluated=100,
            success=True,
        )

        result = health.evaluation_health()

        assert result["latest_evaluation"] is not None
        assert result["latest_evaluation"]["accuracy"] == 0.95
        assert result["latest_nightly"] is not None
        assert result["total_evaluations"] == 1

    def test_evaluation_health_counts_recent_failures(self):
        """Should count recent failures correctly."""
        # Create a recent failure
        ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=False,
            error_message="Test failure",
        )

        result = health.evaluation_health()
        assert result["recent_failures"] == 1

    def test_evaluation_health_scheduled_tasks_status(self, settings):
        """Should report scheduled tasks status."""
        settings.CELERY_BEAT_ENABLED = True
        settings.CELERY_BEAT_SCHEDULE = {
            "task1": {},
            "task2": {},
        }

        result = health.evaluation_health()
        assert result["scheduled_tasks_enabled"] is True
        assert result["scheduled_tasks_count"] == 2


@pytest.mark.slow
@pytest.mark.integration
class TestScheduledTasks:
    """Tests for the scheduled Celery tasks."""

    @patch("recognition.scheduled_tasks._run_face_recognition_evaluation")
    def test_run_scheduled_evaluation_success(self, mock_run_eval, settings):
        """Should create evaluation result on successful run."""
        settings.CELERY_TASK_ALWAYS_EAGER = True

        # Create a mock result
        mock_result = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.95,
            precision=0.94,
            recall=0.93,
            f1_score=0.935,
            far=0.02,
            frr=0.05,
            samples_evaluated=100,
            threshold_used=0.4,
            identities_evaluated=2,
            task_id="test-task-123",
            duration_seconds=5.0,
            success=True,
        )
        mock_run_eval.return_value = mock_result

        from recognition.scheduled_tasks import run_scheduled_evaluation

        result = run_scheduled_evaluation.apply(kwargs={"evaluation_type": "nightly"}).get()

        assert result["success"] is True
        assert result["accuracy"] == 0.95
        assert result["evaluation_type"] == "nightly"

    def test_run_liveness_evaluation_no_data(self, settings):
        """Should handle case with no liveness data."""
        settings.CELERY_TASK_ALWAYS_EAGER = True

        from recognition.scheduled_tasks import run_liveness_evaluation

        result = run_liveness_evaluation.apply(kwargs={"days_back": 7}).get()

        assert result["success"] is True
        assert result["liveness_samples"] == 0
        assert result["liveness_pass_rate"] is None

    def test_run_liveness_evaluation_with_data(self, settings):
        """Should compute liveness metrics correctly."""
        settings.CELERY_TASK_ALWAYS_EAGER = True

        # Create some liveness results
        LivenessResult.objects.create(
            challenge_type="motion",
            challenge_status="passed",
            liveness_confidence=0.95,
        )
        LivenessResult.objects.create(
            challenge_type="motion",
            challenge_status="passed",
            liveness_confidence=0.90,
        )
        LivenessResult.objects.create(
            challenge_type="motion",
            challenge_status="failed",
            liveness_confidence=0.30,
        )

        from recognition.scheduled_tasks import run_liveness_evaluation

        result = run_liveness_evaluation.apply(kwargs={"days_back": 7}).get()

        assert result["success"] is True
        assert result["liveness_samples"] == 3
        # 2 passed out of 3 = ~0.667
        assert 0.6 < result["liveness_pass_rate"] < 0.7

    @pytest.mark.parametrize("days_back", [0, -1, "7", 3.5])
    def test_run_liveness_evaluation_invalid_window(self, settings, days_back):
        """Should return a clear error when the lookback window is invalid."""
        settings.CELERY_TASK_ALWAYS_EAGER = True

        from recognition.scheduled_tasks import run_liveness_evaluation

        result = run_liveness_evaluation.apply(kwargs={"days_back": days_back}).get()

        assert result["success"] is False
        assert "positive integer" in result["error_message"]


class TestSystemHealthDashboardWithEvaluation:
    """Tests for the system health dashboard with evaluation state."""

    def test_dashboard_includes_evaluation_state(self, client, django_user_model, monkeypatch):
        """The system health dashboard should include evaluation state."""
        from recognition import monitoring

        admin = django_user_model.objects.create_user(
            username="admin",
            password="AdminPass!234",
            is_staff=True,
        )
        client.force_login(admin)

        # Mock health functions
        monkeypatch.setattr(
            health,
            "dataset_health",
            lambda: {
                "exists": True,
                "image_count": 2,
                "identity_count": 1,
                "last_updated": dt.datetime.now(tz=dt.timezone.utc),
                "last_updated_display": "2024-01-01T00:00:00Z",
            },
        )
        monkeypatch.setattr(
            health,
            "model_health",
            lambda dataset_last_updated=None: {
                "model_present": True,
                "classes_present": True,
                "report_present": True,
                "last_trained": dt.datetime.now(tz=dt.timezone.utc),
                "last_trained_display": "2024-01-01T00:00:00Z",
                "stale": False,
            },
        )
        monkeypatch.setattr(
            health,
            "recognition_activity",
            lambda: {
                "last_attempt": None,
                "last_spoof": None,
                "last_success": None,
                "last_failure": None,
                "last_outcome": None,
            },
        )
        monkeypatch.setattr(health, "worker_health", lambda: {"status": "online", "workers": 1})
        # Mock monitoring.get_health_snapshot
        monkeypatch.setattr(
            monitoring,
            "get_health_snapshot",
            lambda: {
                "thresholds": {},
                "camera": {
                    "running": False,
                    "consumers": 0,
                    "last_start": None,
                    "last_stop": None,
                    "last_error": None,
                },
                "frames": {"last_frame_timestamp": None, "last_frame_delay": None},
                "metrics": {
                    "camera_start": {"success": 0, "failure": 0},
                    "camera_stop": {"success": 0, "failure": 0, "timeout": 0},
                    "frame_drop_total": 0,
                },
                "stages": {},
                "alerts": [],
            },
        )

        # Create an evaluation result
        ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.95,
            f1_score=0.93,
            success=True,
        )

        from django.urls import reverse

        response = client.get(reverse("admin_system_health"))

        assert response.status_code == 200
        assert "evaluation_state" in response.context
        assert response.context["evaluation_state"]["latest_evaluation"] is not None
        assert response.context["evaluation_state"]["latest_evaluation"]["accuracy"] == 0.95
