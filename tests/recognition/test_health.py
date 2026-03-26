import datetime as dt
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import override_settings
from django.utils import timezone

import pytest

from recognition.health import (
    dataset_health,
    evaluation_health,
    model_health,
    recognition_activity,
    worker_health,
)
from recognition.models import ModelEvaluationResult, RecognitionOutcome
from users.models import RecognitionAttempt

User = get_user_model()


class TestDatasetHealth:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        cache.clear()

    @mock.patch("recognition.health.TRAINING_DATASET_ROOT")
    def test_dataset_health_cached(self, mock_root):
        mock_root.exists.return_value = True
        mock_root.glob.return_value = []
        cache.set(
            "recognition:health:dataset_snapshot",
            {"exists": True, "image_count": 0, "identity_count": 0},
        )
        snapshot = dataset_health()
        assert snapshot["exists"] is True
        mock_root.glob.assert_not_called()

    @mock.patch("recognition.health._safe_mtime")
    @mock.patch("recognition.health.TRAINING_DATASET_ROOT")
    def test_dataset_health_empty(self, mock_root, mock_mtime):
        mock_root.exists.return_value = True
        mock_root.glob.return_value = []
        snapshot = dataset_health()
        assert snapshot["exists"] is True
        assert snapshot["image_count"] == 0
        assert snapshot["identity_count"] == 0
        assert snapshot["last_updated"] is None
        assert snapshot["last_updated_display"] is None

    @mock.patch("recognition.health._safe_mtime")
    @mock.patch("recognition.health.TRAINING_DATASET_ROOT")
    def test_dataset_health_with_files(self, mock_root, mock_mtime):
        mock_root.exists.return_value = True

        file1 = mock.Mock()
        file1.is_file.return_value = True
        file1.parent.name = "user1"

        file2 = mock.Mock()
        file2.is_file.return_value = True
        file2.parent.name = "user2"

        mock_root.glob.return_value = [file1, file2]

        dt_now = timezone.now()
        mock_mtime.side_effect = [dt_now, dt_now - dt.timedelta(hours=1)]

        snapshot = dataset_health()

        assert snapshot["exists"] is True
        assert snapshot["image_count"] == 2
        assert snapshot["identity_count"] == 2
        assert snapshot["last_updated"] == dt_now
        assert snapshot["last_updated_display"] == dt_now.isoformat()


class TestModelHealth:
    @mock.patch("recognition.health._safe_mtime")
    @mock.patch("recognition.health.REPORT_PATH")
    @mock.patch("recognition.health.CLASSES_PATH")
    @mock.patch("recognition.health.MODEL_PATH")
    def test_model_health_all_missing(self, mock_model, mock_classes, mock_report, mock_mtime):
        mock_model.exists.return_value = False
        mock_classes.exists.return_value = False
        mock_report.exists.return_value = False
        mock_mtime.return_value = None

        health = model_health(dataset_last_updated=None)

        assert health["model_present"] is False
        assert health["classes_present"] is False
        assert health["report_present"] is False
        assert health["last_trained"] is None
        assert health["last_trained_display"] is None
        assert health["stale"] is False

    @mock.patch("recognition.health._safe_mtime")
    @mock.patch("recognition.health.REPORT_PATH")
    @mock.patch("recognition.health.CLASSES_PATH")
    @mock.patch("recognition.health.MODEL_PATH")
    def test_model_health_all_present(self, mock_model, mock_classes, mock_report, mock_mtime):
        mock_model.exists.return_value = True
        mock_classes.exists.return_value = True
        mock_report.exists.return_value = True

        dt_trained = timezone.now() - dt.timedelta(hours=2)
        mock_mtime.return_value = dt_trained

        health = model_health(dataset_last_updated=None)

        assert health["model_present"] is True
        assert health["classes_present"] is True
        assert health["report_present"] is True
        assert health["last_trained"] == dt_trained
        assert health["last_trained_display"] == dt_trained.isoformat()
        assert health["stale"] is False

    @mock.patch("recognition.health._safe_mtime")
    @mock.patch("recognition.health.REPORT_PATH")
    @mock.patch("recognition.health.CLASSES_PATH")
    @mock.patch("recognition.health.MODEL_PATH")
    def test_model_health_stale(self, mock_model, mock_classes, mock_report, mock_mtime):
        mock_model.exists.return_value = True
        mock_classes.exists.return_value = True
        mock_report.exists.return_value = True

        dt_trained = timezone.now() - dt.timedelta(hours=2)
        mock_mtime.return_value = dt_trained

        dt_dataset = timezone.now()

        health = model_health(dataset_last_updated=dt_dataset)

        assert health["stale"] is True


@pytest.mark.django_db
class TestEvaluationHealth:
    def test_evaluation_health_empty(self):
        health = evaluation_health()
        assert health["latest_evaluation"] is None
        assert health["latest_nightly"] is None
        assert health["latest_fairness"] is None
        assert health["latest_liveness"] is None
        assert health["trends"] == {}
        assert health["total_evaluations"] == 0
        assert health["recent_failures"] == 0

    def test_evaluation_health_with_data(self):
        # Create some evaluations
        ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            far=0.01,
            frr=0.02,
            samples_evaluated=100,
            identities_evaluated=10,
        )

        ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.FAIRNESS_AUDIT,
            success=True,  # MUST be True for get_latest to return it
        )

        ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.LIVENESS_EVAL,
            success=True,
            accuracy=0.99,
        )

        health = evaluation_health()

        assert health["latest_nightly"] is not None
        assert health["latest_nightly"]["success"] is True
        assert health["latest_nightly"]["accuracy"] == 0.95

        assert health["latest_fairness"] is not None
        assert health["latest_fairness"]["success"] is True

        assert health["latest_liveness"] is not None
        assert health["latest_liveness"]["success"] is True

        assert health["latest_evaluation"] is not None
        assert health["latest_evaluation"]["success"] is True

        assert health["total_evaluations"] == 3


@pytest.mark.django_db
class TestRecognitionActivity:
    def test_recognition_activity_empty(self):
        activity = recognition_activity()
        assert activity["last_attempt"] is None
        assert activity["last_spoof"] is None
        assert activity["last_success"] is None
        assert activity["last_failure"] is None
        assert activity["last_outcome"] is None

    def test_recognition_activity_with_data(self):
        user = User.objects.create(username="testuser")

        # Create a successful attempt
        RecognitionAttempt.objects.create(
            user=user,
            successful=True,
            spoof_detected=False,
            direction="in",
        )

        # Create a spoof attempt
        RecognitionAttempt.objects.create(
            username="unknown",
            successful=False,
            spoof_detected=True,
            direction="in",
        )

        # Create an outcome
        RecognitionOutcome.objects.create(
            username="testuser",
            direction="in",
            accepted=True,
            distance=0.2,
            threshold=0.4,
            confidence=0.9,
        )

        activity = recognition_activity()

        assert activity["last_attempt"] is not None
        assert activity["last_attempt"]["spoof_detected"] is True
        assert activity["last_attempt"]["username"] == "unknown"

        assert activity["last_success"] is not None
        assert activity["last_success"]["username"] == "testuser"

        assert activity["last_spoof"] is not None
        assert activity["last_spoof"]["username"] == "unknown"

        assert activity["last_failure"] is not None
        assert activity["last_failure"]["username"] == "unknown"

        assert activity["last_outcome"] is not None
        assert activity["last_outcome"]["username"] == "testuser"
        assert activity["last_outcome"]["accepted"] is True


class TestWorkerHealth:
    @override_settings(CELERY_BROKER_URL=None)
    def test_worker_health_not_configured(self):
        health = worker_health()
        assert health["status"] == "not-configured"
        assert health["workers"] == 0

    @override_settings(CELERY_BROKER_URL="redis://localhost:6379/0")
    @mock.patch("recognition.health.celery_app")
    def test_worker_health_online(self, mock_celery):
        mock_celery.control.ping.return_value = [{"celery@worker1": {"ok": "pong"}}]

        health = worker_health()
        assert health["status"] == "online"
        assert health["workers"] == 1

    @override_settings(CELERY_BROKER_URL="redis://localhost:6379/0")
    @mock.patch("recognition.health.celery_app")
    def test_worker_health_offline(self, mock_celery):
        mock_celery.control.ping.return_value = []

        health = worker_health()
        assert health["status"] == "unreachable"
        assert health["workers"] == 0


class TestSafeMtime:
    def test_safe_mtime_file_not_found(self):
        from recognition.health import _safe_mtime

        path = mock.Mock()
        path.stat.side_effect = FileNotFoundError
        assert _safe_mtime(path) is None

    def test_safe_mtime_success(self):
        from unittest import mock

        from recognition.health import _safe_mtime

        path = mock.Mock()
        mock_stat = mock.Mock()
        mock_stat.st_mtime = 1600000000.0
        path.stat.return_value = mock_stat

        result = _safe_mtime(path)
        assert result is not None
        assert result.timestamp() == 1600000000.0


class TestIsoformatOrNone:
    def test_isoformat_or_none_naive(self):
        from recognition.health import _isoformat_or_none

        dt_naive = dt.datetime(2023, 1, 1, 12, 0, 0)
        iso = _isoformat_or_none(dt_naive)
        assert iso == "2023-01-01T12:00:00+00:00"

    def test_isoformat_or_none_none(self):
        from recognition.health import _isoformat_or_none

        iso = _isoformat_or_none(None)
        assert iso is None
