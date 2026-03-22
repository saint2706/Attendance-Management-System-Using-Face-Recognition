from datetime import datetime, timedelta
from datetime import timezone as datetime_timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.utils import timezone

import pytest

from recognition.health import (
    _isoformat_or_none,
    _safe_mtime,
    dataset_health,
    evaluation_health,
    model_health,
    recognition_activity,
    worker_health,
)
from recognition.models import ModelEvaluationResult, RecognitionOutcome
from users.models import RecognitionAttempt, User


@pytest.mark.django_db
class TestHelperFunctions:
    def test_safe_mtime_missing_file(self):
        assert _safe_mtime(Path("/nonexistent/file/path")) is None

    @patch("pathlib.Path.stat")
    def test_safe_mtime_os_error(self, mock_stat):
        mock_stat.side_effect = OSError("Access denied")
        assert _safe_mtime(Path("/some/path")) is None

    def test_isoformat_or_none(self):
        assert _isoformat_or_none(None) is None
        now = timezone.now()
        assert _isoformat_or_none(now) == now.isoformat()
        naive = datetime(2023, 1, 1)
        assert (
            _isoformat_or_none(naive)
            == timezone.make_aware(naive, datetime_timezone.utc).isoformat()
        )


@pytest.mark.django_db
class TestDatasetHealth:
    @patch("recognition.health.TRAINING_DATASET_ROOT")
    @patch("recognition.health.cache.get", return_value=None)
    @patch("recognition.health.cache.set")
    def test_empty_dataset(self, mock_cache_set, mock_cache_get, mock_root):
        mock_root.glob.return_value = []
        mock_root.exists.return_value = False
        res = dataset_health()
        assert res["exists"] is False
        assert res["image_count"] == 0
        assert res["identity_count"] == 0
        assert res["last_updated"] is None
        mock_cache_set.assert_called_once()

    @patch("recognition.health.TRAINING_DATASET_ROOT")
    @patch("recognition.health.cache.get", return_value=None)
    @patch("recognition.health._safe_mtime")
    def test_populated_dataset(self, mock_mtime, mock_cache_get, mock_root):
        mock_file1 = MagicMock()
        mock_file1.is_file.return_value = True
        mock_file1.parent.name = "user1"
        mock_file2 = MagicMock()
        mock_file2.is_file.return_value = True
        mock_file2.parent.name = "user2"
        mock_root.glob.return_value = [mock_file1, mock_file2]
        mock_root.exists.return_value = True

        mock_time = timezone.now()
        mock_mtime.return_value = mock_time

        res = dataset_health()
        assert res["exists"] is True
        assert res["image_count"] == 2
        assert res["identity_count"] == 2
        assert res["last_updated"] == mock_time

    @patch("recognition.health.cache.get")
    def test_cached_dataset(self, mock_cache_get):
        mock_cache_get.return_value = {"exists": True, "image_count": 5}
        res = dataset_health()
        assert res["image_count"] == 5


@pytest.mark.django_db
class TestModelHealth:
    @patch("recognition.health.MODEL_PATH")
    @patch("recognition.health.CLASSES_PATH")
    @patch("recognition.health.REPORT_PATH")
    def test_no_artifacts(self, mock_report, mock_classes, mock_model):
        mock_model.exists.return_value = False
        mock_classes.exists.return_value = False
        mock_report.exists.return_value = False

        with patch("recognition.health._safe_mtime", return_value=None):
            res = model_health(dataset_last_updated=None)

        assert res["model_present"] is False
        assert res["classes_present"] is False
        assert res["report_present"] is False
        assert res["last_trained"] is None
        assert res["stale"] is False

    @patch("recognition.health.MODEL_PATH")
    @patch("recognition.health.CLASSES_PATH")
    @patch("recognition.health.REPORT_PATH")
    def test_all_artifacts_present(self, mock_report, mock_classes, mock_model):
        mock_model.exists.return_value = True
        mock_classes.exists.return_value = True
        mock_report.exists.return_value = True

        train_time = timezone.now()
        with patch("recognition.health._safe_mtime", return_value=train_time):
            res = model_health(dataset_last_updated=train_time - timedelta(days=1))

        assert res["model_present"] is True
        assert res["classes_present"] is True
        assert res["report_present"] is True
        assert res["last_trained"] == train_time
        assert res["stale"] is False

    @patch("recognition.health.MODEL_PATH")
    @patch("recognition.health.CLASSES_PATH")
    @patch("recognition.health.REPORT_PATH")
    def test_stale_model(self, mock_report, mock_classes, mock_model):
        mock_model.exists.return_value = True
        mock_classes.exists.return_value = True
        mock_report.exists.return_value = True

        train_time = timezone.now() - timedelta(days=2)
        dataset_time = timezone.now() - timedelta(days=1)

        with patch("recognition.health._safe_mtime", return_value=train_time):
            res = model_health(dataset_last_updated=dataset_time)

        assert res["model_present"] is True
        assert res["last_trained"] == train_time
        assert res["stale"] is True


@pytest.mark.django_db
class TestEvaluationHealth:
    def test_no_evaluations(self):
        res = evaluation_health()
        assert res["total_evaluations"] == 0
        assert res["latest_evaluation"] is None

    def test_with_evaluations(self):
        eval_result = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.95,
            success=True,
            error_message="some error",
        )
        res = evaluation_health()
        assert res["total_evaluations"] == 1
        assert res["latest_evaluation"]["id"] == eval_result.id
        assert res["latest_evaluation"]["accuracy"] == 0.95
        assert res["latest_evaluation"]["success"] is True

    def test_with_failures(self):
        eval_result = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.5,
            success=False,
        )
        # Update created_at using update to bypass auto_now_add
        ModelEvaluationResult.objects.filter(id=eval_result.id).update(
            created_at=timezone.now() - timedelta(days=2)
        )
        res = evaluation_health()
        assert res["total_evaluations"] == 1
        assert res["recent_failures"] == 1


@pytest.mark.django_db
class TestRecognitionActivity:
    def test_no_activity(self):
        res = recognition_activity()
        assert res["last_attempt"] is None
        assert res["last_spoof"] is None
        assert res["last_success"] is None
        assert res["last_failure"] is None
        assert res["last_outcome"] is None

    def test_with_activity(self):
        user = User.objects.create(username="testuser")

        RecognitionAttempt.objects.create(
            user=user, direction="in", successful=True, spoof_detected=False
        )
        RecognitionAttempt.objects.create(
            user=user,
            direction="out",
            successful=False,
            spoof_detected=True,
            error_message="Spoof detected",
        )
        RecognitionOutcome.objects.create(
            username="testuser",
            direction="in",
            accepted=True,
            distance=0.2,
            threshold=0.5,
            confidence=0.9,
        )

        res = recognition_activity()

        assert res["last_success"]["username"] == "testuser"
        assert res["last_success"]["successful"] is True

        assert res["last_spoof"]["username"] == "testuser"
        assert res["last_spoof"]["spoof_detected"] is True
        assert res["last_spoof"]["error"] == "Spoof detected"

        assert res["last_outcome"]["username"] == "testuser"
        assert res["last_outcome"]["accepted"] is True
        assert res["last_outcome"]["distance"] == 0.2

    def test_with_activity_without_user(self):
        RecognitionAttempt.objects.create(
            username="guestuser", direction="in", successful=True, spoof_detected=False
        )
        res = recognition_activity()
        assert res["last_success"]["username"] == "guestuser"


@pytest.mark.django_db
class TestWorkerHealth:
    @patch("recognition.health.settings")
    def test_broker_not_configured(self, mock_settings):
        mock_settings.CELERY_BROKER_URL = None
        res = worker_health()
        assert res["status"] == "not-configured"
        assert res["workers"] == 0

    @patch("recognition.health.settings")
    @patch("recognition.health.celery_app.control")
    def test_workers_online(self, mock_control, mock_settings):
        mock_settings.CELERY_BROKER_URL = "amqp://"
        mock_control.ping.return_value = [{"celery@worker": {"ok": "pong"}}]
        res = worker_health()
        assert res["status"] == "online"
        assert res["workers"] == 1

    @patch("recognition.health.settings")
    @patch("recognition.health.celery_app.control")
    def test_workers_offline(self, mock_control, mock_settings):
        mock_settings.CELERY_BROKER_URL = "amqp://"
        mock_control.ping.return_value = []
        res = worker_health()
        assert res["status"] == "unreachable"
        assert res["workers"] == 0

    @patch("recognition.health.settings")
    @patch("recognition.health.celery_app.control")
    def test_workers_exception(self, mock_control, mock_settings):
        mock_settings.CELERY_BROKER_URL = "amqp://"
        mock_control.ping.side_effect = Exception("error")
        res = worker_health()
        assert res["status"] == "unreachable"
        assert res["workers"] == 0
