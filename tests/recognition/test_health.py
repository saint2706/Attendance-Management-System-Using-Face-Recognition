import datetime as dt
from unittest import mock

from django.core.cache import cache
from django.utils import timezone

import pytest

from recognition import health
from recognition.models import ModelEvaluationResult, RecognitionOutcome
from users.models import RecognitionAttempt


@pytest.fixture(autouse=True)
def clear_cache():
    cache.clear()


@pytest.fixture
def mock_dataset_root(tmp_path):
    with mock.patch("recognition.health.TRAINING_DATASET_ROOT", tmp_path):
        yield tmp_path


@pytest.fixture
def mock_model_files(tmp_path):
    model_path = tmp_path / "svc.sav"
    classes_path = tmp_path / "classes.npy"
    report_path = tmp_path / "classification_report.txt"

    with (
        mock.patch("recognition.health.MODEL_PATH", model_path),
        mock.patch("recognition.health.CLASSES_PATH", classes_path),
        mock.patch("recognition.health.REPORT_PATH", report_path),
    ):
        yield model_path, classes_path, report_path


class TestDatasetHealth:
    def test_empty_dataset(self, mock_dataset_root):
        result = health.dataset_health()
        assert result["exists"] is True
        assert result["image_count"] == 0
        assert result["identity_count"] == 0
        assert result["last_updated"] is None

    def test_populated_dataset(self, mock_dataset_root):
        # Create some fake images
        (mock_dataset_root / "person1").mkdir()
        (mock_dataset_root / "person1" / "1.jpg").touch()
        (mock_dataset_root / "person1" / "2.jpg").touch()
        (mock_dataset_root / "person2").mkdir()
        (mock_dataset_root / "person2" / "1.jpg").touch()

        result = health.dataset_health()
        assert result["image_count"] == 3
        assert result["identity_count"] == 2
        assert result["last_updated"] is not None
        assert result["last_updated_display"] is not None


class TestModelHealth:
    def test_no_artifacts(self, mock_model_files):
        result = health.model_health(dataset_last_updated=None)
        assert result["model_present"] is False
        assert result["classes_present"] is False
        assert result["report_present"] is False
        assert result["last_trained"] is None
        assert result["stale"] is False

    def test_all_artifacts_present(self, mock_model_files):
        model, classes, report = mock_model_files
        model.touch()
        classes.touch()
        report.touch()

        result = health.model_health(dataset_last_updated=None)
        assert result["model_present"] is True
        assert result["classes_present"] is True
        assert result["report_present"] is True
        assert result["last_trained"] is not None

    def test_stale_model(self, mock_model_files):
        model, classes, report = mock_model_files
        model.touch()

        # Dataset updated after model
        dataset_time = timezone.now() + dt.timedelta(hours=1)
        result = health.model_health(dataset_last_updated=dataset_time)
        assert result["stale"] is True


@pytest.mark.django_db
class TestEvaluationHealth:
    def test_no_evaluations(self):
        result = health.evaluation_health()
        assert result["latest_evaluation"] is None
        assert result["total_evaluations"] == 0
        assert result["recent_failures"] == 0

    def test_with_evaluations(self):
        eval_result = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.95,
            success=True,
            created_at=timezone.now(),
        )

        result = health.evaluation_health()
        assert result["total_evaluations"] == 1
        assert result["latest_nightly"]["id"] == eval_result.id
        assert result["latest_nightly"]["accuracy"] == 0.95
        assert result["latest_evaluation"]["id"] == eval_result.id

    def test_with_failures(self):
        ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            accuracy=0.5,
            success=False,
            created_at=timezone.now(),
        )

        result = health.evaluation_health()
        assert result["total_evaluations"] == 1
        assert result["recent_failures"] == 1


@pytest.mark.django_db
class TestRecognitionActivity:
    def test_no_activity(self):
        result = health.recognition_activity()
        assert result["last_attempt"] is None
        assert result["last_spoof"] is None
        assert result["last_success"] is None
        assert result["last_outcome"] is None

    def test_with_activity(self, django_user_model):
        user = django_user_model.objects.create(username="testuser")

        RecognitionAttempt.objects.create(
            user=user, username="testuser", successful=True, spoof_detected=False
        )

        RecognitionOutcome.objects.create(
            username="testuser", accepted=True, distance=0.1, threshold=0.4, confidence=0.9
        )

        result = health.recognition_activity()
        assert result["last_attempt"]["username"] == "testuser"
        assert result["last_success"]["username"] == "testuser"
        assert result["last_outcome"]["username"] == "testuser"


class TestWorkerHealth:
    def test_broker_not_configured(self, settings):
        settings.CELERY_BROKER_URL = None
        result = health.worker_health()
        assert result["status"] == "not-configured"

    @mock.patch("recognition.health.celery_app.control.ping")
    def test_workers_online(self, mock_ping, settings):
        settings.CELERY_BROKER_URL = "redis://localhost:6379/0"
        mock_ping.return_value = [{"celery@worker1": {"ok": "pong"}}]

        result = health.worker_health()
        assert result["status"] == "online"
        assert result["workers"] == 1

    @mock.patch("recognition.health.celery_app.control.ping")
    def test_workers_offline(self, mock_ping, settings):
        settings.CELERY_BROKER_URL = "redis://localhost:6379/0"
        mock_ping.return_value = []

        result = health.worker_health()
        assert result["status"] == "unreachable"
        assert result["workers"] == 0

    def test_cached_dataset(self, mock_dataset_root):
        # Create some fake images
        (mock_dataset_root / "person1").mkdir()
        (mock_dataset_root / "person1" / "1.jpg").touch()

        # First call hits the filesystem
        result1 = health.dataset_health()
        assert result1["image_count"] == 1

        # Add another image, but it shouldn't be reflected in the cached result
        (mock_dataset_root / "person1" / "2.jpg").touch()
        result2 = health.dataset_health()
        assert result2["image_count"] == 1

    def test_isoformat_naive_datetime(self):
        naive_dt = dt.datetime(2023, 1, 1, 12, 0)
        result = health._isoformat_or_none(naive_dt)
        assert result is not None
        assert "T12:00:00+00:00" in result
