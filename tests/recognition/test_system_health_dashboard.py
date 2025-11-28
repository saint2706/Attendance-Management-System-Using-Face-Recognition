from __future__ import annotations

import datetime as dt
import os

import pytest
from django.urls import reverse

from recognition import health
from recognition.models import RecognitionOutcome
from users.models import RecognitionAttempt

pytestmark = pytest.mark.django_db


def test_health_helpers_report_dataset_and_model(tmp_path, monkeypatch):
    """Dataset and model helpers should report counts and freshness."""

    data_root = tmp_path / "face_recognition_data"
    training_root = data_root / "training_dataset"
    user_dir = training_root / "user-1"
    user_dir.mkdir(parents=True, exist_ok=True)

    image_path = user_dir / "img.jpg"
    image_path.write_bytes(b"data")

    model_path = data_root / "svc.sav"
    classes_path = data_root / "classes.npy"
    report_path = data_root / "classification_report.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    classes_path.write_bytes(b"classes")
    report_path.write_bytes(b"report")

    monkeypatch.setattr(health, "DATA_ROOT", data_root)
    monkeypatch.setattr(health, "TRAINING_DATASET_ROOT", training_root)
    monkeypatch.setattr(health, "MODEL_PATH", model_path)
    monkeypatch.setattr(health, "CLASSES_PATH", classes_path)
    monkeypatch.setattr(health, "REPORT_PATH", report_path)

    dataset_snapshot = health.dataset_health()
    assert dataset_snapshot["image_count"] == 1
    assert dataset_snapshot["identity_count"] == 1
    assert dataset_snapshot["exists"] is True

    stale_reference = dt.datetime.now(tz=dt.timezone.utc) + dt.timedelta(seconds=5)
    os.utime(image_path, (stale_reference.timestamp(), stale_reference.timestamp()))
    os.utime(model_path, (stale_reference.timestamp() - 100, stale_reference.timestamp() - 100))
    os.utime(classes_path, (stale_reference.timestamp() - 100, stale_reference.timestamp() - 100))

    model_snapshot = health.model_health(dataset_last_updated=dataset_snapshot["last_updated"])
    assert model_snapshot["model_present"] is True
    assert model_snapshot["classes_present"] is True
    assert model_snapshot["stale"] is True



def test_recognition_activity_captures_attempts(django_user_model):
    """Recognition activity should expose last attempts and outcomes."""

    user = django_user_model.objects.create_user(username="activity-user", password="Password!234")

    RecognitionAttempt.objects.create(
        username=user.username,
        direction=RecognitionAttempt.Direction.IN,
        site="hq",
        source="webcam",
        successful=False,
        spoof_detected=True,
        error_message="Liveness gate failed",
    )
    RecognitionOutcome.objects.create(
        username=user.username,
        direction="in",
        accepted=False,
        distance=0.9,
        threshold=0.5,
    )

    activity = health.recognition_activity()

    assert activity["last_attempt"]["spoof_detected"] is True
    assert activity["last_outcome"]["accepted"] is False



def test_system_health_dashboard_context(client, django_user_model, monkeypatch):
    """The admin dashboard should expose dataset, model, worker, and activity state."""

    admin = django_user_model.objects.create_user(
        username="admin",
        password="AdminPass!234",
        is_staff=True,
    )
    client.force_login(admin)

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
    monkeypatch.setattr(
        health, "worker_health", lambda: {"status": "online", "workers": 1}
    )

    response = client.get(reverse("admin_system_health"))

    assert response.status_code == 200
    assert response.context["dataset"]["image_count"] == 2
    assert response.context["model"]["model_present"] is True
    assert response.context["worker_state"]["status"] == "online"
