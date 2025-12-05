import json
import os
import sys
from unittest.mock import MagicMock

import django
from django.core.cache import cache
from django.test import override_settings
from django.urls import reverse

import numpy as np
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")

sys.modules.setdefault("cv2", MagicMock())

# Only setup Django if it hasn't been configured yet (e.g., running standalone)
if not django.apps.apps.ready:
    django.setup()

from recognition import views_legacy  # noqa: E402, F401

pytestmark = pytest.mark.django_db


@override_settings(
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_face_recognition_api_returns_match(client, monkeypatch):
    cache.clear()

    monkeypatch.setattr(
        views_legacy,
        "_load_dataset_embeddings_for_matching",
        lambda *args, **kwargs: [
            {
                "embedding": np.array([0.1, 0.2, 0.3], dtype=float),
                "username": "alice",
                "identity": "alice/sample.jpg",
            }
        ],
    )
    monkeypatch.setattr(
        views_legacy,
        "find_closest_dataset_match",
        lambda embedding, dataset, metric: ("alice", 0.05, "alice/sample.jpg"),
    )

    url = reverse("face-recognition-api")
    payload = json.dumps({"embedding": [0.1, 0.2, 0.3]})
    response = client.post(url, data=payload, content_type="application/json")

    assert response.status_code == 200
    data = response.json()
    assert data["recognized"] is True
    assert data["username"] == "alice"
    assert data["identity"] == "alice/sample.jpg"
    assert data["distance_metric"] == "euclidean_l2"
    assert pytest.approx(data["distance"]) == 0.05


@override_settings(
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_face_recognition_api_rate_limit(client, monkeypatch):
    cache.clear()

    monkeypatch.setattr(
        views_legacy,
        "_load_dataset_embeddings_for_matching",
        lambda *args, **kwargs: [
            {
                "embedding": np.array([0.1, 0.2, 0.3], dtype=float),
                "username": "alice",
                "identity": "alice/sample.jpg",
            }
        ],
    )
    monkeypatch.setattr(
        views_legacy,
        "find_closest_dataset_match",
        lambda embedding, dataset, metric: ("alice", 0.05, "alice/sample.jpg"),
    )

    url = reverse("face-recognition-api")
    payload = json.dumps({"embedding": [0.1, 0.2, 0.3]})

    for _ in range(5):
        ok_response = client.post(url, data=payload, content_type="application/json")
        assert ok_response.status_code == 200

    limited = client.post(url, data=payload, content_type="application/json")
    assert limited.status_code == 429
