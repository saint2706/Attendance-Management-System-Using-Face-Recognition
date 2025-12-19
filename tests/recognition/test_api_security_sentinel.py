import json
import pytest
from django.urls import reverse
from django.test import override_settings
import numpy as np
from recognition import views_legacy
from users.models import RecognitionAttempt
from django.core.cache import cache

@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("secure-key-1",),
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_long_username_truncation(client, monkeypatch):
    cache.clear()

    # Mock matching failure so it logs failure with submitted username
    monkeypatch.setattr(
        views_legacy,
        "_load_dataset_embeddings_for_matching",
        lambda *args, **kwargs: [],
    )

    url = reverse("face-recognition-api")
    long_username = "a" * 200
    payload = json.dumps({
        "embedding": [0.1, 0.2, 0.3],
        "username": long_username
    })

    # We expect 503 because no embeddings are loaded, but we care about the logging side effect
    response = client.post(
        url,
        data=payload,
        content_type="application/json",
        HTTP_X_API_KEY="secure-key-1"
    )

    assert response.status_code == 503

    # Check the DB record
    attempt = RecognitionAttempt.objects.order_by('-created_at').first()
    assert attempt is not None
    # This assertion checks if truncation works
    assert len(attempt.username) <= 150
    assert attempt.username == long_username[:150]

@pytest.mark.django_db
@override_settings(RECOGNITION_API_KEYS=("secure-key-1", "secure-key-2"))
def test_api_key_authentication_works(client, monkeypatch):
    cache.clear()

    # We need to mock _load_dataset_embeddings_for_matching to avoid 503 and return success/failure
    monkeypatch.setattr(
        views_legacy,
        "_load_dataset_embeddings_for_matching",
        lambda *args, **kwargs: [],
    )

    url = reverse("face-recognition-api")
    payload = json.dumps({"embedding": [0.1, 0.2, 0.3]})

    # Test valid key
    response = client.post(
        url,
        data=payload,
        content_type="application/json",
        HTTP_X_API_KEY="secure-key-1"
    )
    # Should be 503 because we mocked empty embeddings, BUT NOT 401
    assert response.status_code == 503

    # Test invalid key
    response = client.post(
        url,
        data=payload,
        content_type="application/json",
        HTTP_X_API_KEY="wrong-key"
    )
    assert response.status_code == 401
