from django.contrib.auth import get_user_model
from django.urls import reverse

import pytest

from recognition.models import RecognitionOutcome
from users.models import RecognitionAttempt


@pytest.mark.django_db
def test_attendance_session_page_requires_login(client):
    response = client.get(reverse("attendance-session"))
    assert response.status_code == 302
    assert reverse("login") in response.url


@pytest.mark.django_db
def test_attendance_session_feed_surfaces_liveness_and_outcomes(client):
    user = get_user_model().objects.create_user(
        username="session-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    RecognitionAttempt.objects.create(
        username="spoof-user",
        direction=RecognitionAttempt.Direction.IN,
        spoof_detected=True,
        successful=False,
        source="webcam",
    )
    RecognitionOutcome.objects.create(
        username="real-user",
        direction="in",
        accepted=True,
        confidence=0.82,
        source="webcam",
    )

    response = client.get(reverse("attendance-session-feed"))

    assert response.status_code == 200
    payload = response.json()
    events = payload["events"]
    assert any(event.get("event_type") == "attempt" for event in events)
    assert any(event.get("liveness") == "failed" for event in events)
    assert any(event.get("event_type") == "outcome" and event.get("accepted") for event in events)
