from django.contrib.auth import get_user_model
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

import pytest

from users.models import Direction, RecognitionAttempt


@pytest.mark.django_db
def test_attendance_session_feed_query_count(client):
    admin = get_user_model().objects.create_user(
        username="admin",
        password="password",
        is_staff=True,
    )
    client.force_login(admin)

    # Create multiple users and attempts with EMPTY username but LINKED user
    for i in range(10):
        user = get_user_model().objects.create_user(username=f"user_{i}", password="password")
        RecognitionAttempt.objects.create(
            user=user,
            username="",  # Force fallback to user.username
            direction=Direction.IN,
            successful=True,
            source="webcam",
        )

    url = reverse("attendance-session-feed")

    with CaptureQueriesContext(connection) as ctx:
        response = client.get(url)

    assert response.status_code == 200

    # We expect query count to be LOW (constant) now.
    # 1. Session
    # 2. Auth user
    # 3. RecognitionOutcome
    # 4. RecognitionAttempt (with select_related)
    # Total ~4

    assert (
        len(ctx.captured_queries) <= 5
    ), f"Expected optimized queries, got {len(ctx.captured_queries)}"


@pytest.mark.django_db
def test_attendance_session_feed_query_count_mixed_scenarios(client):
    """Test query count with mixed scenarios: some with username, some without."""
    admin = get_user_model().objects.create_user(
        username="admin",
        password="password",
        is_staff=True,
    )
    client.force_login(admin)

    # Create mixed scenarios
    for i in range(10):
        user = get_user_model().objects.create_user(username=f"user_{i}", password="password")
        # Alternate between populated username and empty username
        username = f"custom_username_{i}" if i % 2 == 0 else ""
        RecognitionAttempt.objects.create(
            user=user,
            username=username,
            direction=Direction.IN,
            successful=True,
            source="webcam",
        )

    url = reverse("attendance-session-feed")

    with CaptureQueriesContext(connection) as ctx:
        response = client.get(url)

    assert response.status_code == 200

    # Query count should remain constant regardless of mixed username scenarios
    assert (
        len(ctx.captured_queries) <= 5
    ), f"Expected optimized queries, got {len(ctx.captured_queries)}"
