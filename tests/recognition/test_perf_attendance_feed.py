from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password
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

    hashed_password = make_password("password")

    # Create multiple users and attempts with EMPTY username but LINKED user
    users = [get_user_model()(username=f"user_{i}", password=hashed_password) for i in range(10)]
    created_users = get_user_model().objects.bulk_create(users)

    attempts = [
        RecognitionAttempt(
            user=u,
            username="",  # Force fallback to user.username
            direction=Direction.IN,
            successful=True,
            source="webcam",
        )
        for u in created_users
    ]
    RecognitionAttempt.objects.bulk_create(attempts)

    url = reverse("attendance-session-feed")

    with CaptureQueriesContext(connection) as ctx:
        response = client.get(url)

    assert response.status_code == 200

    filtered_queries = [
        q
        for q in ctx.captured_queries
        if not q["sql"].startswith("EXPLAIN")
        and "silk_" not in q["sql"].lower()
        and not q["sql"].startswith("SAVEPOINT")
        and not q["sql"].startswith("RELEASE SAVEPOINT")
    ]

    # We expect query count to be LOW (constant) now.
    # 1. Session
    # 2. Auth user
    # 3. RecognitionOutcome
    # 4. RecognitionAttempt (with select_related)
    # Total ~4

    assert len(filtered_queries) <= 5, f"Expected optimized queries, got {len(filtered_queries)}"


@pytest.mark.django_db
def test_attendance_session_feed_query_count_mixed_scenarios(client):
    """Test query count with mixed scenarios: some with username, some without."""
    admin = get_user_model().objects.create_user(
        username="admin",
        password="password",
        is_staff=True,
    )
    client.force_login(admin)

    hashed_password = make_password("password")

    # Create mixed scenarios
    users = [get_user_model()(username=f"user_{i}", password=hashed_password) for i in range(10)]
    created_users = get_user_model().objects.bulk_create(users)

    attempts = [
        RecognitionAttempt(
            user=u,
            username=f"custom_username_{i}" if i % 2 == 0 else "",
            direction=Direction.IN,
            successful=True,
            source="webcam",
        )
        for i, u in enumerate(created_users)
    ]
    RecognitionAttempt.objects.bulk_create(attempts)

    url = reverse("attendance-session-feed")

    with CaptureQueriesContext(connection) as ctx:
        response = client.get(url)

    assert response.status_code == 200

    filtered_queries = [
        q
        for q in ctx.captured_queries
        if not q["sql"].startswith("EXPLAIN")
        and "silk_" not in q["sql"].lower()
        and not q["sql"].startswith("SAVEPOINT")
        and not q["sql"].startswith("RELEASE SAVEPOINT")
    ]

    # Query count should remain constant regardless of mixed username scenarios
    assert len(filtered_queries) <= 5, f"Expected optimized queries, got {len(filtered_queries)}"
