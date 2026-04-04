from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

import pytest

from users.models import Direction, RecognitionAttempt


@pytest.mark.django_db
def test_attendance_session_feed_query_count(client):
    User = get_user_model()
    # ⚡ Bolt: Optimize setup by hashing password once
    hashed_pw = make_password("password")
    admin = User.objects.create(
        username="admin",
        password=hashed_pw,
        is_staff=True,
    )
    client.force_login(admin)

    # ⚡ Bolt: Use bulk_create for test objects to optimize test speed
    users_to_create = [User(username=f"user_{i}", password=hashed_pw) for i in range(10)]
    User.objects.bulk_create(users_to_create)
    created_users = User.objects.filter(username__startswith="user_")

    # Create multiple attempts with EMPTY username but LINKED user
    attempts_to_create = [
        RecognitionAttempt(
            user=user,
            username="",  # Force fallback to user.username
            direction=Direction.IN,
            successful=True,
            source="webcam",
        )
        for user in created_users
    ]
    RecognitionAttempt.objects.bulk_create(attempts_to_create)

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
    User = get_user_model()
    # ⚡ Bolt: Optimize setup by hashing password once
    hashed_pw = make_password("password")
    admin = User.objects.create(
        username="admin",
        password=hashed_pw,
        is_staff=True,
    )
    client.force_login(admin)

    # ⚡ Bolt: Use bulk_create for test objects to optimize test speed
    users_to_create = [User(username=f"user_{i}", password=hashed_pw) for i in range(10)]
    User.objects.bulk_create(users_to_create)
    created_users = list(User.objects.filter(username__startswith="user_").order_by("username"))

    # Create mixed scenarios
    attempts_to_create = []
    for i, user in enumerate(created_users):
        # Alternate between populated username and empty username
        username = f"custom_username_{i}" if i % 2 == 0 else ""
        attempts_to_create.append(
            RecognitionAttempt(
                user=user,
                username=username,
                direction=Direction.IN,
                successful=True,
                source="webcam",
            )
        )
    RecognitionAttempt.objects.bulk_create(attempts_to_create)

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
