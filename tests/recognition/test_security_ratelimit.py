from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.cache import cache
from django.urls import reverse

import pytest


@pytest.fixture(autouse=True)
def clear_rate_limit_cache():
    """Clear the rate-limit cache before and after each test to prevent state leakage."""
    cache.clear()
    yield
    cache.clear()


@pytest.mark.django_db
def test_train_view_rate_limit(client):
    """Test that the train view is rate limited."""
    # Create a staff user
    user = User.objects.create_user(username="staff_user", password="password", is_staff=True)
    client.force_login(user)

    url = reverse("train")

    # We need to mock the task delay to avoid starting actual tasks
    with patch("recognition.tasks.train_recognition_model.delay") as mock_task:
        mock_task.return_value.id = "fake-task-id"

        # Make 4 requests (rate limit is 3/m)
        for i in range(4):
            response = client.post(url)

            # The 4th request should be rate limited
            if i < 3:
                # First 3 should succeed (status 200 or 302 redirect)
                assert response.status_code in [200, 302], f"Request {i + 1} should succeed"
            else:
                # 4th request should be rate limited
                # Check if messages contain rate limit error
                messages = list(response.wsgi_request._messages)
                assert any(
                    "Too many training requests" in str(m) for m in messages
                ), "Rate limit message should be shown"
                return  # Test passed!

    pytest.fail("Rate limit was not triggered after 4 requests")


@pytest.mark.django_db
def test_add_photos_view_rate_limit(client):
    """Test that the add_photos view is rate limited."""
    # Create a staff user
    user = User.objects.create_user(username="staff_user", password="password", is_staff=True)
    client.force_login(user)

    url = reverse("add-photos")

    # We need to mock the task delay to avoid starting actual tasks
    with patch("recognition.tasks.capture_dataset.delay") as mock_task:
        mock_task.return_value.id = "fake-task-id"

        # Make 11 requests (rate limit is 10/m)
        for i in range(11):
            response = client.post(url, {"username": "testuser"})

            # The 11th request should be rate limited
            if i < 10:
                # First 10 should process normally
                assert response.status_code in [200, 302], f"Request {i + 1} should succeed"
            else:
                # 11th request should be rate limited
                # Check if messages contain rate limit error
                messages = list(response.wsgi_request._messages)
                assert any(
                    "Too many capture requests" in str(m) for m in messages
                ), "Rate limit message should be shown"
                return  # Test passed!

    pytest.fail("Rate limit was not triggered after 11 requests")
