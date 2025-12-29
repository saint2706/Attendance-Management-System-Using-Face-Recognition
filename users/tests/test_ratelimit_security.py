from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import Client
from django.urls import reverse

import pytest

from users.models import SetupWizardProgress


@pytest.fixture(autouse=True)
def clear_rate_limit_cache():
    """Clear rate limit cache before each test."""
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def test_user(db):
    """Create a test user for login tests."""
    User = get_user_model()
    return User.objects.create_user(
        username="testuser",
        password="testpassword123",
        email="testuser@example.com"
    )


@pytest.mark.django_db
def test_register_rate_limit_fix(client):
    User = get_user_model()
    staff_user = User.objects.create_user(
        username="staff_limit_2", password="password", is_staff=True
    )
    client.force_login(staff_user)
    url = reverse("register")

    # Limit is 10/m
    for i in range(10):
        response = client.post(url, {"username": f"user{i}", "password": "password"})
        assert response.status_code != 429

    # 11th attempt
    response = client.post(url, {"username": "user11", "password": "password"})
    assert response.status_code == 429
    assert b"Too many registration attempts" in response.content


@pytest.mark.django_db
def test_wizard_step3_rate_limit(client):
    User = get_user_model()
    staff_user = User.objects.create_user(username="staff_wiz3", password="password", is_staff=True)
    # Setup prerequisite state
    SetupWizardProgress.objects.create(
        user=staff_user,
        current_step=SetupWizardProgress.Step.ADD_EMPLOYEE,
        camera_tested=True,
        liveness_tested=True,
    )

    client.force_login(staff_user)
    url = reverse("setup-wizard-step3")

    # Limit is 10/m - send simple POSTs to trigger rate limit
    for i in range(10):
        # Send POST with minimal data to avoid successful form submission
        response = client.post(url, {})
        assert response.status_code != 429

    # 11th attempt - should be rate limited
    response = client.post(url, {})
    assert response.status_code == 200

    messages = list(response.context["messages"])
    assert len(messages) > 0
    assert "Too many attempts" in str(messages[0])


@pytest.mark.django_db
def test_wizard_step4_rate_limit(client):
    User = get_user_model()
    staff_user = User.objects.create_user(username="staff_wiz", password="password", is_staff=True)
    # Setup prerequisite state
    SetupWizardProgress.objects.create(
        user=staff_user,
        current_step=SetupWizardProgress.Step.TRAIN_MODEL,
        first_employee_username="testemployee",
        first_employee_photos_captured=True,
        camera_tested=True,
        liveness_tested=True,
    )

    client.force_login(staff_user)
    url = reverse("setup-wizard-step4")

    # Limit is 5/m
    for i in range(5):
        # We don't send 'start_training' to avoid mocking celery here, just POSTing is enough to trigger ratelimit
        response = client.post(url, {})
        assert response.status_code != 429

    # 6th attempt
    response = client.post(url, {})
    # Note: setup_wizard steps return 200 (re-render) even if limited.
    assert response.status_code == 200

    messages = list(response.context["messages"])
    assert len(messages) > 0
    assert "Too many attempts" in str(messages[0])


@pytest.mark.django_db
def test_login_ip_based_rate_limit(client, test_user):
    """
    Test that login attempts from the same IP are rate-limited.
    This tests the existing IP-based rate limiting behavior.
    """
    url = reverse("login")
    
    # Make 5 failed login attempts (at the limit)
    for i in range(5):
        response = client.post(url, {
            "username": "testuser",
            "password": "wrongpassword"
        })
        # Should not be rate limited yet
        assert response.status_code in [200, 302]  # 302 if wrong password redirects
        if response.status_code == 200:
            # Check that it's not a 429 error
            assert b"Too many login attempts" not in response.content
    
    # 6th attempt should be rate limited
    response = client.post(url, {
        "username": "testuser",
        "password": "wrongpassword"
    })
    assert response.status_code == 429
    assert b"Too many login attempts" in response.content


@pytest.mark.django_db
def test_login_username_based_rate_limit(client, test_user):
    """
    Test that login attempts for the same username are rate-limited,
    even from the same IP.
    """
    url = reverse("login")
    
    # Make 5 failed login attempts for the same username
    for i in range(5):
        response = client.post(url, {
            "username": "testuser",
            "password": f"wrongpassword{i}"
        })
        # Should not be rate limited yet
        assert response.status_code in [200, 302]
        if response.status_code == 200:
            assert b"Too many login attempts" not in response.content
    
    # 6th attempt should be rate limited
    response = client.post(url, {
        "username": "testuser",
        "password": "wrongpassword6"
    })
    assert response.status_code == 429
    assert b"Too many login attempts" in response.content


@pytest.mark.django_db
def test_distributed_brute_force_prevention(test_user):
    """
    Test that username-based rate limiting prevents distributed brute-force attacks.
    Multiple requests from different IPs targeting the same username should be rate-limited.
    """
    url = reverse("login")
    
    # Simulate requests from different IPs by creating separate clients with different REMOTE_ADDR
    for i in range(5):
        client = Client()
        # Simulate different IP addresses
        response = client.post(
            url,
            {
                "username": "testuser",
                "password": f"wrongpassword{i}"
            },
            REMOTE_ADDR=f"192.168.1.{i+1}"
        )
        # Should not be rate limited yet
        assert response.status_code in [200, 302]
        if response.status_code == 200:
            assert b"Too many login attempts" not in response.content
    
    # 6th attempt from yet another IP should be rate limited due to username limit
    client = Client()
    response = client.post(
        url,
        {
            "username": "testuser",
            "password": "wrongpassword6"
        },
        REMOTE_ADDR="192.168.1.100"
    )
    assert response.status_code == 429
    assert b"Too many login attempts" in response.content


@pytest.mark.django_db
def test_successful_login_under_rate_limit(client, test_user):
    """
    Test that successful logins work correctly when under the rate limit.
    """
    url = reverse("login")
    
    # Make a few failed attempts (under the limit)
    for i in range(3):
        response = client.post(url, {
            "username": "testuser",
            "password": "wrongpassword"
        })
        assert response.status_code in [200, 302]
    
    # Successful login should still work
    response = client.post(url, {
        "username": "testuser",
        "password": "testpassword123"
    })
    # Successful login should redirect
    assert response.status_code == 302
    
    # Verify user is logged in by checking if we can access a protected page
    # or by checking the session
    assert "_auth_user_id" in client.session


@pytest.mark.django_db
def test_different_usernames_have_separate_limits(db):
    """
    Test that rate limits are applied per username, not globally.
    Different usernames should have separate rate limit counters.
    """
    User = get_user_model()
    user1 = User.objects.create_user(username="user1", password="password1")
    user2 = User.objects.create_user(username="user2", password="password2")
    
    url = reverse("login")
    
    # Make 5 failed attempts for user1 (at the limit) from different IPs to avoid IP limit
    for i in range(5):
        client = Client()
        response = client.post(
            url,
            {
                "username": "user1",
                "password": "wrongpassword"
            },
            REMOTE_ADDR=f"10.0.0.{i+1}"
        )
        assert response.status_code in [200, 302]
    
    # Attempts for user2 should still work (separate counter)
    for i in range(3):
        client = Client()
        response = client.post(
            url,
            {
                "username": "user2",
                "password": "wrongpassword"
            },
            REMOTE_ADDR=f"10.0.1.{i+1}"
        )
        assert response.status_code in [200, 302]
        if response.status_code == 200:
            assert b"Too many login attempts" not in response.content
    
    # But user1 should be rate limited on next attempt (even from a new IP)
    client = Client()
    response = client.post(
        url,
        {
            "username": "user1",
            "password": "wrongpassword"
        },
        REMOTE_ADDR="10.0.0.100"
    )
    assert response.status_code == 429
    assert b"Too many login attempts" in response.content

