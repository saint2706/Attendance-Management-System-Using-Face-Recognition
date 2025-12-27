"""Tests for logout functionality."""

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

User = get_user_model()


@pytest.mark.django_db
def test_logout_get_request_not_allowed(client):
    """Test that GET request to logout endpoint returns 405."""
    # Create and login a user
    user = User.objects.create_user(
        username="testuser",
        email="test@example.com",
        password="testpass123"
    )
    client.force_login(user)
    
    # Attempt GET request to logout
    response = client.get(reverse("logout"))
    
    # Should return 405 Method Not Allowed
    assert response.status_code == 405


@pytest.mark.django_db
def test_logout_post_request_succeeds(client):
    """Test that POST request to logout endpoint works correctly."""
    # Create and login a user
    user = User.objects.create_user(
        username="testuser",
        email="test@example.com",
        password="testpass123"
    )
    client.force_login(user)
    
    # Verify user is authenticated
    response = client.get(reverse("dashboard"))
    assert response.status_code == 200
    
    # POST request to logout
    response = client.post(reverse("logout"))
    
    # Should redirect (302 or 303)
    assert response.status_code in [302, 303]
    
    # Should redirect to home page
    assert response.url == reverse("home")
    
    # Verify user is logged out by trying to access dashboard
    response = client.get(reverse("dashboard"))
    # Should redirect to login since user is no longer authenticated
    assert response.status_code == 302
    assert "/login/" in response.url


@pytest.mark.django_db
def test_logout_with_follow_redirect(client):
    """Test that logout works and follows redirect properly."""
    # Create and login a user
    user = User.objects.create_user(
        username="testuser",
        email="test@example.com",
        password="testpass123"
    )
    client.force_login(user)
    
    # POST request to logout with follow=True to follow the redirect
    response = client.post(reverse("logout"), follow=True)
    
    # Should successfully redirect and land on home page
    assert response.status_code == 200
    # Final URL should be the home page
    assert response.redirect_chain[-1][0] == reverse("home")
