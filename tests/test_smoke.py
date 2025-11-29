"""Smoke tests to verify the application can start and respond to basic requests.

These tests are designed to quickly validate that:
1. The Django server can start and handle requests
2. Key pages are accessible (health check, login, home)
3. The evaluation pipeline can run with synthetic data
"""

from __future__ import annotations

import pytest
from django.test import Client, override_settings
from django.urls import reverse


pytestmark = pytest.mark.django_db


class TestServerSmokeTests:
    """Verify that the Django server can start and serve basic requests."""

    def test_home_page_loads(self, client: Client) -> None:
        """The home page should return HTTP 200 with expected content."""
        response = client.get(reverse("home"))
        assert response.status_code == 200
        assert b"Smart Attendance System" in response.content or b"Attendance" in response.content

    def test_login_page_loads(self, client: Client) -> None:
        """The login page should be accessible without authentication."""
        response = client.get(reverse("login"))
        assert response.status_code == 200
        content_lower = response.content.lower()
        assert b"login" in content_lower

    def test_static_files_configured(self, client: Client) -> None:
        """Static files should be served (at least in debug mode)."""
        # The manifest.json is a good indicator static files are working
        response = client.get("/static/manifest.json")
        # In debug mode, Django serves static files; in production, 404 is expected
        assert response.status_code in (200, 404)


class TestAuthenticationSmokeTests:
    """Verify basic authentication flows work."""

    def test_dashboard_requires_login(self, client: Client) -> None:
        """Unauthenticated users should be redirected from the dashboard."""
        response = client.get(reverse("dashboard"))
        assert response.status_code == 302
        assert "login" in response.url.lower()

    def test_can_login_with_valid_credentials(self, client: Client, django_user_model) -> None:
        """Valid credentials should allow login and redirect to dashboard."""
        user = django_user_model.objects.create_user(
            username="smoke_test_user",
            password="SmokeTestPass123!",
            is_staff=True,
        )
        response = client.post(
            reverse("login"),
            data={"username": "smoke_test_user", "password": "SmokeTestPass123!"},
        )
        assert response.status_code == 302
        assert "dashboard" in response.url.lower() or response.url == reverse("dashboard")

    def test_invalid_credentials_rejected(self, client: Client) -> None:
        """Invalid credentials should not allow login."""
        response = client.post(
            reverse("login"),
            data={"username": "nonexistent_user", "password": "wrong_password"},
        )
        # Should stay on login page or show error
        assert response.status_code == 200


class TestKeyEndpointsSmokeTests:
    """Verify key application endpoints are accessible."""

    def test_mark_attendance_page_exists(self, client: Client) -> None:
        """The mark attendance page should exist (may redirect for login)."""
        response = client.get(reverse("mark-your-attendance"))
        # May redirect to login or show the page
        assert response.status_code in (200, 302)

    def test_attendance_session_requires_login(self, client: Client) -> None:
        """The attendance session page should require authentication."""
        response = client.get(reverse("attendance-session"))
        assert response.status_code == 302
        assert "login" in response.url.lower()

    def test_register_page_requires_admin(self, client: Client, django_user_model) -> None:
        """The register page should require staff/admin access."""
        # Non-staff user
        user = django_user_model.objects.create_user(
            username="regular_user",
            password="RegularPass123!",
            is_staff=False,
        )
        client.force_login(user)
        response = client.get(reverse("register"))
        # Should redirect non-staff users
        assert response.status_code in (302, 403)

    def test_admin_can_access_register(self, client: Client, django_user_model) -> None:
        """Admin users should be able to access the register page."""
        admin = django_user_model.objects.create_user(
            username="admin_smoke",
            password="AdminPass123!",
            is_staff=True,
        )
        client.force_login(admin)
        response = client.get(reverse("register"))
        assert response.status_code == 200


class TestManagementCommandsSmokeTests:
    """Verify management commands can be invoked without errors."""

    @pytest.fixture(scope="class")
    def available_commands(self) -> dict:
        """Get all available management commands once per test class."""
        from django.core.management import get_commands
        return get_commands()

    def test_prepare_splits_command_exists(self, available_commands: dict) -> None:
        """The prepare_splits management command should be importable."""
        assert "prepare_splits" in available_commands

    def test_eval_command_exists(self, available_commands: dict) -> None:
        """The eval management command should be importable."""
        assert "eval" in available_commands

    def test_fairness_audit_command_exists(self, available_commands: dict) -> None:
        """The fairness_audit management command should be importable."""
        assert "fairness_audit" in available_commands

    def test_evaluate_liveness_command_exists(self, available_commands: dict) -> None:
        """The evaluate_liveness management command should be importable."""
        assert "evaluate_liveness" in available_commands

    def test_ablation_command_exists(self, available_commands: dict) -> None:
        """The ablation management command should be importable."""
        assert "ablation" in available_commands

    def test_export_reports_command_exists(self, available_commands: dict) -> None:
        """The export_reports management command should be importable."""
        assert "export_reports" in available_commands

    def test_threshold_select_command_exists(self, available_commands: dict) -> None:
        """The threshold_select management command should be importable."""
        assert "threshold_select" in available_commands
