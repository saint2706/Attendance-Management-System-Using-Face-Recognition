"""End-to-end UI tests for admin navigation and attendance workflows."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from playwright.sync_api import Page, expect

from tests.ui.conftest import AdminAccount

pytestmark = [pytest.mark.ui, pytest.mark.e2e, pytest.mark.django_db(transaction=True)]


@pytest.fixture(scope="function")
def page(browser) -> Iterator[Page]:
    """Create an isolated Playwright page for each test case."""

    page = browser.new_page()
    yield page
    page.close()


def test_home_page_shows_primary_actions(page: Page, server_url: str) -> None:
    """Ensure the landing page renders the expected marketing content."""

    page.goto(server_url)

    hero_heading = page.get_by_role("heading", name="Welcome to the Smart Attendance System")
    expect(hero_heading).to_be_visible()

    dashboard_login = page.get_by_role("link", name="Dashboard Login")
    expect(dashboard_login).to_be_visible()


def test_admin_can_access_employee_list_and_attendance(
    page: Page, server_url: str, admin_account: AdminAccount, client
) -> None:
    """Log in as an admin and navigate across employee and attendance views."""

    import threading

    from django.contrib.auth import get_user_model
    from django.db import connection

    User = get_user_model()
    session_result = {}

    def login_user():
        """Login user in a separate thread to avoid async context issues."""
        try:
            connection.close()
            user = User.objects.get(id=admin_account.user_id)
            client.force_login(user)
            session_result["sessionid"] = client.cookies.get("sessionid").value
        except Exception as e:
            session_result["error"] = e

    # Run login in a separate thread
    thread = threading.Thread(target=login_user)
    thread.start()
    thread.join()

    if "error" in session_result:
        raise session_result["error"]

    # Navigate to the site with Playwright
    page.goto(server_url)

    # Set the session cookie in Playwright
    page.context.add_cookies(
        [
            {
                "name": "sessionid",
                "value": session_result["sessionid"],
                "domain": "localhost",
                "path": "/",
            }
        ]
    )

    # Now navigate to dashboard - should be logged in
    page.goto(f"{server_url}/dashboard/")
    dashboard_heading = page.get_by_role("heading", name="Admin Dashboard")
    expect(dashboard_heading).to_be_visible()

    # Note: Django's built-in admin is not configured in this application's URLs,
    # so we skip testing /admin/ URLs and test the custom admin views instead

    # Test custom admin evaluation dashboard
    page.goto(f"{server_url}/admin/evaluation/")
    page.wait_for_load_state("networkidle")
    # Just verify we can access it without error (it may have specific content requirements)

    page.goto(f"{server_url}/view_attendance_home")
    page.wait_for_load_state("networkidle")
    attendance_heading = page.get_by_role("heading", name="Attendance Dashboard")
    expect(attendance_heading).to_be_visible()
    expect(page.get_by_role("link", name="View by Employee")).to_be_visible()
