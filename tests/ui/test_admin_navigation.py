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
    page: Page, server_url: str, admin_account: AdminAccount
) -> None:
    """Log in as an admin and navigate across employee and attendance views."""

    page.goto(server_url)

    page.get_by_role("link", name="Login").click()
    expect(page).to_have_url(f"{server_url}/login/")

    page.fill('input[name="username"]', admin_account.username)
    page.fill('input[name="password"]', admin_account.password)
    page.get_by_role("button", name="Login").click()

    page.wait_for_url(f"{server_url}/dashboard/")
    dashboard_heading = page.get_by_role("heading", name="Admin Dashboard")
    expect(dashboard_heading).to_be_visible()

    page.goto(f"{server_url}/admin/auth/user/")
    page.wait_for_load_state("networkidle")
    admin_heading = page.get_by_role("heading", name="Select user to change")
    expect(admin_heading).to_be_visible()
    expect(page.locator("#result_list")).to_be_visible()

    page.goto(f"{server_url}/view_attendance_home")
    page.wait_for_load_state("networkidle")
    attendance_heading = page.get_by_role("heading", name="Attendance Dashboard")
    expect(attendance_heading).to_be_visible()
    expect(page.get_by_role("link", name="View by Employee")).to_be_visible()
