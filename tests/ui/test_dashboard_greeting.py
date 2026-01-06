"""End-to-end UI tests for the time-based greeting feature on the Admin Dashboard."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

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


def test_dashboard_shows_morning_greeting(
    page: Page, server_url: str, admin_account: AdminAccount, client
) -> None:
    """Verify that 'Good morning' greeting displays when hour is before 12."""

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

    # Mock the time to be in the morning (e.g., 10 AM)
    # Inject a script to set testGreetingHour before the page script runs
    page.add_init_script("""
        window.testGreetingHour = 10;
    """)

    # Navigate to Django dashboard
    page.goto(f"{server_url}/dashboard/")
    dashboard_heading = page.get_by_role("heading", name="Admin Dashboard")
    expect(dashboard_heading).to_be_visible()

    # Wait for the greeting JavaScript to execute
    page.wait_for_function(
        """() => {
            const elem = document.getElementById('greeting-text');
            return elem && !elem.textContent.startsWith('Welcome,');
        }""",
        timeout=5000
    )

    # Verify the morning greeting appears with the username
    greeting_text = page.locator("#greeting-text")
    expect(greeting_text).to_have_text(f"Good morning, {admin_account.username}!")


def test_dashboard_shows_afternoon_greeting(
    page: Page, server_url: str, admin_account: AdminAccount, client
) -> None:
    """Verify that 'Good afternoon' greeting displays when hour is between 12 and 18."""

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

    # Mock the time to be in the afternoon (e.g., 3 PM)
    # Inject a script to set testGreetingHour before the page script runs
    page.add_init_script("""
        window.testGreetingHour = 15;
    """)

    # Navigate to Django dashboard
    page.goto(f"{server_url}/dashboard/")
    dashboard_heading = page.get_by_role("heading", name="Admin Dashboard")
    expect(dashboard_heading).to_be_visible()

    # Wait for the greeting JavaScript to execute
    page.wait_for_function(
        """() => {
            const elem = document.getElementById('greeting-text');
            return elem && !elem.textContent.startsWith('Welcome,');
        }""",
        timeout=5000
    )

    # Verify the afternoon greeting appears with the username
    greeting_text = page.locator("#greeting-text")
    expect(greeting_text).to_have_text(f"Good afternoon, {admin_account.username}!")


def test_dashboard_shows_evening_greeting(
    page: Page, server_url: str, admin_account: AdminAccount, client
) -> None:
    """Verify that 'Good evening' greeting displays when hour is 18 or later."""

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

    # Mock the time to be in the evening (e.g., 8 PM)
    # Inject a script to set testGreetingHour before the page script runs
    page.add_init_script("""
        window.testGreetingHour = 20;
    """)

    # Navigate to Django dashboard
    page.goto(f"{server_url}/dashboard/")
    dashboard_heading = page.get_by_role("heading", name="Admin Dashboard")
    expect(dashboard_heading).to_be_visible()

    # Wait for the greeting JavaScript to execute
    page.wait_for_function(
        """() => {
            const elem = document.getElementById('greeting-text');
            return elem && !elem.textContent.startsWith('Welcome,');
        }""",
        timeout=5000
    )

    # Verify the evening greeting appears with the username
    greeting_text = page.locator("#greeting-text")
    expect(greeting_text).to_have_text(f"Good evening, {admin_account.username}!")


def test_dashboard_greeting_boundary_at_noon(
    page: Page, server_url: str, admin_account: AdminAccount, client
) -> None:
    """Verify that greeting switches from morning to afternoon at exactly 12:00."""

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

    # Mock the time to be exactly at noon (12:00 PM)
    # Inject a script to set testGreetingHour before the page script runs
    page.add_init_script("""
        window.testGreetingHour = 12;
    """)

    # Navigate to Django dashboard
    page.goto(f"{server_url}/dashboard/")
    dashboard_heading = page.get_by_role("heading", name="Admin Dashboard")
    expect(dashboard_heading).to_be_visible()

    # Wait for the greeting JavaScript to execute
    page.wait_for_function(
        """() => {
            const elem = document.getElementById('greeting-text');
            return elem && !elem.textContent.startsWith('Welcome,');
        }""",
        timeout=5000
    )

    # At exactly 12:00, it should be "Good afternoon"
    greeting_text = page.locator("#greeting-text")
    expect(greeting_text).to_have_text(f"Good afternoon, {admin_account.username}!")


def test_dashboard_greeting_boundary_at_6pm(
    page: Page, server_url: str, admin_account: AdminAccount, client
) -> None:
    """Verify that greeting switches from afternoon to evening at exactly 18:00."""

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

    # Mock the time to be exactly at 6 PM (18:00)
    # Inject a script to set testGreetingHour before the page script runs
    page.add_init_script("""
        window.testGreetingHour = 18;
    """)

    # Navigate to Django dashboard
    page.goto(f"{server_url}/dashboard/")
    dashboard_heading = page.get_by_role("heading", name="Admin Dashboard")
    expect(dashboard_heading).to_be_visible()

    # Wait for the greeting JavaScript to execute
    page.wait_for_function(
        """() => {
            const elem = document.getElementById('greeting-text');
            return elem && !elem.textContent.startsWith('Welcome,');
        }""",
        timeout=5000
    )

    # At exactly 18:00, it should be "Good evening"
    greeting_text = page.locator("#greeting-text")
    expect(greeting_text).to_have_text(f"Good evening, {admin_account.username}!")
