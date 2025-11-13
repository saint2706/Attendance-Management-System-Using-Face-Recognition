"""Pytest configuration for UI tests.

This file contains fixtures and configuration for Playwright tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator
from uuid import uuid4

import pytest
from playwright.sync_api import Browser, sync_playwright


@dataclass(frozen=True)
class AdminAccount:
    """Container for dynamically created admin credentials used in UI tests."""

    username: str
    password: str


# Mark all tests in this module to use Django database transactions
pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args: Dict[str, Any]) -> Dict[str, Any]:
    """Configure browser context defaults for consistent viewport and locale."""

    return {
        **browser_context_args,
        "viewport": {
            "width": 1280,
            "height": 720,
        },
        "locale": "en-US",
    }


@pytest.fixture(scope="function")
def browser() -> Iterator[Browser]:
    """Create a Chromium browser instance for each test."""

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture(scope="function")
def server_url(live_server) -> str:
    """Provide the base URL for the live Django server."""

    return live_server.url


@pytest.fixture(scope="function")
def admin_account(django_user_model) -> AdminAccount:
    """Create a staff superuser for Playwright admin flows."""

    username = f"playwright-admin-{uuid4().hex[:8]}"
    password = f"PlaywrightPass-{uuid4().hex[:8]}"
    django_user_model.objects.create_superuser(
        username=username,
        email=f"{username}@example.com",
        password=password,
    )
    return AdminAccount(username=username, password=password)


# Add markers for different test categories
def pytest_configure(config):
    """Register custom markers for UI categorisation."""

    config.addinivalue_line("markers", "accessibility: Tests for accessibility features")
    config.addinivalue_line("markers", "mobile: Tests for mobile responsiveness")
    config.addinivalue_line("markers", "theme: Tests for theme toggling and dark mode")
    config.addinivalue_line("markers", "table: Tests for table enhancements")
    config.addinivalue_line("markers", "e2e: End-to-end UI workflows executed in Playwright")
