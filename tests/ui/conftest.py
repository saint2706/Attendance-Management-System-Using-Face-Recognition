"""
Pytest configuration for UI tests.

This file contains fixtures and configuration for Playwright tests.
"""

import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for tests."""
    return {
        **browser_context_args,
        "viewport": {
            "width": 1280,
            "height": 720,
        },
        "locale": "en-US",
    }


@pytest.fixture(scope="session")
def browser():
    """Create a browser instance for the test session."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


# Add markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "accessibility: Tests for accessibility features"
    )
    config.addinivalue_line(
        "markers", "mobile: Tests for mobile responsiveness"
    )
    config.addinivalue_line(
        "markers", "theme: Tests for theme toggling and dark mode"
    )
    config.addinivalue_line(
        "markers", "table: Tests for table enhancements"
    )
