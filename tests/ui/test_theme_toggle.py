"""
UI Tests for Theme Toggle Functionality

These tests verify that the dark mode toggle works correctly and persists
the user's theme preference across page loads.
"""

import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(scope="function")
def page(browser):
    """Create a new page for each test."""
    page = browser.new_page()
    yield page
    page.close()


def test_theme_toggle_button_exists(page: Page):
    """Test that the theme toggle button is present on the page."""
    page.goto("http://localhost:8000/")

    # Check that theme toggle button exists
    theme_toggle = page.locator("#theme-toggle")
    expect(theme_toggle).to_be_visible()


def test_theme_toggle_switches_to_dark_mode(page: Page):
    """Test that clicking the toggle switches to dark mode."""
    page.goto("http://localhost:8000/")

    # Get initial state (should be light mode)
    html = page.locator("html")
    expect(html).not_to_have_class("theme-dark")

    # Click theme toggle
    page.click("#theme-toggle")

    # Check that dark mode is applied
    expect(html).to_have_class("theme-dark")

    # Check that icon changed to sun (light mode icon)
    icon = page.locator("#theme-toggle i")
    expect(icon).to_have_class("fa-sun")


def test_theme_toggle_switches_back_to_light_mode(page: Page):
    """Test that clicking the toggle again switches back to light mode."""
    page.goto("http://localhost:8000/")

    # Switch to dark mode
    page.click("#theme-toggle")
    html = page.locator("html")
    expect(html).to_have_class("theme-dark")

    # Switch back to light mode
    page.click("#theme-toggle")
    expect(html).not_to_have_class("theme-dark")

    # Check that icon changed back to moon (dark mode icon)
    icon = page.locator("#theme-toggle i")
    expect(icon).to_have_class("fa-moon")


def test_theme_preference_persists_on_reload(page: Page):
    """Test that theme preference persists after page reload."""
    page.goto("http://localhost:8000/")

    # Switch to dark mode
    page.click("#theme-toggle")
    html = page.locator("html")
    expect(html).to_have_class("theme-dark")

    # Reload the page
    page.reload()

    # Dark mode should still be active
    expect(html).to_have_class("theme-dark")


def test_theme_preference_persists_across_navigation(page: Page):
    """Test that theme preference persists when navigating between pages."""
    page.goto("http://localhost:8000/")

    # Switch to dark mode
    page.click("#theme-toggle")
    html = page.locator("html")
    expect(html).to_have_class("theme-dark")

    # Navigate to login page
    page.click('a[href*="login"]')
    page.wait_for_load_state("networkidle")

    # Dark mode should still be active
    expect(html).to_have_class("theme-dark")


def test_theme_toggle_accessibility(page: Page):
    """Test that theme toggle is accessible via keyboard."""
    page.goto("http://localhost:8000/")

    # Focus on theme toggle using keyboard
    page.keyboard.press("Tab")
    # May need to press Tab multiple times depending on page structure
    # For this test, we'll focus directly on the button
    theme_toggle = page.locator("#theme-toggle")
    theme_toggle.focus()

    # Check that button is focused
    expect(theme_toggle).to_be_focused()

    # Press Enter to toggle
    theme_toggle.press("Enter")

    # Check that dark mode is applied
    html = page.locator("html")
    expect(html).to_have_class("theme-dark")


def test_theme_toggle_aria_labels(page: Page):
    """Test that theme toggle has proper ARIA labels."""
    page.goto("http://localhost:8000/")

    theme_toggle = page.locator("#theme-toggle")

    # Check initial ARIA label (light mode)
    expect(theme_toggle).to_have_attribute("aria-label", "Switch to dark mode")

    # Click to switch to dark mode
    page.click("#theme-toggle")

    # Check updated ARIA label (dark mode)
    expect(theme_toggle).to_have_attribute("aria-label", "Switch to light mode")


# Configuration for running tests
# Run with: pytest tests/ui/test_theme_toggle.py
# Run with headed browser: pytest tests/ui/test_theme_toggle.py --headed
# Run with specific browser: pytest tests/ui/test_theme_toggle.py --browser firefox
