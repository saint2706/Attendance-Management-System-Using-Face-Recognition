"""
UI Tests for Mobile Navigation

These tests verify that the mobile navigation menu works correctly
on small screens and touch devices.
"""

import re

import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.ui, pytest.mark.django_db(transaction=True)]


@pytest.fixture(scope="function")
def mobile_page(browser):
    """Create a page with mobile viewport."""
    context = browser.new_context(
        viewport={"width": 375, "height": 667},  # iPhone SE size
        is_mobile=True,
        has_touch=True,
    )
    page = context.new_page()
    yield page
    context.close()


@pytest.mark.mobile
def test_mobile_menu_toggle_button_visible(mobile_page: Page, server_url: str):
    """Test that mobile menu toggle button is visible on small screens."""
    mobile_page.goto(server_url)

    # Check that mobile menu toggle is visible
    toggle_button = mobile_page.locator("#mobile-menu-toggle")
    expect(toggle_button).to_be_visible()


@pytest.mark.mobile
def test_mobile_menu_opens_on_click(mobile_page: Page, server_url: str):
    """Test that clicking the toggle button opens the mobile menu."""
    mobile_page.goto(server_url)

    # Get navigation menu
    nav_menu = mobile_page.locator("#navbar-nav")

    # Menu should not be open initially
    expect(nav_menu).not_to_have_class(re.compile(r"is-open"))

    # Click toggle button
    mobile_page.click("#mobile-menu-toggle")

    # Menu should now be open
    expect(nav_menu).to_have_class(re.compile(r"is-open"))


@pytest.mark.mobile
def test_mobile_menu_closes_on_second_click(mobile_page: Page, server_url: str):
    """Test that clicking the toggle button again closes the menu."""
    mobile_page.goto(server_url)

    nav_menu = mobile_page.locator("#navbar-nav")

    # Open menu
    mobile_page.click("#mobile-menu-toggle")
    expect(nav_menu).to_have_class(re.compile(r"is-open"))

    # Close menu
    mobile_page.click("#mobile-menu-toggle")
    expect(nav_menu).not_to_have_class(re.compile(r"is-open"))


@pytest.mark.mobile
def test_mobile_menu_closes_on_escape_key(mobile_page: Page, server_url: str):
    """Test that pressing Escape closes the mobile menu."""
    mobile_page.goto(server_url)

    nav_menu = mobile_page.locator("#navbar-nav")

    # Open menu
    mobile_page.click("#mobile-menu-toggle")
    expect(nav_menu).to_have_class(re.compile(r"is-open"))

    # Press Escape key
    mobile_page.keyboard.press("Escape")

    # Menu should be closed
    expect(nav_menu).not_to_have_class(re.compile(r"is-open"))


@pytest.mark.mobile
def test_mobile_menu_closes_on_outside_click(mobile_page: Page, server_url: str):
    """Test that clicking outside the menu closes it."""
    mobile_page.goto(server_url)

    nav_menu = mobile_page.locator("#navbar-nav")

    # Open menu
    mobile_page.click("#mobile-menu-toggle")
    expect(nav_menu).to_have_class(re.compile(r"is-open"))

    # Click outside the menu (on main content)
    mobile_page.click("#main-content")

    # Menu should be closed
    expect(nav_menu).not_to_have_class(re.compile(r"is-open"))


@pytest.mark.mobile
def test_mobile_menu_icon_changes(mobile_page: Page, server_url: str):
    """Test that the toggle icon changes between bars and X."""
    mobile_page.goto(server_url)

    icon = mobile_page.locator("#mobile-menu-toggle i")

    # Initial icon should be bars (menu closed)
    expect(icon).to_have_class(re.compile(r"fa-bars"))

    # Click to open menu
    mobile_page.click("#mobile-menu-toggle")

    # Icon should change to X (menu open)
    expect(icon).to_have_class(re.compile(r"fa-times"))

    # Click to close menu
    mobile_page.click("#mobile-menu-toggle")

    # Icon should change back to bars
    expect(icon).to_have_class(re.compile(r"fa-bars"))


@pytest.mark.mobile
@pytest.mark.accessibility
def test_mobile_menu_aria_expanded_attribute(mobile_page: Page, server_url: str):
    """Test that aria-expanded attribute is updated correctly."""
    mobile_page.goto(server_url)

    toggle_button = mobile_page.locator("#mobile-menu-toggle")

    # Initial state should be collapsed
    expect(toggle_button).to_have_attribute("aria-expanded", "false")

    # Open menu
    mobile_page.click("#mobile-menu-toggle")

    # aria-expanded should be true
    expect(toggle_button).to_have_attribute("aria-expanded", "true")

    # Close menu
    mobile_page.click("#mobile-menu-toggle")

    # aria-expanded should be false again
    expect(toggle_button).to_have_attribute("aria-expanded", "false")


@pytest.mark.mobile
def test_mobile_menu_links_are_clickable(mobile_page: Page, server_url: str):
    """Test that navigation links in mobile menu are clickable."""
    mobile_page.goto(server_url)

    # Open menu
    mobile_page.click("#mobile-menu-toggle")

    # Find and click a navigation link
    login_link = mobile_page.locator('#navbar-nav a[href*="login"]')
    expect(login_link).to_be_visible()

    # Click should navigate to login page
    login_link.click()
    mobile_page.wait_for_load_state("networkidle")

    # Verify navigation occurred
    expect(mobile_page).to_have_url(re.compile(r"login"))


# Configuration notes:
# Run mobile tests only: pytest tests/ui/test_mobile_navigation.py -m mobile
# Run with headed browser: pytest tests/ui/test_mobile_navigation.py --headed
# Run specific test: pytest tests/ui/test_mobile_navigation.py::test_mobile_menu_opens_on_click
