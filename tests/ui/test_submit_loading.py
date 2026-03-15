"""
UI Tests for Form Loading State
"""

import re

import pytest
from playwright.sync_api import Page, expect

# Use the same markers as other UI tests
pytestmark = [pytest.mark.ui, pytest.mark.django_db(transaction=True)]


@pytest.fixture(scope="function")
def page(browser):
    """Create a new page for each test."""
    page = browser.new_page()
    yield page
    page.close()


def test_submit_loading_state(page: Page, server_url: str):
    """Test that submitting a form shows the loading spinner on the button."""

    # Navigate to login page
    page.goto(f"{server_url}/login/")

    # Fill form with dummy data to ensure checkValidity() passes
    page.locator("input[name='username']").fill("testuser")
    page.locator("input[name='password']").fill("password")

    # Add a submit listener that prevents default to stop navigation
    # This runs after the application's listener (which sets loading state)
    # ensuring the UI updates but the page doesn't reload.
    page.evaluate(
        """
        const form = document.querySelector('form');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
        });
    """
    )

    # Click submit
    submit_btn = page.locator("button[type='submit']")
    submit_btn.click()

    # Check that button is disabled
    expect(submit_btn).to_be_disabled()

    # Check that button has the spinner icon
    icon = submit_btn.locator("i")
    expect(icon).to_have_class(re.compile(r"fa-spinner"))
    expect(icon).to_have_class(re.compile(r"fa-spin"))
