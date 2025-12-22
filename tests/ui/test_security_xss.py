import pytest
from playwright.sync_api import Page, expect

# Ensure we use sync playwright and transaction db
pytestmark = [pytest.mark.ui, pytest.mark.django_db(transaction=True)]


@pytest.fixture(scope="function")
def page(browser):
    """Create a new page for each test using the sync browser fixture."""
    context = browser.new_context()
    page = context.new_page()
    yield page
    page.close()
    context.close()


def test_alert_manager_xss_protection(page: Page, server_url: str):
    """
    Verify that AlertManager.show() escapes HTML in messages to prevent XSS.
    """
    # Navigate to login page which loads main.js and AlertManager
    page.goto(f"{server_url}/login/")

    # Wait for network idle to ensure scripts are loaded
    page.wait_for_load_state("networkidle")

    # This payload would execute if inserted via innerHTML
    # We use double quotes for attributes to avoid conflicts
    xss_payload = '<img src=x onerror=alert(1) class="xss-img">'

    # Trigger alert via the exposed global app instance
    # We pass the payload as an argument to avoid syntax errors
    page.evaluate("""
        (payload) => {
            if (window.AttendanceApp && window.AttendanceApp.modules.alerts) {
                window.AttendanceApp.modules.alerts.show(
                    payload, 'danger', false
                );
            } else {
                throw new Error("AttendanceApp or alerts module not found");
            }
        }
    """, xss_payload)

    # Locate the alert
    alert = page.locator(".alert.alert-danger")
    expect(alert).to_be_visible()

    # The payload should be rendered as text, so the <img> tag should NOT exist
    # in the DOM. If vulnerable, innerHTML creates the img element.
    # If fixed, textContent creates a text node containing "<img src=x...>"
    expect(alert.locator("img.xss-img")).not_to_be_visible()

    # Confirm the text is present (safe)
    expect(alert).to_contain_text(xss_payload)
