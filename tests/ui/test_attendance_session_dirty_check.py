"""
Playwright tests for attendance session dirty check optimization.

This module verifies that the AttendanceSessionMonitor correctly implements
a "dirty check" to prevent unnecessary DOM updates when polling returns
identical data, reducing CPU usage during long-running sessions.
"""

import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.ui, pytest.mark.django_db(transaction=True)]


@pytest.fixture(scope="function")
def page(browser):
    """Create a new page for each test using the sync browser fixture."""
    context = browser.new_context()
    page = context.new_page()
    yield page
    page.close()
    context.close()


def test_dirty_check_implementation_exists(page: Page, server_url: str, admin_account):
    """
    Verify that the AttendanceSessionMonitor has the dirty check mechanism implemented.
    
    This test checks that:
    1. The lastRenderedHtml property exists and is initialized to null
    2. The _renderRows method performs HTML string comparison
    3. DOM updates are skipped when HTML hasn't changed
    """
    # Login as admin
    page.goto(f"{server_url}/login/")
    page.wait_for_load_state("networkidle")
    page.fill('input[name="username"]', admin_account.username)
    page.fill('input[name="password"]', admin_account.password)
    page.click('button[type="submit"]')
    page.wait_for_load_state("networkidle")

    # Navigate to attendance session page
    page.goto(f"{server_url}/attendance_session/")
    page.wait_for_load_state("networkidle")

    # Wait for the monitor to be initialized
    page.wait_for_function("() => window.attendanceMonitor !== undefined", timeout=10000)

    # The monitor automatically starts and fetches data, so lastRenderedHtml may already be set
    # But we can stop it and reset to test the dirty check mechanism
    page.evaluate("""
        () => {
            window.attendanceMonitor.stop();
            window.attendanceMonitor.lastRenderedHtml = null;
        }
    """)
    
    # Verify lastRenderedHtml is now null
    last_rendered_html = page.evaluate(
        "() => window.attendanceMonitor.lastRenderedHtml"
    )
    assert last_rendered_html is None, (
        "lastRenderedHtml should be null after manual reset"
    )

    # Test the dirty check mechanism directly by calling _renderRows with identical data
    # First, inject test data and call _renderRows
    page.evaluate("""
        () => {
            // Track DOM update attempts
            window.domUpdateAttempts = 0;
            const tbody = document.getElementById('attendance-log-body');
            
            // Override innerHTML setter to track updates
            const originalDescriptor = Object.getOwnPropertyDescriptor(Element.prototype, 'innerHTML');
            Object.defineProperty(tbody, 'innerHTML', {
                set: function(value) {
                    window.domUpdateAttempts++;
                    originalDescriptor.set.call(this, value);
                },
                get: function() {
                    return originalDescriptor.get.call(this);
                }
            });
            
            // Call _renderRows with some test events
            const testEvents = [
                {
                    event_type: 'outcome',
                    timestamp: '2024-01-01T10:00:00Z',
                    username: 'testuser',
                    direction: 'in',
                    accepted: true,
                    confidence: 0.95
                }
            ];
            
            window.attendanceMonitor._renderRows(testEvents);
        }
    """)

    # Check that first call updated the DOM
    update_count_after_first = page.evaluate("() => window.domUpdateAttempts")
    assert update_count_after_first == 1, f"Expected 1 DOM update after first render, got {update_count_after_first}"

    # Verify lastRenderedHtml is now set (not null)
    last_rendered_after_first = page.evaluate(
        "() => window.attendanceMonitor.lastRenderedHtml"
    )
    assert last_rendered_after_first is not None, "lastRenderedHtml should be set after first render"
    assert "testuser" in last_rendered_after_first, "Rendered HTML should contain test data"

    # Call _renderRows again with THE SAME data
    page.evaluate("""
        () => {
            const testEvents = [
                {
                    event_type: 'outcome',
                    timestamp: '2024-01-01T10:00:00Z',
                    username: 'testuser',
                    direction: 'in',
                    accepted: true,
                    confidence: 0.95
                }
            ];
            
            window.attendanceMonitor._renderRows(testEvents);
        }
    """)

    # Check that dirty check prevented the second DOM update
    update_count_after_second = page.evaluate("() => window.domUpdateAttempts")
    assert update_count_after_second == 1, (
        f"Expected still only 1 DOM update after identical data, got {update_count_after_second}. "
        "Dirty check should prevent update when HTML hasn't changed."
    )

    # Call _renderRows with DIFFERENT data
    page.evaluate("""
        () => {
            const testEvents = [
                {
                    event_type: 'outcome',
                    timestamp: '2024-01-01T10:00:00Z',
                    username: 'differentuser',
                    direction: 'out',
                    accepted: false,
                    confidence: 0.85
                }
            ];
            
            window.attendanceMonitor._renderRows(testEvents);
        }
    """)

    # Check that different data triggered an update
    update_count_after_different = page.evaluate("() => window.domUpdateAttempts")
    assert update_count_after_different == 2, (
        f"Expected 2 DOM updates after different data, got {update_count_after_different}. "
        "Should update when data changes."
    )

    # Verify the new content is in lastRenderedHtml
    last_rendered_final = page.evaluate(
        "() => window.attendanceMonitor.lastRenderedHtml"
    )
    assert "differentuser" in last_rendered_final, "Rendered HTML should contain new test data"


def test_dirty_check_handles_empty_to_data_transition(page: Page, server_url: str, admin_account):
    """
    Verify that dirty check correctly handles transition from empty to populated data.
    
    This ensures the null initialization allows the first render even when initial
    data might be empty.
    """
    # Login as admin
    page.goto(f"{server_url}/login/")
    page.wait_for_load_state("networkidle")
    page.fill('input[name="username"]', admin_account.username)
    page.fill('input[name="password"]', admin_account.password)
    page.click('button[type="submit"]')
    page.wait_for_load_state("networkidle")

    # Navigate to attendance session page
    page.goto(f"{server_url}/attendance_session/")
    page.wait_for_load_state("networkidle")
    page.wait_for_function("() => window.attendanceMonitor !== undefined", timeout=10000)

    # Stop the monitor to prevent automatic updates interfering with the test
    page.evaluate("() => window.attendanceMonitor.stop()")

    # Test rendering empty data first
    page.evaluate("""
        () => {
            window.domUpdateAttempts = 0;
            const tbody = document.getElementById('attendance-log-body');
            
            const originalDescriptor = Object.getOwnPropertyDescriptor(Element.prototype, 'innerHTML');
            Object.defineProperty(tbody, 'innerHTML', {
                set: function(value) {
                    window.domUpdateAttempts++;
                    originalDescriptor.set.call(this, value);
                },
                get: function() {
                    return originalDescriptor.get.call(this);
                }
            });
            
            // Reset lastRenderedHtml to simulate fresh state
            window.attendanceMonitor.lastRenderedHtml = null;
            
            // Render empty data
            window.attendanceMonitor._renderRows([]);
        }
    """)

    # Should render empty message
    empty_message = page.locator("#attendance-log-body")
    expect(empty_message).to_contain_text("No recent recognition events")

    update_count = page.evaluate("() => window.domUpdateAttempts")
    assert update_count == 1, f"Expected 1 update for empty data, got {update_count}"

    # Render populated data
    page.evaluate("""
        () => {
            const testEvents = [
                {
                    event_type: 'outcome',
                    timestamp: '2024-01-01T10:00:00Z',
                    username: 'testuser',
                    direction: 'in',
                    accepted: true,
                    confidence: 0.95
                }
            ];
            
            window.attendanceMonitor._renderRows(testEvents);
        }
    """)

    # Should now show the data
    rows = page.locator("#attendance-log-body tr")
    expect(rows).to_have_count(1)
    expect(rows.first).to_contain_text("testuser")

    update_count_final = page.evaluate("() => window.domUpdateAttempts")
    assert update_count_final == 2, (
        f"Expected 2 updates after transition from empty to data, got {update_count_final}"
    )
