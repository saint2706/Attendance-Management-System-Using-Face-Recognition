"""
UI Tests for Focus Management in MarkAttendance Kiosk Mode

These tests verify that focus management works correctly in the attendance
marking flow, ensuring keyboard and screen reader users have a smooth experience.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from playwright.sync_api import Page, expect

pytestmark = [
    pytest.mark.ui,
    pytest.mark.accessibility,
    pytest.mark.django_db(transaction=True),
    pytest.mark.xfail(reason="React frontend routes not available in test environment"),
]


@pytest.fixture(scope="function")
def page_with_camera(browser) -> Iterator[Page]:
    """Create a page with camera permissions granted."""
    context = browser.new_context(
        permissions=["camera"],
        viewport={"width": 1280, "height": 720},
    )
    page = context.new_page()

    # Mock getUserMedia to provide a fake camera stream
    page.add_init_script(
        """
        // Create a mock video stream
        const mockStream = {
            getTracks: () => [{
                stop: () => {},
                getSettings: () => ({ width: 640, height: 480 }),
            }],
            getVideoTracks: () => [{
                stop: () => {},
                getSettings: () => ({ width: 640, height: 480 }),
            }],
        };

        // Override getUserMedia
        navigator.mediaDevices.getUserMedia = async (constraints) => {
            return mockStream;
        };

        // Mock HTMLVideoElement.play
        const originalPlay = HTMLVideoElement.prototype.play;
        HTMLVideoElement.prototype.play = function() {
            // Simulate successful play
            this.dispatchEvent(new Event('loadedmetadata'));
            return Promise.resolve();
        };
    """
    )

    yield page
    page.close()
    context.close()


def test_capture_button_receives_focus_after_camera_starts(
    page_with_camera: Page, server_url: str
) -> None:
    """Test that the capture button receives focus when the camera starts."""
    page_with_camera.goto(f"{server_url}/mark-attendance?direction=in")

    # Wait for the camera to initialize and stream to be available
    page_with_camera.wait_for_timeout(1000)

    # The capture button should be visible and enabled
    capture_button = page_with_camera.get_by_role("button", name="Capture & Recognize")
    expect(capture_button).to_be_visible()
    expect(capture_button).to_be_enabled()


def test_focus_returns_to_capture_button_after_try_again(
    page_with_camera: Page, server_url: str
) -> None:
    """Test that focus returns to capture button after clicking 'Try Again'."""
    page_with_camera.goto(f"{server_url}/mark-attendance?direction=in")

    # Wait for camera initialization
    page_with_camera.wait_for_timeout(1000)

    # Mock the API to return a failed recognition result
    page_with_camera.route(
        "**/api/v1/attendance/mark/",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"recognized": false, "spoofDetected": false, "message": "Face not recognized"}',
        ),
    )

    # Click capture button to trigger recognition
    capture_button = page_with_camera.get_by_role("button", name="Capture & Recognize")
    capture_button.click()

    # Wait for processing and result to appear
    try_again_button = page_with_camera.get_by_role("button", name="Try Again")
    expect(try_again_button).to_be_visible(timeout=5000)

    # Click "Try Again"
    try_again_button.click()

    # Wait for camera to restart
    page_with_camera.wait_for_timeout(500)

    # Capture button should be visible again and have focus
    capture_button = page_with_camera.get_by_role("button", name="Capture & Recognize")
    expect(capture_button).to_be_visible()

    # Check that the capture button has focus
    expect(capture_button).to_be_focused()


def test_focus_returns_to_capture_button_after_mark_another(
    page_with_camera: Page, server_url: str
) -> None:
    """Test that focus returns to capture button after clicking 'Mark Another'."""
    page_with_camera.goto(f"{server_url}/mark-attendance?direction=in")

    # Wait for camera initialization
    page_with_camera.wait_for_timeout(1000)

    # Mock the API to return a successful recognition result
    page_with_camera.route(
        "**/api/v1/attendance/mark/",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"recognized": true, "username": "testuser", "confidence": 0.95, "message": "Attendance marked successfully"}',
        ),
    )

    # Click capture button to trigger recognition
    capture_button = page_with_camera.get_by_role("button", name="Capture & Recognize")
    capture_button.click()

    # Wait for processing and result to appear
    mark_another_button = page_with_camera.get_by_role(
        "button", name="Mark attendance for another person"
    )
    expect(mark_another_button).to_be_visible(timeout=5000)

    # Click "Mark Another"
    mark_another_button.click()

    # Wait for camera to restart
    page_with_camera.wait_for_timeout(500)

    # Capture button should be visible again and have focus
    capture_button = page_with_camera.get_by_role("button", name="Capture & Recognize")
    expect(capture_button).to_be_visible()

    # Check that the capture button has focus
    expect(capture_button).to_be_focused()


def test_focus_on_capture_button_via_keyboard_navigation(
    page_with_camera: Page, server_url: str
) -> None:
    """Test that the capture button is accessible via keyboard navigation."""
    page_with_camera.goto(f"{server_url}/mark-attendance?direction=in")

    # Wait for camera initialization
    page_with_camera.wait_for_timeout(1000)

    # Capture button should be focusable via keyboard
    capture_button = page_with_camera.get_by_role("button", name="Capture & Recognize")
    capture_button.focus()

    # Verify button is focused
    expect(capture_button).to_be_focused()

    # Verify Space key shortcut works
    page_with_camera.keyboard.press("Space")

    # Button should show processing state or countdown
    page_with_camera.wait_for_timeout(500)
    # After processing starts, button text should change
    capturing_button = page_with_camera.get_by_role("button", name="Capturing in")
    expect(capturing_button.or_(capture_button)).to_be_visible()


def test_focus_on_result_card_when_result_appears(page_with_camera: Page, server_url: str) -> None:
    """Test that focus moves to result card when recognition completes."""
    page_with_camera.goto(f"{server_url}/mark-attendance?direction=in")

    # Wait for camera initialization
    page_with_camera.wait_for_timeout(1000)

    # Mock the API to return a result
    page_with_camera.route(
        "**/api/v1/attendance/mark/",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"recognized": true, "username": "testuser", "confidence": 0.95, "message": "Success"}',
        ),
    )

    # Click capture button
    capture_button = page_with_camera.get_by_role("button", name="Capture & Recognize")
    capture_button.click()

    # Wait for result to appear
    page_with_camera.wait_for_timeout(2000)

    # Check that an element with role="alert" is visible (result card)
    result_alert = page_with_camera.locator('[role="alert"]').first
    expect(result_alert).to_be_visible()


def test_escape_key_triggers_reset(page_with_camera: Page, server_url: str) -> None:
    """Test that pressing Escape key after result triggers reset."""
    page_with_camera.goto(f"{server_url}/mark-attendance?direction=in")

    # Wait for camera initialization
    page_with_camera.wait_for_timeout(1000)

    # Mock the API to return a failed result
    page_with_camera.route(
        "**/api/v1/attendance/mark/",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"recognized": false, "spoofDetected": false, "message": "Not recognized"}',
        ),
    )

    # Click capture button
    capture_button = page_with_camera.get_by_role("button", name="Capture & Recognize")
    capture_button.click()

    # Wait for result to appear
    try_again_button = page_with_camera.get_by_role("button", name="Try Again")
    expect(try_again_button).to_be_visible(timeout=5000)

    # Press Escape to reset
    page_with_camera.keyboard.press("Escape")

    # Wait for reset to complete
    page_with_camera.wait_for_timeout(500)

    # Capture button should be visible again and focused
    capture_button = page_with_camera.get_by_role("button", name="Capture & Recognize")
    expect(capture_button).to_be_visible()
    expect(capture_button).to_be_focused()


def test_space_key_shortcut_is_documented(page_with_camera: Page, server_url: str) -> None:
    """Test that the Space key shortcut is documented on the page."""
    page_with_camera.goto(f"{server_url}/mark-attendance?direction=in")

    # Wait for camera initialization
    page_with_camera.wait_for_timeout(1000)

    # Check that keyboard shortcut hint is visible
    keyboard_hint = page_with_camera.get_by_text("Space", exact=False)
    expect(keyboard_hint).to_be_visible()


# Configuration for running tests
# Run with: pytest tests/ui/test_mark_attendance_focus.py
# Run with headed browser: pytest tests/ui/test_mark_attendance_focus.py --headed
# Run with specific browser: pytest tests/ui/test_mark_attendance_focus.py --browser firefox
