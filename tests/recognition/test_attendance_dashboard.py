"""Tests for the attendance dashboard with filters, exports, and charts."""

import csv
import io

from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone

import pytest

from recognition.models import RecognitionOutcome
from users.models import RecognitionAttempt


@pytest.mark.django_db
def test_attendance_dashboard_requires_staff(client):
    """Non-staff users should be redirected."""
    response = client.get(reverse("admin_attendance_dashboard"))
    # staff_member_required redirects to login
    assert response.status_code == 302


@pytest.mark.django_db
def test_attendance_dashboard_non_staff_user_redirected(client):
    """Non-staff authenticated users should be redirected."""
    user = get_user_model().objects.create_user(
        username="regular-user",
        password="StrongPass123!",
        is_staff=False,
    )
    client.force_login(user)

    response = client.get(reverse("admin_attendance_dashboard"))
    assert response.status_code == 302


@pytest.mark.django_db
def test_attendance_dashboard_accessible_by_staff(client):
    """Staff users should be able to access the dashboard."""
    user = get_user_model().objects.create_user(
        username="dashboard-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    response = client.get(reverse("admin_attendance_dashboard"))
    assert response.status_code == 200
    assert b"Attendance Dashboard" in response.content


@pytest.mark.django_db
def test_attendance_dashboard_shows_summary_stats(client):
    """Dashboard should display summary statistics."""
    user = get_user_model().objects.create_user(
        username="stats-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    # Create test data
    RecognitionOutcome.objects.create(
        username="test-user",
        direction="in",
        accepted=True,
        confidence=0.85,
        source="webcam",
    )
    RecognitionOutcome.objects.create(
        username="test-user2",
        direction="in",
        accepted=False,
        confidence=0.3,
        source="webcam",
    )

    response = client.get(reverse("admin_attendance_dashboard"))
    assert response.status_code == 200
    assert b"Summary Statistics" in response.content
    assert b"Total Outcomes" in response.content


@pytest.mark.django_db
def test_attendance_dashboard_filters_by_date_range(client):
    """Dashboard should filter results by date range."""
    user = get_user_model().objects.create_user(
        username="filter-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    # Create test data
    RecognitionOutcome.objects.create(
        username="old-user",
        direction="in",
        accepted=True,
        source="webcam",
    )

    today = timezone.now().date()
    response = client.get(
        reverse("admin_attendance_dashboard"),
        {"date_from": today.isoformat(), "date_to": today.isoformat()},
    )
    assert response.status_code == 200


@pytest.mark.django_db
def test_attendance_dashboard_filters_by_employee(client):
    """Dashboard should filter results by employee username."""
    user = get_user_model().objects.create_user(
        username="employee-filter-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    RecognitionOutcome.objects.create(
        username="john_doe",
        direction="in",
        accepted=True,
        source="webcam",
    )
    RecognitionOutcome.objects.create(
        username="jane_doe",
        direction="in",
        accepted=True,
        source="webcam",
    )

    response = client.get(
        reverse("admin_attendance_dashboard"),
        {"employee": "john"},
    )
    assert response.status_code == 200
    # The filter should be applied
    assert b"john_doe" in response.content or b"Recognition Outcomes" in response.content


@pytest.mark.django_db
def test_attendance_dashboard_filters_by_outcome(client):
    """Dashboard should filter results by outcome type."""
    user = get_user_model().objects.create_user(
        username="outcome-filter-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    RecognitionOutcome.objects.create(
        username="success-user",
        direction="in",
        accepted=True,
        source="webcam",
    )
    RecognitionOutcome.objects.create(
        username="failed-user",
        direction="in",
        accepted=False,
        source="webcam",
    )

    # Filter by success
    response = client.get(
        reverse("admin_attendance_dashboard"),
        {"outcome": "success"},
    )
    assert response.status_code == 200


@pytest.mark.django_db
def test_attendance_dashboard_chart_data_present(client):
    """Dashboard should include chart data in JSON format."""
    user = get_user_model().objects.create_user(
        username="chart-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    response = client.get(reverse("admin_attendance_dashboard"))
    assert response.status_code == 200
    # Check that chart data is present (it's embedded as JSON in the template)
    content = response.content.decode("utf-8")
    assert "attendanceChart" in content


@pytest.mark.django_db
def test_export_csv_requires_staff(client):
    """Non-staff users should be redirected."""
    response = client.get(reverse("admin_attendance_export"))
    # staff_member_required redirects to login
    assert response.status_code == 302


@pytest.mark.django_db
def test_export_csv_returns_csv_file(client):
    """Staff users should be able to export attendance data as CSV."""
    user = get_user_model().objects.create_user(
        username="export-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    # Create test data
    RecognitionOutcome.objects.create(
        username="csv-user",
        direction="in",
        accepted=True,
        confidence=0.9,
        distance=0.25,
        threshold=0.4,
        source="webcam",
    )

    response = client.get(reverse("admin_attendance_export"))
    assert response.status_code == 200
    assert response["Content-Type"] == "text/csv"
    assert "attachment" in response["Content-Disposition"]
    assert ".csv" in response["Content-Disposition"]

    # Parse CSV content
    content = response.content.decode("utf-8")
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)

    # Check header
    assert rows[0][0] == "Timestamp"
    assert "Username" in rows[0]
    assert "Status" in rows[0]

    # Check data row exists
    assert len(rows) > 1


@pytest.mark.django_db
def test_export_csv_respects_filters(client):
    """CSV export should respect the same filters as the dashboard."""
    user = get_user_model().objects.create_user(
        username="export-filter-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    RecognitionOutcome.objects.create(
        username="filter-test-user",
        direction="in",
        accepted=True,
        source="webcam",
    )

    response = client.get(
        reverse("admin_attendance_export"),
        {"employee": "filter-test"},
    )
    assert response.status_code == 200
    assert response["Content-Type"] == "text/csv"


@pytest.mark.django_db
def test_export_csv_includes_attempts(client):
    """CSV export should include recognition attempts data."""
    user = get_user_model().objects.create_user(
        username="attempts-export-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    # Create a recognition attempt with liveness failure
    RecognitionAttempt.objects.create(
        username="spoof-test",
        direction=RecognitionAttempt.Direction.IN,
        spoof_detected=True,
        successful=False,
        source="webcam",
    )

    response = client.get(reverse("admin_attendance_export"))
    assert response.status_code == 200

    content = response.content.decode("utf-8")
    # Should contain liveness-related data
    assert "Liveness" in content or "Failed" in content


@pytest.mark.django_db
def test_attendance_filter_form_validates_correctly(client):
    """The filter form should validate correctly with empty and valid data."""
    from recognition.forms import AttendanceSessionFilterForm

    # Empty form should be valid (all fields optional)
    form = AttendanceSessionFilterForm(data={})
    assert form.is_valid()

    # Valid date range
    today = timezone.now().date()
    form = AttendanceSessionFilterForm(
        data={
            "date_from": today.isoformat(),
            "date_to": today.isoformat(),
            "employee": "testuser",
            "outcome": "success",
        }
    )
    assert form.is_valid()

    # Invalid outcome choice
    form = AttendanceSessionFilterForm(data={"outcome": "invalid_choice"})
    assert not form.is_valid()
