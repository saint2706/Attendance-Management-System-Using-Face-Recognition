"""Integration-style tests for the primary attendance workflows."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone

import numpy as np
import pytest

from recognition import views
from users.models import Present, Time

pytestmark = [pytest.mark.django_db, pytest.mark.attendance_flows]


def _create_admin_user() -> Any:
    """Return a staff user suitable for exercising admin-only flows."""

    return get_user_model().objects.create_user(
        username="admin", password="AdminPass!234", is_staff=True
    )


def test_admin_can_register_employee(client):
    """A staff member should be able to register a brand-new employee."""

    admin = _create_admin_user()
    client.force_login(admin)

    response = client.post(
        reverse("register"),
        data={
            "username": "employee-1",
            "password1": "StrongPassword123",
            "password2": "StrongPassword123",
        },
    )

    assert response.status_code == 302
    assert response.url == reverse("dashboard")

    new_user = get_user_model().objects.get(username="employee-1")
    assert not new_user.is_staff
    assert not new_user.is_superuser


def test_add_photos_creates_dataset_for_existing_user(client, monkeypatch):
    """Posting to the Add Photos view should trigger dataset creation."""

    admin = _create_admin_user()
    employee = get_user_model().objects.create_user(username="face-user", password="SomePass!234")
    client.force_login(admin)

    created_for: Dict[str, str] = {}

    monkeypatch.setattr(views, "username_present", lambda username: username == employee.username)

    def _fake_create_dataset(username: str) -> None:
        created_for["username"] = username

    monkeypatch.setattr(views, "create_dataset", _fake_create_dataset)

    response = client.post(reverse("add-photos"), data={"username": employee.username})

    assert response.status_code == 302
    assert response.url == reverse("add-photos")
    assert created_for["username"] == employee.username


class _StubConsumer:
    """Minimal context manager that mimics a webcam frame consumer."""

    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame

    def __enter__(self) -> "_StubConsumer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None

    def read(self, timeout: float | None = 1.0) -> np.ndarray:
        return self._frame


class _StubWebcamManager:
    """Return a single reusable frame consumer for tests."""

    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame

    def frame_consumer(self) -> _StubConsumer:
        return _StubConsumer(self._frame)


@override_settings(
    RECOGNITION_HEADLESS=True,
    RECOGNITION_HEADLESS_ATTENDANCE_FRAMES=1,
    RECOGNITION_HEADLESS_FRAME_SLEEP=0,
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
)
def test_mark_attendance_records_successful_check_in(client, django_user_model, monkeypatch):
    """Successful recognition should enqueue a check-in record for processing."""

    employee = django_user_model.objects.create_user(
        username="recognised-user", password="Password!234"
    )
    client.force_login(employee)

    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(views, "get_webcam_manager", lambda: _StubWebcamManager(dummy_frame))
    monkeypatch.setattr(views, "_is_headless_environment", lambda: True)
    monkeypatch.setattr(views.imutils, "resize", lambda frame, width: frame)
    monkeypatch.setattr(
        views,
        "_load_dataset_embeddings_for_matching",
        lambda *args, **kwargs: [
            {
                "embedding": np.array([0.1, 0.2, 0.3], dtype=float),
                "username": employee.username,
                "identity": "recognised-user/sample.jpg",
            }
        ],
    )
    monkeypatch.setattr(
        views,
        "find_closest_dataset_match",
        lambda embedding, dataset, metric: (employee.username, 0.05, "recognised-user/sample.jpg"),
    )
    monkeypatch.setattr(
        views.DeepFace,
        "represent",
        staticmethod(
            lambda **kwargs: [
                {"embedding": [0.1, 0.2, 0.3], "facial_area": {"x": 1, "y": 1, "w": 2, "h": 2}}
            ]
        ),
    )
    monkeypatch.setattr(views, "_passes_liveness_check", lambda *args, **kwargs: True)
    monkeypatch.setattr(views, "log_recognition_outcome", lambda **kwargs: None)
    monkeypatch.setattr(views.monitoring, "observe_stage_duration", lambda *args, **kwargs: None)

    captured_batches: Dict[str, Iterable[Dict[str, Any]]] = {}

    class _Result:
        id = "batch-id"

    def _capture_records(records: List[Dict[str, Any]]) -> _Result:
        captured_batches["records"] = records
        return _Result()

    monkeypatch.setattr(views, "_enqueue_attendance_records", _capture_records)

    response = client.get(reverse("mark-your-attendance"))

    assert response.status_code == 302
    assert response.url == reverse("home")

    records = list(captured_batches["records"])
    assert records[0]["direction"] == "in"
    assert records[0]["present"] == {employee.username: True}


def test_admin_can_view_attendance_by_date(client, monkeypatch):
    """Admin attendance reports should surface annotated attendance data."""

    admin = _create_admin_user()
    employee = get_user_model().objects.create_user(
        username="report-user", password="ReportPass!234"
    )
    client.force_login(admin)

    attendance_date = timezone.localdate()
    Present.objects.create(user=employee, date=attendance_date, present=True)
    Time.objects.create(user=employee, date=attendance_date, time=timezone.now(), out=False)

    def _fake_hours_vs_employee(present_qs, time_qs):
        return present_qs, "chart-url"

    monkeypatch.setattr(views, "hours_vs_employee_given_date", _fake_hours_vs_employee)

    response = client.post(
        reverse("view-attendance-date"),
        data={"date": attendance_date.strftime("%Y-%m-%d")},
    )

    assert response.status_code == 200
    assert response.context["hours_vs_employee_chart_url"] == "chart-url"
    assert response.context["qs"].count() == 1
    assert response.context["qs"].first().user == employee


def test_attendance_dashboard_shows_summary_metrics(client, monkeypatch):
    """The attendance summary view should render the aggregated metrics."""

    admin = _create_admin_user()
    client.force_login(admin)

    monkeypatch.setattr(views, "total_number_employees", lambda: 42)
    monkeypatch.setattr(views, "employees_present_today", lambda: 7)
    monkeypatch.setattr(views, "this_week_emp_count_vs_date", lambda: "this-week.png")
    monkeypatch.setattr(views, "last_week_emp_count_vs_date", lambda: "last-week.png")

    response = client.get(reverse("view-attendance-home"))

    assert response.status_code == 200
    assert response.context["total_num_of_emp"] == 42
    assert response.context["emp_present_today"] == 7
    assert response.context["this_week_graph_url"] == "this-week.png"
    assert response.context["last_week_graph_url"] == "last-week.png"
