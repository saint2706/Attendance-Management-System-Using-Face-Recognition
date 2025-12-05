"""Integration-style tests for the primary attendance workflows."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from django.contrib.auth import get_user_model
from django.contrib.messages import get_messages
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone

import numpy as np
import pytest

from recognition import views, views_legacy
from users.models import Direction, Present, Time

# Fast unit tests: admin registration, add photos (mocked), admin views
# Slow integration tests: mark attendance, full training flow (marked individually)
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

    # Mock the Celery task
    class MockAsyncResult:
        def __init__(self, username: str):
            created_for["username"] = username
            self.id = "mock-task-id"

    def _fake_capture_dataset_delay(username: str) -> MockAsyncResult:
        return MockAsyncResult(username)

    # Import and patch the tasks module where it's used
    from recognition import tasks

    monkeypatch.setattr(tasks.capture_dataset, "delay", _fake_capture_dataset_delay)

    response = client.post(reverse("add-photos"), data={"username": employee.username})

    assert response.status_code == 302
    assert "task_id=mock-task-id" in response.url
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


@pytest.mark.slow
@pytest.mark.integration
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
        lambda embedding, dataset, metric: (
            employee.username,
            0.05,
            "recognised-user/sample.jpg",
        ),
    )
    monkeypatch.setattr(
        views.DeepFace,
        "represent",
        staticmethod(
            lambda **kwargs: [
                {
                    "embedding": [0.1, 0.2, 0.3],
                    "facial_area": {"x": 1, "y": 1, "w": 2, "h": 2},
                }
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
    Time.objects.create(
        user=employee, date=attendance_date, time=timezone.now(), direction=Direction.IN
    )

    def _fake_hours_vs_employee(present_qs, time_qs):
        return present_qs, "chart-url"

    monkeypatch.setattr(views_legacy, "hours_vs_employee_given_date", _fake_hours_vs_employee)

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

    monkeypatch.setattr(views_legacy, "total_number_employees", lambda: 42)
    monkeypatch.setattr(views_legacy, "employees_present_today", lambda: 7)
    monkeypatch.setattr(views_legacy, "this_week_emp_count_vs_date", lambda: "this-week.png")
    monkeypatch.setattr(views_legacy, "last_week_emp_count_vs_date", lambda: "last-week.png")

    response = client.get(reverse("view-attendance-home"))

    assert response.status_code == 200
    assert response.context["total_num_of_emp"] == 42
    assert response.context["emp_present_today"] == 7
    assert response.context["this_week_graph_url"] == "this-week.png"
    assert response.context["last_week_graph_url"] == "last-week.png"


@pytest.mark.slow
@pytest.mark.integration
@override_settings(
    RECOGNITION_HEADLESS=True,
    RECOGNITION_HEADLESS_ATTENDANCE_FRAMES=1,
    RECOGNITION_HEADLESS_FRAME_SLEEP=0,
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
)
def test_registration_training_and_attendance_flow(client, django_user_model, monkeypatch):
    """A staff user can register, trigger training, and mark attendance successfully."""

    admin = django_user_model.objects.create_user(
        username="admin-flow", password="AdminPass!234", is_staff=True
    )
    client.force_login(admin)

    registration_response = client.post(
        reverse("register"),
        data={
            "username": "newhire",
            "password1": "StrongPassword123",
            "password2": "StrongPassword123",
        },
    )

    assert registration_response.status_code == 302
    assert registration_response.url == reverse("dashboard")

    employee = django_user_model.objects.get(username="newhire")

    from recognition import tasks

    capture_request: Dict[str, str] = {}

    class _CaptureResult:
        id = "capture-task-id"

    def _mock_capture_delay(username):
        capture_request["username"] = username
        return _CaptureResult()

    monkeypatch.setattr(
        tasks.capture_dataset,
        "delay",
        _mock_capture_delay,
    )

    add_photos_response = client.post(reverse("add-photos"), data={"username": employee.username})

    assert add_photos_response.status_code == 302
    assert capture_request["username"] == employee.username

    class _TrainResult:
        id = "train-task-id"

    train_request: Dict[str, str | None] = {}

    def _mock_train_delay(initiated_by=None):
        train_request["initiated_by"] = initiated_by
        return _TrainResult()

    monkeypatch.setattr(
        tasks.train_recognition_model,
        "delay",
        _mock_train_delay,
    )

    training_response = client.post(reverse("train"))

    assert training_response.status_code == 302
    assert train_request["initiated_by"] == admin.username

    client.logout()
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
                "identity": f"{employee.username}/sample.jpg",
            }
        ],
    )
    monkeypatch.setattr(
        views,
        "find_closest_dataset_match",
        lambda embedding, dataset, metric: (
            employee.username,
            0.05,
            f"{employee.username}/sample.jpg",
        ),
    )
    monkeypatch.setattr(
        views.DeepFace,
        "represent",
        staticmethod(
            lambda **kwargs: [
                {
                    "embedding": [0.1, 0.2, 0.3],
                    "facial_area": {"x": 1, "y": 1, "w": 2, "h": 2},
                }
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

    attendance_response = client.get(reverse("mark-your-attendance"))

    assert attendance_response.status_code == 302
    assert attendance_response.url == reverse("home")

    records = list(captured_batches["records"])
    assert records[0]["direction"] == "in"
    assert records[0]["present"] == {employee.username: True}


@pytest.mark.slow
@pytest.mark.integration
@override_settings(
    RECOGNITION_HEADLESS=True,
    RECOGNITION_HEADLESS_ATTENDANCE_FRAMES=1,
    RECOGNITION_HEADLESS_FRAME_SLEEP=0,
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
)
def test_liveness_failure_blocks_attendance(client, django_user_model, monkeypatch):
    """Spoofed faces should not be queued for attendance updates."""

    employee = django_user_model.objects.create_user(
        username="liveness-user", password="Password!234"
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
                "identity": "liveness-user/sample.jpg",
            }
        ],
    )
    monkeypatch.setattr(
        views,
        "find_closest_dataset_match",
        lambda *_args, **_kwargs: (employee.username, 0.05, "liveness-user/sample.jpg"),
    )
    monkeypatch.setattr(
        views.DeepFace,
        "represent",
        staticmethod(
            lambda **kwargs: [
                {
                    "embedding": [0.1, 0.2, 0.3],
                    "facial_area": {"x": 1, "y": 1, "w": 2, "h": 2},
                }
            ]
        ),
    )
    monkeypatch.setattr(views, "_passes_liveness_check", lambda *args, **kwargs: False)
    monkeypatch.setattr(views, "log_recognition_outcome", lambda **kwargs: None)
    monkeypatch.setattr(views.monitoring, "observe_stage_duration", lambda *args, **kwargs: None)

    def _fail_on_enqueue(_records: List[Dict[str, Any]]):
        pytest.fail("Attendance should not be enqueued when liveness fails")

    monkeypatch.setattr(views, "_enqueue_attendance_records", _fail_on_enqueue)

    response = client.get(reverse("mark-your-attendance"), follow=True)

    assert response.status_code == 200
    message_texts = [message.message for message in get_messages(response.wsgi_request)]
    assert views.LIVENESS_FAILURE_MESSAGE in message_texts


@pytest.mark.slow
@pytest.mark.integration
@override_settings(
    RECOGNITION_HEADLESS=True,
    RECOGNITION_HEADLESS_ATTENDANCE_FRAMES=1,
    RECOGNITION_HEADLESS_FRAME_SLEEP=0,
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
)
def test_unknown_face_does_not_create_attendance_records(client, django_user_model, monkeypatch):
    """High-distance matches should be ignored and not create attendance records."""

    employee = django_user_model.objects.create_user(
        username="unknown-face", password="Password!234"
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
                "username": "different-user",
                "identity": "different-user/sample.jpg",
            }
        ],
    )
    monkeypatch.setattr(
        views,
        "find_closest_dataset_match",
        lambda *_args, **_kwargs: ("different-user", 0.92, "different-user/sample.jpg"),
    )
    monkeypatch.setattr(
        views.DeepFace,
        "represent",
        staticmethod(
            lambda **kwargs: [
                {
                    "embedding": [0.1, 0.2, 0.3],
                    "facial_area": {"x": 1, "y": 1, "w": 2, "h": 2},
                }
            ]
        ),
    )
    monkeypatch.setattr(views, "_passes_liveness_check", lambda *args, **kwargs: True)
    monkeypatch.setattr(views, "log_recognition_outcome", lambda **kwargs: None)
    monkeypatch.setattr(views.monitoring, "observe_stage_duration", lambda *args, **kwargs: None)

    def _fail_on_enqueue(_records: List[Dict[str, Any]]):
        pytest.fail("High-distance matches must not enqueue attendance")

    monkeypatch.setattr(views, "_enqueue_attendance_records", _fail_on_enqueue)

    response = client.get(reverse("mark-your-attendance"))

    assert response.status_code == 302
    assert response.url == reverse("home")


@pytest.mark.slow
@pytest.mark.integration
@override_settings(
    RECOGNITION_HEADLESS=True,
    RECOGNITION_HEADLESS_ATTENDANCE_FRAMES=1,
    RECOGNITION_HEADLESS_FRAME_SLEEP=0,
)
def test_missing_training_data_short_circuits_attendance(client, django_user_model, monkeypatch):
    """The attendance flow should guide users when no encrypted dataset is present."""

    employee = django_user_model.objects.create_user(username="no-dataset", password="Password!234")
    client.force_login(employee)

    monkeypatch.setattr(
        views,
        "get_webcam_manager",
        lambda: _StubWebcamManager(np.zeros((1, 1, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(views, "_load_dataset_embeddings_for_matching", lambda *args, **kwargs: [])

    def _fail_on_enqueue(_records: List[Dict[str, Any]]):
        pytest.fail("No attendance should be enqueued when dataset is missing")

    monkeypatch.setattr(views, "_enqueue_attendance_records", _fail_on_enqueue)

    response = client.get(reverse("mark-your-attendance"), follow=True)

    assert response.status_code == 200
    messages = [message.message for message in get_messages(response.wsgi_request)]
    assert any("No encrypted training data" in message for message in messages)
