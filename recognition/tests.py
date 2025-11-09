"""
Tests for the recognition app.

This module contains test cases for the core functionalities of the recognition app,
including face recognition-based attendance marking, database updates, admin-only
views, and user access control.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.contrib.messages.storage.fallback import FallbackStorage
from django.test import RequestFactory, TestCase
from django.urls import reverse
from django.utils import timezone

import numpy as np

# Mock the cv2 module before it's imported by views to avoid installation in test environments
sys.modules.setdefault("cv2", MagicMock())

from recognition import views  # noqa: E402
from users.models import Present, Time  # noqa: E402


class DeepFaceAttendanceTest(TestCase):
    """Test suite for the core DeepFace attendance marking functionality."""

    def setUp(self):
        """Set up common test objects and mock filesystem."""
        self.factory = RequestFactory()
        self.user = User.objects.create_user("tester", "tester@example.com", "password")
        # Mock the training data directory
        self.db_path = Path(views.TRAINING_DATASET_ROOT)
        (self.db_path / "tester").mkdir(parents=True, exist_ok=True)

    def test_default_deepface_configuration(self):
        """Defaults should mirror the legacy DeepFace configuration."""

        self.assertEqual(views._get_face_recognition_model(), "Facenet")
        self.assertEqual(views._get_face_detection_backend(), "ssd")
        self.assertEqual(views._get_deepface_distance_metric(), "euclidean_l2")
        self.assertFalse(views._should_enforce_detection())
        self.assertTrue(views._is_liveness_enabled())

    def test_deepface_configuration_overrides(self):
        """Environment or settings overrides should cascade through helpers."""

        overrides = {
            "backend": "opencv",
            "model": "ArcFace",
            "detector_backend": "retinaface",
            "distance_metric": "cosine",
            "enforce_detection": True,
            "anti_spoofing": False,
        }

        with self.settings(DEEPFACE_OPTIMIZATIONS=overrides):
            self.assertEqual(views._get_face_recognition_model(), "ArcFace")
            self.assertEqual(views._get_face_detection_backend(), "retinaface")
            self.assertEqual(views._get_deepface_distance_metric(), "cosine")
            self.assertTrue(views._should_enforce_detection())
            self.assertFalse(views._is_liveness_enabled())

    def _setup_mocks(self, mock_manager_factory, mock_cv2, frames=None):
        """Configure common mocks for video processing to isolate view logic."""

        manager = MagicMock()
        consumer = MagicMock()
        consumer.__enter__.return_value = consumer
        consumer.__exit__.return_value = False

        if frames is None:
            frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        frame_iter = iter(frames)

        def _read(*args, **kwargs):
            try:
                frame = next(frame_iter)
            except StopIteration:
                return None
            return frame

        consumer.read.side_effect = _read
        manager.frame_consumer.return_value = consumer
        mock_manager_factory.return_value = manager

        # Simulate pressing 'q' to ensure the video loop terminates immediately
        mock_cv2.waitKey.return_value = ord("q")

        return manager, consumer

    @patch("recognition.views.DeepFace.analyze")
    def test_liveness_disabled_bypasses_analyze(self, mock_analyze):
        """Disabling anti-spoofing should skip DeepFace.analyze entirely."""

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.settings(DEEPFACE_OPTIMIZATIONS={"anti_spoofing": False}):
            self.assertTrue(views._passes_liveness_check(frame))

        mock_analyze.assert_not_called()

    @patch("recognition.views.update_attendance_in_db_in")
    @patch("recognition.views._load_dataset_embeddings_for_matching")
    @patch("recognition.views.DeepFace.represent")
    @patch("recognition.views.cv2")
    @patch("recognition.views.get_webcam_manager")
    def test_mark_attendance_in_recognizes_user(
        self,
        mock_get_manager,
        mock_cv2,
        mock_deepface_represent,
        mock_dataset_loader,
        mock_update_db,
    ):
        """Verify that a recognized user is correctly marked for check-in."""
        request = self.factory.get("/mark_attendance/")
        request.user = self.user
        self._setup_mocks(mock_get_manager, mock_cv2)

        mock_dataset_loader.return_value = [
            {
                "identity": str(self.db_path / "tester" / "1.jpg"),
                "embedding": np.array([0.1, 0.2], dtype=float),
                "username": "tester",
            }
        ]
        mock_deepface_represent.return_value = [
            {
                "embedding": [0.1, 0.2],
                "facial_area": {"x": 10, "y": 10, "w": 50, "h": 50},
            }
        ]

        # Call the view function
        views.mark_your_attendance(request)

        # Assert that the database update function was called with the correct user
        mock_update_db.assert_called_once_with({"tester": True})

    @patch("recognition.views.update_attendance_in_db_in")
    @patch("recognition.views._find_closest_dataset_match")
    @patch("recognition.views._load_dataset_embeddings_for_matching")
    @patch("recognition.views.DeepFace.represent")
    @patch("recognition.views.cv2")
    @patch("recognition.views.get_webcam_manager")
    def test_mark_attendance_respects_configured_parameters(
        self,
        mock_get_manager,
        mock_cv2,
        mock_deepface_represent,
        mock_dataset_loader,
        mock_find_match,
        mock_update_db,
    ):
        """Attendance marking should honour DeepFace configuration overrides."""

        overrides = {
            "backend": "opencv",
            "model": "SFace",
            "detector_backend": "mtcnn",
            "distance_metric": "cosine",
            "enforce_detection": True,
            "anti_spoofing": False,
        }

        request = self.factory.get("/mark_attendance/")
        request.user = self.user
        self._setup_mocks(mock_get_manager, mock_cv2)

        mock_dataset_loader.return_value = [
            {
                "identity": str(self.db_path / "tester" / "1.jpg"),
                "embedding": np.array([0.1, 0.2], dtype=float),
                "username": "tester",
            }
        ]
        mock_find_match.return_value = (
            "tester",
            0.05,
            str(self.db_path / "tester" / "1.jpg"),
        )
        mock_deepface_represent.return_value = [
            {
                "embedding": [0.1, 0.2],
                "facial_area": {"x": 10, "y": 10, "w": 50, "h": 50},
            }
        ]

        with self.settings(DEEPFACE_OPTIMIZATIONS=overrides):
            views.mark_your_attendance(request)

        mock_update_db.assert_called_once_with({"tester": True})

        _, kwargs = mock_deepface_represent.call_args
        self.assertEqual(kwargs["model_name"], "SFace")
        self.assertEqual(kwargs["detector_backend"], "mtcnn")
        self.assertTrue(kwargs["enforce_detection"])

        args, _ = mock_find_match.call_args
        self.assertEqual(args[2], "cosine")

    @patch("recognition.views.update_attendance_in_db_out")
    @patch("recognition.views._load_dataset_embeddings_for_matching")
    @patch("recognition.views.DeepFace.represent")
    @patch("recognition.views.cv2")
    @patch("recognition.views.get_webcam_manager")
    def test_mark_attendance_out_recognizes_user(
        self,
        mock_get_manager,
        mock_cv2,
        mock_deepface_represent,
        mock_dataset_loader,
        mock_update_db,
    ):
        """Verify that a recognized user is correctly marked for check-out."""
        request = self.factory.get("/mark_attendance_out/")
        request.user = self.user
        self._setup_mocks(mock_get_manager, mock_cv2)

        # Simulate DeepFace finding a user with a low distance score (high confidence)
        mock_dataset_loader.return_value = [
            {
                "identity": str(self.db_path / "tester" / "1.jpg"),
                "embedding": np.array([0.1, 0.2], dtype=float),
                "username": "tester",
            }
        ]
        mock_deepface_represent.return_value = [
            {
                "embedding": [0.1, 0.2],
                "facial_area": {"x": 10, "y": 10, "w": 50, "h": 50},
            }
        ]

        views.mark_your_attendance_out(request)

        mock_update_db.assert_called_once_with({"tester": True})

    @patch("recognition.views.update_attendance_in_db_in")
    @patch("recognition.views._load_dataset_embeddings_for_matching")
    @patch("recognition.views.DeepFace.represent")
    @patch("recognition.views.cv2")
    @patch("recognition.views.get_webcam_manager")
    def test_mark_attendance_in_ignores_low_confidence(
        self,
        mock_get_manager,
        mock_cv2,
        mock_deepface_represent,
        mock_dataset_loader,
        mock_update_db,
    ):
        """Ensure that matches with high distance (low confidence) are ignored."""
        request = self.factory.get("/mark_attendance/")
        request.user = self.user
        self._setup_mocks(mock_get_manager, mock_cv2)

        # Simulate a match with a distance score above the threshold
        mock_dataset_loader.return_value = [
            {
                "identity": str(self.db_path / "tester" / "1.jpg"),
                "embedding": np.array([5.0, 5.0], dtype=float),
                "username": "tester",
            }
        ]
        mock_deepface_represent.return_value = [
            {
                "embedding": [0.0, 0.0],
                "facial_area": {"x": 10, "y": 10, "w": 50, "h": 50},
            }
        ]

        # Temporarily set the distance threshold for this test
        with self.settings(RECOGNITION_DISTANCE_THRESHOLD=0.4):
            views.mark_your_attendance(request)

        # Verify that the database update was called with an empty dictionary
        mock_update_db.assert_called_once_with({})

    @patch("recognition.views.time.sleep", return_value=None)
    @patch("recognition.views._is_headless_environment", return_value=True)
    @patch("recognition.views.update_attendance_in_db_in")
    @patch("recognition.views._load_dataset_embeddings_for_matching")
    @patch("recognition.views.DeepFace.represent", return_value=[])
    @patch("recognition.views.cv2")
    @patch("recognition.views.get_webcam_manager")
    def test_mark_attendance_in_headless_exits(
        self,
        mock_get_manager,
        mock_cv2,
        _mock_deepface_represent,
        mock_dataset_loader,
        mock_update_db,
        _mock_headless,
        _mock_sleep,
    ):
        """Headless mode should exit automatically after a bounded number of frames."""

        request = self.factory.get("/mark_attendance/")
        request.user = self.user
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        self._setup_mocks(mock_get_manager, mock_cv2, frames=frames)

        mock_dataset_loader.return_value = [
            {
                "identity": str(self.db_path / "tester" / "1.jpg"),
                "embedding": np.array([0.0, 0.0], dtype=float),
                "username": "tester",
            }
        ]

        with self.settings(RECOGNITION_HEADLESS_ATTENDANCE_FRAMES=2):
            views.mark_your_attendance(request)

        mock_update_db.assert_called_once_with({})
        mock_cv2.imshow.assert_not_called()
        mock_cv2.waitKey.assert_not_called()

    @patch("recognition.views.time.sleep", return_value=None)
    @patch("recognition.views._is_headless_environment", return_value=True)
    @patch("recognition.views.update_attendance_in_db_out")
    @patch("recognition.views._load_dataset_embeddings_for_matching")
    @patch("recognition.views.DeepFace.represent", return_value=[])
    @patch("recognition.views.cv2")
    @patch("recognition.views.get_webcam_manager")
    def test_mark_attendance_out_headless_exits(
        self,
        mock_get_manager,
        mock_cv2,
        _mock_deepface_represent,
        mock_dataset_loader,
        mock_update_db,
        _mock_headless,
        _mock_sleep,
    ):
        """Headless mode should exit automatically for check-out as well."""

        request = self.factory.get("/mark_attendance_out/")
        request.user = self.user
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
        self._setup_mocks(mock_get_manager, mock_cv2, frames=frames)

        mock_dataset_loader.return_value = [
            {
                "identity": str(self.db_path / "tester" / "1.jpg"),
                "embedding": np.array([0.0, 0.0], dtype=float),
                "username": "tester",
            }
        ]

        with self.settings(RECOGNITION_HEADLESS_ATTENDANCE_FRAMES=1):
            views.mark_your_attendance_out(request)

        mock_update_db.assert_called_once_with({})
        mock_cv2.imshow.assert_not_called()
        mock_cv2.waitKey.assert_not_called()


class DatabaseUpdateTest(TestCase):
    """Test suite for database update functions."""

    def setUp(self):
        """Create a test user and store the current date."""
        self.user = User.objects.create_user("testuser")
        self.today = timezone.localdate()

    def test_update_attendance_in_db_in_new_present_record(self):
        """Verify that a new check-in creates both Present and Time records."""
        views.update_attendance_in_db_in({"testuser": True})
        self.assertTrue(
            Present.objects.filter(user=self.user, date=self.today, present=True).exists()
        )
        self.assertTrue(Time.objects.filter(user=self.user, date=self.today, out=False).exists())

    def test_update_attendance_in_db_out_creates_time_record(self):
        """Verify that a check-out creates a Time record with the 'out' flag."""
        views.update_attendance_in_db_out({"testuser": True})
        self.assertTrue(Time.objects.filter(user=self.user, date=self.today, out=True).exists())

    def test_update_attendance_handles_missing_user(self):
        """Ensure the system doesn't crash when trying to update a non-existent user."""
        missing_username = "ghost"
        self.assertFalse(User.objects.filter(username=missing_username).exists())

        # The function should log a warning but not raise an exception
        try:
            views.update_attendance_in_db_in({missing_username: True})
        except Exception as exc:  # pragma: no cover - fail loudly if raised
            self.fail(f"update_attendance_in_db_in raised an exception: {exc}")

        # No records should be created for the missing user
        self.assertFalse(Present.objects.filter(date=self.today).exists())
        self.assertFalse(Time.objects.filter(date=self.today).exists())


class AddPhotosTest(TestCase):
    """Test suite for the 'add_photos' view."""

    def setUp(self):
        """Set up a request factory and an admin user."""
        self.factory = RequestFactory()
        self.admin_user = User.objects.create_user(
            "manager", "manager@example.com", "password", is_staff=True
        )

    @patch("recognition.views.username_present", return_value=True)
    @patch("recognition.views.create_dataset")
    def test_add_photos_success(self, mock_create_dataset, mock_username_present):
        """Test successful photo dataset creation for an existing user."""
        request = self.factory.post("/add_photos/", {"username": "newuser"})
        request.user = self.admin_user
        # Mock Django's messaging framework for the request
        setattr(request, "session", "session")
        messages = FallbackStorage(request)
        setattr(request, "_messages", messages)

        response = views.add_photos(request)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("add-photos"))
        mock_create_dataset.assert_called_once_with("newuser")

    @patch("recognition.views.username_present", return_value=False)
    @patch("recognition.views.create_dataset")
    def test_add_photos_user_not_found(self, mock_create_dataset, mock_username_present):
        """Test that adding photos fails if the username does not exist."""
        request = self.factory.post("/add_photos/", {"username": "nonexistent"})
        request.user = self.admin_user
        setattr(request, "session", "session")
        messages = FallbackStorage(request)
        setattr(request, "_messages", messages)

        response = views.add_photos(request)

        # Should redirect to the dashboard with a warning message
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("dashboard"))
        mock_create_dataset.assert_not_called()


class AdminAccessViewsTest(TestCase):
    """Test suite for view access control based on user roles."""

    def setUp(self):
        """Create a staff user and a regular user."""
        self.staff_user = User.objects.create_user(
            "manager", "manager@example.com", "password", is_staff=True
        )
        self.regular_user = User.objects.create_user("employee", "employee@example.com", "password")

    def test_dashboard_staff_user_sees_admin_dashboard(self):
        """Verify that staff users are shown the admin dashboard."""
        self.client.force_login(self.staff_user)
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "recognition/admin_dashboard.html")

    def test_dashboard_regular_user_sees_employee_dashboard(self):
        """Verify that non-staff users are shown the employee dashboard."""
        self.client.force_login(self.regular_user)
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "recognition/employee_dashboard.html")

    def test_admin_views_allow_staff_user(self):
        """Ensure staff users can access all admin-only views."""
        self.client.force_login(self.staff_user)

        # Test access to 'add-photos' page
        response = self.client.get(reverse("add-photos"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "recognition/add_photos.html")

        # Test access to 'view-attendance-date' page
        response = self.client.get(reverse("view-attendance-date"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "recognition/view_attendance_date.html")

    def test_view_attendance_home_employee_count_excludes_admin_accounts(self):
        """Verify that the total employee count correctly excludes admins and superusers."""
        self.client.force_login(self.staff_user)

        # Create a superuser and another regular user
        User.objects.create_superuser("admin", "admin@example.com", "password")
        User.objects.create_user("employee2", "employee2@example.com", "password")

        response = self.client.get(reverse("view-attendance-home"))

        # The count should only include 'employee' and 'employee2'
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["total_num_of_emp"], 2)

        # Check another admin view for completeness
        response = self.client.get(reverse("view-attendance-employee"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "recognition/view_attendance_employee.html")

        # The obsolete 'train' view should redirect to the dashboard
        response = self.client.get(reverse("train"))
        self.assertRedirects(response, reverse("dashboard"))

    def test_admin_views_redirect_regular_user(self):
        """Ensure regular users are redirected from admin-only views."""
        self.client.force_login(self.regular_user)

        # List of admin URLs to test
        admin_urls = [
            "add-photos",
            "view-attendance-date",
            "view-attendance-employee",
            "train",
        ]

        for url_name in admin_urls:
            with self.subTest(url=url_name):
                response = self.client.get(reverse(url_name))
                # All should redirect to the 'not-authorised' page
                self.assertRedirects(response, reverse("not-authorised"))

    def test_employee_view_blocks_staff_user(self):
        """Verify that staff users cannot access employee-specific views."""
        self.client.force_login(self.staff_user)
        response = self.client.get(reverse("view-my-attendance-employee-login"))
        self.assertRedirects(response, reverse("not-authorised"))

    def test_employee_view_allows_regular_user(self):
        """Verify that regular users can access their own attendance view."""
        self.client.force_login(self.regular_user)
        response = self.client.get(reverse("view-my-attendance-employee-login"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "recognition/view_my_attendance_employee_login.html")


class DateFormISOFormatTest(TestCase):
    """Test suite to verify that forms correctly parse ISO-formatted dates."""

    def _assert_invalid_date_selection_message(self, response):
        """Helper method to verify that invalid date selection message is present."""
        self.assertEqual(response.status_code, 302)
        messages_list = list(response.wsgi_request._messages)
        self.assertTrue(
            any("Invalid date selection" in str(msg) for msg in messages_list),
            "Expected warning message about invalid date selection",
        )

    def test_date_form_accepts_iso_formatted_date(self):
        """Verify DateForm accepts ISO-formatted date string (YYYY-MM-DD)."""
        from recognition.forms import DateForm

        form_data = {"date": "2024-12-15"}
        form = DateForm(data=form_data)
        self.assertTrue(form.is_valid(), f"Form errors: {form.errors}")
        self.assertEqual(form.cleaned_data["date"].isoformat(), "2024-12-15")

    def test_username_and_date_form_accepts_iso_formatted_dates(self):
        """Verify UsernameAndDateForm accepts ISO-formatted date strings."""
        from recognition.forms import UsernameAndDateForm

        form_data = {
            "username": "testuser",
            "date_from": "2024-12-01",
            "date_to": "2024-12-31",
        }
        form = UsernameAndDateForm(data=form_data)
        self.assertTrue(form.is_valid(), f"Form errors: {form.errors}")
        self.assertEqual(form.cleaned_data["date_from"].isoformat(), "2024-12-01")
        self.assertEqual(form.cleaned_data["date_to"].isoformat(), "2024-12-31")

    def test_date_form_2_accepts_iso_formatted_dates(self):
        """Verify DateForm_2 accepts ISO-formatted date strings."""
        from recognition.forms import DateForm_2

        form_data = {
            "date_from": "2024-11-01",
            "date_to": "2024-11-30",
        }
        form = DateForm_2(data=form_data)
        self.assertTrue(form.is_valid(), f"Form errors: {form.errors}")
        self.assertEqual(form.cleaned_data["date_from"].isoformat(), "2024-11-01")
        self.assertEqual(form.cleaned_data["date_to"].isoformat(), "2024-11-30")

    def test_date_form_rejects_invalid_format(self):
        """Verify DateForm rejects dates in non-ISO format."""
        from recognition.forms import DateForm

        # Try with MM/DD/YYYY format (should fail)
        form_data = {"date": "12/15/2024"}
        form = DateForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("date", form.errors)

    def test_username_and_date_form_validates_date_order_in_view(self):
        """Verify that view_attendance_employee properly validates date ranges."""
        staff_user = User.objects.create_user(
            "admin", "admin@example.com", "password", is_staff=True
        )
        User.objects.create_user("testuser", "test@example.com", "password")

        self.client.force_login(staff_user)

        # Post with date_to before date_from (invalid)
        response = self.client.post(
            reverse("view-attendance-employee"),
            {
                "username": "testuser",
                "date_from": "2024-12-31",
                "date_to": "2024-12-01",
            },
        )

        self._assert_invalid_date_selection_message(response)

    def test_date_form_2_validates_date_order_in_view(self):
        """Verify that view_my_attendance_employee_login properly validates date ranges."""
        regular_user = User.objects.create_user("employee", "employee@example.com", "password")

        self.client.force_login(regular_user)

        # Post with date_to before date_from (invalid)
        response = self.client.post(
            reverse("view-my-attendance-employee-login"),
            {
                "date_from": "2024-12-31",
                "date_to": "2024-12-01",
            },
        )

        self._assert_invalid_date_selection_message(response)
