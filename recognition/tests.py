import datetime
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.modules.setdefault("cv2", MagicMock())

import numpy as np
import pandas as pd
from django.contrib.auth.models import User
from django.contrib.messages.storage.fallback import FallbackStorage
from django.test import RequestFactory, TestCase
from django.urls import reverse
from django.utils import timezone

from recognition import views
from users.models import Present, Time


class DeepFaceAttendanceTest(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_user("tester", "tester@example.com", "password")
        self.db_path = Path(views.TRAINING_DATASET_ROOT)
        (self.db_path / "tester").mkdir(parents=True, exist_ok=True)

    def _setup_mocks(self, mock_videostream, mock_cv2):
        """Configure common mocks for video processing."""
        mock_stream = MagicMock()
        mock_stream.start.return_value = mock_stream
        mock_stream.read.return_value = np.zeros(
            (480, 640, 3), dtype=np.uint8
        )  # Return a mock image
        mock_videostream.return_value = mock_stream

        mock_cv2.waitKey.return_value = ord("q")  # Ensure the loop terminates

    @patch("recognition.views.update_attendance_in_db_in")
    @patch("recognition.views.DeepFace.find")
    @patch("recognition.views.cv2")
    @patch("recognition.views.VideoStream")
    def test_mark_attendance_in_recognizes_user(
        self, mock_videostream, mock_cv2, mock_deepface_find, mock_update_db
    ):
        request = self.factory.get("/mark_attendance/")
        self._setup_mocks(mock_videostream, mock_cv2)

        # Simulate DeepFace finding a user
        mock_df = pd.DataFrame(
            [
                {
                    "identity": str(self.db_path / "tester" / "1.jpg"),
                    "source_x": 10,
                    "source_y": 10,
                    "source_w": 50,
                    "source_h": 50,
                }
            ]
        )
        mock_deepface_find.return_value = [mock_df]

        views.mark_your_attendance(request)

        mock_update_db.assert_called_once_with({"tester": True})

    @patch("recognition.views.update_attendance_in_db_out")
    @patch("recognition.views.DeepFace.find")
    @patch("recognition.views.cv2")
    @patch("recognition.views.VideoStream")
    def test_mark_attendance_out_recognizes_user(
        self, mock_videostream, mock_cv2, mock_deepface_find, mock_update_db
    ):
        request = self.factory.get("/mark_attendance_out/")
        self._setup_mocks(mock_videostream, mock_cv2)

        # Simulate DeepFace finding a user
        mock_df = pd.DataFrame(
            [
                {
                    "identity": str(self.db_path / "tester" / "1.jpg"),
                    "source_x": 10,
                    "source_y": 10,
                    "source_w": 50,
                    "source_h": 50,
                }
            ]
        )
        mock_deepface_find.return_value = [mock_df]

        views.mark_your_attendance_out(request)

        mock_update_db.assert_called_once_with({"tester": True})


class DatabaseUpdateTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("testuser")
        self.today = timezone.localdate()

    def test_update_attendance_in_db_in_new_present_record(self):
        views.update_attendance_in_db_in({"testuser": True})
        self.assertTrue(
            Present.objects.filter(user=self.user, date=self.today, present=True).exists()
        )
        self.assertTrue(
            Time.objects.filter(user=self.user, date=self.today, out=False).exists()
        )

    def test_update_attendance_in_db_out_creates_time_record(self):
        views.update_attendance_in_db_out({"testuser": True})
        self.assertTrue(
            Time.objects.filter(user=self.user, date=self.today, out=True).exists()
        )

    def test_update_attendance_handles_missing_user(self):
        missing_username = "ghost"
        self.assertFalse(User.objects.filter(username=missing_username).exists())

        try:
            views.update_attendance_in_db_in({missing_username: True})
        except Exception as exc:  # pragma: no cover - fail loudly if raised
            self.fail(f"update_attendance_in_db_in raised an exception: {exc}")

        self.assertFalse(Present.objects.filter(date=self.today).exists())
        self.assertFalse(Time.objects.filter(date=self.today).exists())


class AddPhotosTest(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.admin_user = User.objects.create_user(
            "manager", "manager@example.com", "password", is_staff=True
        )

    @patch("recognition.views.username_present", return_value=True)
    @patch("recognition.views.create_dataset")
    def test_add_photos_success(self, mock_create_dataset, mock_username_present):
        request = self.factory.post("/add_photos/", {"username": "newuser"})
        request.user = self.admin_user
        setattr(request, "session", "session")
        messages = FallbackStorage(request)
        setattr(request, "_messages", messages)

        response = views.add_photos(request)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("add-photos"))
        mock_create_dataset.assert_called_once_with("newuser")

    @patch("recognition.views.username_present", return_value=False)
    @patch("recognition.views.create_dataset")
    def test_add_photos_user_not_found(
        self, mock_create_dataset, mock_username_present
    ):
        request = self.factory.post("/add_photos/", {"username": "nonexistent"})
        request.user = self.admin_user
        setattr(request, "session", "session")
        messages = FallbackStorage(request)
        setattr(request, "_messages", messages)

        response = views.add_photos(request)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("dashboard"))
        mock_create_dataset.assert_not_called()


class AdminAccessViewsTest(TestCase):
    def setUp(self):
        self.staff_user = User.objects.create_user(
            "manager", "manager@example.com", "password", is_staff=True
        )
        self.regular_user = User.objects.create_user(
            "employee", "employee@example.com", "password"
        )

    def test_dashboard_staff_user_sees_admin_dashboard(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "recognition/admin_dashboard.html")

    def test_dashboard_regular_user_sees_employee_dashboard(self):
        self.client.force_login(self.regular_user)
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "recognition/employee_dashboard.html")

    def test_admin_views_allow_staff_user(self):
        self.client.force_login(self.staff_user)

        response = self.client.get(reverse("add-photos"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "recognition/add_photos.html")

        response = self.client.get(reverse("view-attendance-date"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response, "recognition/view_attendance_date.html"
        )

        response = self.client.get(reverse("view-attendance-employee"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response, "recognition/view_attendance_employee.html"
        )

        response = self.client.get(reverse("train"))
        self.assertRedirects(response, reverse("dashboard"))

    def test_admin_views_redirect_regular_user(self):
        self.client.force_login(self.regular_user)

        response = self.client.get(reverse("add-photos"))
        self.assertRedirects(response, reverse("not-authorised"))

        response = self.client.get(reverse("view-attendance-date"))
        self.assertRedirects(response, reverse("not-authorised"))

        response = self.client.get(reverse("view-attendance-employee"))
        self.assertRedirects(response, reverse("not-authorised"))

        response = self.client.get(reverse("train"))
        self.assertRedirects(response, reverse("not-authorised"))

    def test_employee_view_blocks_staff_user(self):
        self.client.force_login(self.staff_user)
        response = self.client.get(reverse("view-my-attendance-employee-login"))
        self.assertRedirects(response, reverse("not-authorised"))

    def test_employee_view_allows_regular_user(self):
        self.client.force_login(self.regular_user)
        response = self.client.get(reverse("view-my-attendance-employee-login"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response, "recognition/view_my_attendance_employee_login.html"
        )
