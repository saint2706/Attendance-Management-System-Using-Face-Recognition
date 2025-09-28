import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

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


class AddPhotosTest(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.admin_user = User.objects.create_user(
            "admin", "admin@example.com", "password", is_staff=True
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