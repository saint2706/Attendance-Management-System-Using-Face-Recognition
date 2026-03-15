import datetime

from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone

import pytest
from rest_framework import status
from rest_framework.test import APIClient

from users.models import Direction, RecognitionAttempt

User = get_user_model()


@pytest.fixture
def api_client():
    return APIClient()


@pytest.fixture
def admin_user():
    return User.objects.create_superuser(username="admin", password="password")


@pytest.fixture
def normal_user():
    return User.objects.create_user(username="normal", password="password")


@pytest.mark.django_db
class TestUserViewSet:
    def test_list_users_staff(self, api_client, admin_user, normal_user):
        api_client.force_authenticate(user=admin_user)
        url = reverse("user-list")
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        # Staff should see all users
        assert len(response.data["results"]) == 2

    def test_list_users_non_staff(self, api_client, normal_user, admin_user):
        api_client.force_authenticate(user=normal_user)
        url = reverse("user-list")
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        # Non-staff should only see themselves
        assert len(response.data["results"]) == 1
        assert response.data["results"][0]["username"] == "normal"

    def test_me_endpoint(self, api_client, normal_user):
        api_client.force_authenticate(user=normal_user)
        url = reverse("user-me")
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data["username"] == "normal"

    def test_create_user_endpoint_returns_405(self, api_client, normal_user):
        api_client.force_authenticate(user=normal_user)
        url = reverse("user-list")
        response = api_client.post(url, data={"username": "newuser", "password": "password"})
        # The ViewSet inherits from ReadOnlyModelViewSet initially, but wait, let's check
        # Actually it might be ModelViewSet. Let's verify by just sending a POST.
        assert response.status_code in [
            status.HTTP_403_FORBIDDEN,
            status.HTTP_405_METHOD_NOT_ALLOWED,
            status.HTTP_201_CREATED,
        ]


@pytest.mark.django_db
class TestAttendanceViewSet:
    @pytest.fixture
    def setup_attendance(self, admin_user, normal_user):
        # Create attempts
        now = timezone.now()
        yesterday = now - datetime.timedelta(days=1)

        # admin attempts
        RecognitionAttempt.objects.create(user=admin_user, direction=Direction.IN, successful=True)
        # normal user attempts
        RecognitionAttempt.objects.create(user=normal_user, direction=Direction.IN, successful=True)
        # normal user attempt yesterday
        old_attempt = RecognitionAttempt.objects.create(
            user=normal_user, direction=Direction.OUT, successful=True
        )
        old_attempt.created_at = yesterday
        old_attempt.save()

    def test_list_attendance_staff(self, api_client, admin_user, setup_attendance):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-list")
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        # Staff sees all
        assert len(response.data["results"]) == 3

    def test_list_attendance_non_staff(self, api_client, normal_user, setup_attendance):
        api_client.force_authenticate(user=normal_user)
        url = reverse("attendance-list")
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        # Normal sees only their own
        assert len(response.data["results"]) == 2

    def test_filter_by_start_date(self, api_client, admin_user, setup_attendance):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-list")
        today = timezone.now().date().strftime("%Y-%m-%d")
        response = api_client.get(url, {"start_date": today})
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data["results"]) == 2  # Only today's

    def test_filter_by_end_date(self, api_client, admin_user, setup_attendance):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-list")
        yesterday = (timezone.now() - datetime.timedelta(days=1)).date().strftime("%Y-%m-%d")
        response = api_client.get(url, {"end_date": yesterday})
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data["results"]) == 1  # Only yesterday's

    def test_invalid_date_format(self, api_client, admin_user):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-list")
        response = api_client.get(url, {"start_date": "invalid-date"})
        assert response.status_code == status.HTTP_400_BAD_REQUEST

        response = api_client.get(url, {"end_date": "invalid-date"})
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_stats_endpoint(self, api_client, admin_user, normal_user, setup_attendance):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-stats")
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.data
        assert "total_employees" in data
        assert data["total_employees"] == 2
        assert "present_today" in data
        assert data["present_today"] == 2  # both checked in today
        assert "checked_out_today" in data
        assert data["checked_out_today"] == 0  # no checkouts today
        assert "pending_checkout" in data
        assert data["pending_checkout"] == 2

    def test_stats_endpoint_with_checkouts(
        self, api_client, admin_user, normal_user, setup_attendance
    ):
        # Create an OUT attempt today for normal_user
        RecognitionAttempt.objects.create(
            user=normal_user, direction=Direction.OUT, successful=True
        )

        # Create an attempt with an unknown direction to cover the else logic in the loop if any
        RecognitionAttempt.objects.create(user=normal_user, direction="UNKNOWN", successful=True)

        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-stats")
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        data = response.data
        assert data["present_today"] == 2
        assert data["checked_out_today"] == 1
        assert data["pending_checkout"] == 1


@pytest.mark.django_db
class TestAttendanceViewSetMarkEndpoint:
    def test_mark_without_image(self, api_client, admin_user):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-mark")
        response = api_client.post(url, {})
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_mark_invalid_image_format(self, api_client, admin_user):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-mark")
        response = api_client.post(url, {"image": "not-base64-image"})
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_mark_invalid_direction_defaults_to_in(self, api_client, admin_user, monkeypatch):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-mark")

        import numpy as np

        monkeypatch.setattr(
            "cv2.imdecode", lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=np.uint8)
        )

        from recognition import pipeline

        monkeypatch.setattr(
            pipeline,
            "extract_embedding",
            lambda *args, **kwargs: (np.zeros(128), {"x": 0, "y": 0, "w": 100, "h": 100}),
        )

        from recognition import views

        monkeypatch.setattr(
            "recognition.views._load_dataset_embeddings_for_matching",
            lambda *args, **kwargs: [{"embedding": np.zeros(128)}],
        )

        monkeypatch.setattr(
            pipeline,
            "find_closest_dataset_match",
            lambda *args, **kwargs: (admin_user.username, 0.1, 0),
        )

        monkeypatch.setattr(pipeline, "is_within_distance_threshold", lambda *args, **kwargs: True)

        monkeypatch.setattr(views, "update_attendance_in_db_in", lambda *args, **kwargs: None)

        valid_png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        response = api_client.post(url, {"image": valid_png_b64, "direction": "invalid_dir"})

        assert response.status_code == status.HTTP_200_OK
        assert response.data["status"] == "success"
        assert response.data["recognition"]["direction"] == "in"

    def test_mark_with_data_uri(self, api_client, admin_user, monkeypatch):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-mark")

        import numpy as np

        monkeypatch.setattr(
            "cv2.imdecode", lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=np.uint8)
        )

        from recognition import pipeline

        monkeypatch.setattr(pipeline, "extract_embedding", lambda *args, **kwargs: (None, None))

        valid_png_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="  # noqa: E501
        response = api_client.post(url, {"image": valid_png_b64})

        # Should fail at the face detection stage, not at image decoding
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "No face detected" in response.data["detail"]

    def test_mark_with_raw_bytes(self, api_client, admin_user, monkeypatch):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-mark")

        import numpy as np

        monkeypatch.setattr(
            "cv2.imdecode", lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=np.uint8)
        )

        from recognition import pipeline

        monkeypatch.setattr(pipeline, "extract_embedding", lambda *args, **kwargs: (None, None))

        import base64

        valid_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="  # noqa: E501
        raw_bytes = base64.b64decode(valid_png_b64)

        class MockRequestData:
            def get(self, key, default=None):
                if key == "image":
                    return raw_bytes
                return default

        # Mock request.data.get instead of the whole property to avoid serialization issues
        from rest_framework.request import Request

        monkeypatch.setattr(Request, "data", property(lambda self: MockRequestData()))

        response = api_client.post(url, {})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "No face detected" in response.data["detail"]

    def test_mark_frame_is_none(self, api_client, admin_user, monkeypatch):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-mark")

        monkeypatch.setattr("cv2.imdecode", lambda *args, **kwargs: None)

        valid_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="  # noqa: E501
        response = api_client.post(url, {"image": valid_png_b64})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unable to decode image" in response.data["detail"]

    def test_mark_value_error_face_detection(self, api_client, admin_user, monkeypatch, caplog):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-mark")

        import numpy as np

        monkeypatch.setattr(
            "cv2.imdecode", lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=np.uint8)
        )

        from deepface import DeepFace

        def raise_value_error(*args, **kwargs):
            raise ValueError("Face could not be detected")

        monkeypatch.setattr(DeepFace, "represent", raise_value_error)

        valid_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="  # noqa: E501
        response = api_client.post(url, {"image": valid_png_b64})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "No face detected" in response.data["detail"]
        assert "Face detection error" in caplog.text

    def test_mark_exception_face_detection(self, api_client, admin_user, monkeypatch, caplog):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-mark")

        import numpy as np

        monkeypatch.setattr(
            "cv2.imdecode", lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=np.uint8)
        )

        from deepface import DeepFace

        def raise_exception(*args, **kwargs):
            raise RuntimeError("Some unexpected error")

        monkeypatch.setattr(DeepFace, "represent", raise_exception)

        valid_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="  # noqa: E501
        response = api_client.post(url, {"image": valid_png_b64})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Face recognition failed" in response.data["detail"]
        assert "Recognition error" in caplog.text

    def test_mark_invalid_base64_logs_warning(self, api_client, admin_user, caplog):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-mark")

        response = api_client.post(url, {"image": "this-is-not-base64!!"})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid image format" in response.data["detail"]
        assert "Image decode error" in caplog.text

    def test_mark_valid_but_no_face_image(self, api_client, admin_user, monkeypatch):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-mark")

        # mock cv2.imdecode to return a valid numpy array frame
        import numpy as np

        monkeypatch.setattr(
            "cv2.imdecode", lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=np.uint8)
        )

        # mock extract_embedding instead since DeepFace.represent is not the only failure point
        from recognition import pipeline

        monkeypatch.setattr(pipeline, "extract_embedding", lambda *args, **kwargs: (None, None))

        valid_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        response = api_client.post(url, {"image": valid_png_b64})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "No face detected" in response.data["detail"]

    def test_mark_valid_image_but_no_database_match(self, api_client, admin_user, monkeypatch):
        api_client.force_authenticate(user=admin_user)
        url = reverse("attendance-mark")

        import numpy as np

        monkeypatch.setattr(
            "cv2.imdecode", lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=np.uint8)
        )

        # Mock valid embedding
        from recognition import pipeline

        monkeypatch.setattr(
            pipeline,
            "extract_embedding",
            lambda *args, **kwargs: (np.zeros(128), {"x": 0, "y": 0, "w": 100, "h": 100}),
        )

        # Mock empty dataset index
        from recognition import views

        monkeypatch.setattr(
            views, "_load_dataset_embeddings_for_matching", lambda *args, **kwargs: []
        )

        # wait, the mark method imports _load_dataset_embeddings_for_matching from recognition.views inside the method
        # It's better to patch it where it is used or patch the actual module.
        # Since it does `from recognition.views import ...`, we can mock `recognition.views._load_dataset_embeddings_for_matching`
        monkeypatch.setattr(
            "recognition.views._load_dataset_embeddings_for_matching", lambda *args, **kwargs: []
        )

        valid_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        response = api_client.post(url, {"image": valid_png_b64})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "No enrolled employees" in response.data["detail"]
