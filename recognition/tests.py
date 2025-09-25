import datetime
from contextlib import ExitStack
from typing import Callable, Tuple

import numpy as np
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.test import RequestFactory, TestCase
from unittest.mock import MagicMock, patch

from recognition import views


class AttendanceFlowTest(TestCase):
    """Ensure the attendance views handle scalar predictions correctly."""

    def setUp(self) -> None:
        self.factory = RequestFactory()
        self.user = User.objects.create_user("tester", "tester@example.com", "password")

    def _build_fake_artifacts(self) -> Tuple[Callable, MagicMock, object, object]:
        fake_face = object()

        def fake_detector(_gray_frame, _upsample: int):
            return [fake_face]

        fake_aligner = MagicMock()
        fake_aligner.align.return_value = "aligned"

        class FakeSVC:
            def predict_proba(self, _encodings):
                return np.array([[0.6, 0.4]])

        class FakeEncoder:
            labels = np.array(["alice", "bob"])

            def inverse_transform(self, indices):
                return [self.labels[index] for index in indices]

        return fake_detector, fake_aligner, FakeSVC(), FakeEncoder()

    def _exercise_attendance_view(self, view_callable, update_target: str) -> HttpResponse:
        request = self.factory.get("/attendance/")
        request.user = self.user

        fake_detector, fake_aligner, fake_svc, fake_encoder = self._build_fake_artifacts()

        fake_stream = MagicMock()
        fake_stream.start.return_value = fake_stream
        fake_stream.read.return_value = "frame"

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "recognition.views.load_recognition_artifacts",
                    return_value=(fake_detector, fake_aligner, fake_svc, fake_encoder),
                )
            )
            stack.enter_context(
                patch("recognition.views.VideoStream", return_value=fake_stream)
            )
            stack.enter_context(
                patch("recognition.views.imutils.resize", side_effect=lambda frame, width: frame)
            )
            stack.enter_context(
                patch("recognition.views.cv2.cvtColor", return_value="gray")
            )
            stack.enter_context(
                patch("recognition.views.face_utils.rect_to_bb", return_value=(0, 0, 10, 10))
            )
            stack.enter_context(
                patch("recognition.views.predict", return_value=(0, 0.95))
            )
            stack.enter_context(patch("recognition.views.cv2.rectangle"))
            stack.enter_context(patch("recognition.views.cv2.putText"))
            stack.enter_context(patch("recognition.views.cv2.imshow"))
            stack.enter_context(
                patch("recognition.views.cv2.waitKey", return_value=ord("q"))
            )
            stack.enter_context(patch("recognition.views.cv2.destroyAllWindows"))
            stack.enter_context(
                patch(
                    "recognition.views.timezone.now",
                    return_value=datetime.datetime(2020, 1, 1, 8, 0, 0),
                )
            )
            update_mock = stack.enter_context(patch(update_target))
            stack.enter_context(
                patch(
                    "recognition.views.redirect",
                    side_effect=lambda name: HttpResponse(name),
                )
            )

            response = view_callable(request)

        update_mock.assert_called_once()
        return response

    def test_mark_attendance_in_uses_scalar_prediction(self):
        response = self._exercise_attendance_view(
            views.mark_your_attendance, "recognition.views.update_attendance_in_db_in"
        )
        self.assertEqual(response.content, b"home")

    def test_mark_attendance_out_uses_scalar_prediction(self):
        response = self._exercise_attendance_view(
            views.mark_your_attendance_out, "recognition.views.update_attendance_in_db_out"
        )
        self.assertEqual(response.content, b"home")
