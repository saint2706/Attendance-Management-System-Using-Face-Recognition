"""Integration-oriented tests for the face recognition workflow."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from django.test import RequestFactory, override_settings

import django

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings"
)
django.setup()

from django.contrib.auth import get_user_model
from recognition import tasks
from recognition import views as recognition_views
from recognition import webcam_manager as webcam_module
from users.models import RecognitionAttempt

User = get_user_model()


@pytest.mark.django_db
class TestFaceRecognitionWorkflow:
    """Validate the most important steps of the face recognition pipeline."""

    def setup_method(self) -> None:
        self.factory = RequestFactory()

    def teardown_method(self) -> None:
        webcam_module.reset_webcam_manager()

    def _make_request(self) -> Callable[[], object]:
        """Return a callable creating POST requests for the API view."""

        def _builder() -> object:
            payload = json.dumps({"embedding": [0.1, 0.2, 0.3]})
            return self.factory.post(
                "/api/face-recognition/",
                data=payload,
                content_type="application/json",
            )

        return _builder

    def test_face_encoding_generation(self, monkeypatch):
        """Computed face encodings should be NumPy arrays with float64 dtype."""

        expected_embedding = [0.5, 0.25, 0.75]
        monkeypatch.setattr(
            tasks,
            "_get_or_compute_cached_embedding",
            lambda *args, **kwargs: expected_embedding,
        )

        encoding = tasks.compute_face_encoding(Path("/tmp/fake-image.jpg"))

        assert isinstance(encoding, np.ndarray)
        assert encoding.dtype == np.float64
        assert np.array_equal(encoding, np.array(expected_embedding, dtype=np.float64))

    @override_settings(
        RECOGNITION_DISTANCE_THRESHOLD=0.5,
        DEEPFACE_OPTIMIZATIONS={
            "distance_metric": "euclidean_l2",
            "model": "Facenet",
            "detector_backend": "opencv",
            "enforce_detection": False,
            "anti_spoofing": False,
        },
    )
    def test_attendance_marking_accuracy(self, monkeypatch) -> None:
        """The API should flip the recognition flag around the distance threshold."""

        dataset_entry = {
            "embedding": np.array([0.1, 0.2, 0.3], dtype=float),
            "username": "jane",
            "identity": "jane/reference.jpg",
        }
        monkeypatch.setattr(
            recognition_views,
            "_load_dataset_embeddings_for_matching",
            lambda *args, **kwargs: [dataset_entry],
        )
        distances = iter([0.3, 0.7])

        def _fake_find(*_args, **_kwargs):
            return ("jane", next(distances), "jane/reference.jpg")

        monkeypatch.setattr(
            recognition_views,
            "_find_closest_dataset_match",
            _fake_find,
        )
        monkeypatch.setattr(
            recognition_views,
            "_passes_liveness_check",
            lambda *args, **kwargs: True,
        )

        view = recognition_views.FaceRecognitionAPI.as_view()
        request_factory = self._make_request()

        first_response = view(request_factory())
        second_response = view(request_factory())

        assert first_response.status_code == 200
        assert second_response.status_code == 200

        first_payload = json.loads(first_response.content.decode("utf-8"))
        second_payload = json.loads(second_response.content.decode("utf-8"))

        assert first_payload["recognized"] is True
        assert second_payload["recognized"] is False

        attempts = list(RecognitionAttempt.objects.order_by("id"))
        assert len(attempts) == 2
        assert attempts[0].successful is True
        assert attempts[1].successful is False
        assert attempts[0].direction == RecognitionAttempt.Direction.IN

    @override_settings(
        RECOGNITION_DISTANCE_THRESHOLD=0.5,
        DEEPFACE_OPTIMIZATIONS={
            "distance_metric": "euclidean_l2",
            "model": "Facenet",
            "detector_backend": "opencv",
            "enforce_detection": False,
            "anti_spoofing": False,
        },
    )
    def test_api_logs_attempt_metadata(self, monkeypatch) -> None:
        """Recognition attempts logged via the API should capture metadata."""

        user = User.objects.create_user(username="jane", password="testpass")

        dataset_entry = {
            "embedding": np.array([0.1, 0.2, 0.3], dtype=float),
            "username": "jane",
            "identity": "jane/reference.jpg",
        }
        monkeypatch.setattr(
            recognition_views,
            "_load_dataset_embeddings_for_matching",
            lambda *args, **kwargs: [dataset_entry],
        )

        monkeypatch.setattr(
            recognition_views,
            "_find_closest_dataset_match",
            lambda *_args, **_kwargs: ("jane", 0.25, "jane/reference.jpg"),
        )
        monkeypatch.setattr(
            recognition_views,
            "_passes_liveness_check",
            lambda *args, **kwargs: True,
        )

        view = recognition_views.FaceRecognitionAPI.as_view()
        request = self._make_request()()
        request.META["HTTP_X_RECOGNITION_SITE"] = "hq"

        response = view(request)
        assert response.status_code == 200

        attempt = RecognitionAttempt.objects.get()
        assert attempt.successful is True
        assert attempt.username == "jane"
        assert attempt.user == user
        assert attempt.site == "hq"
        assert attempt.direction == RecognitionAttempt.Direction.IN
        assert attempt.source == "api"

    def test_concurrent_attendance_requests(self, monkeypatch):
        """Concurrent consumers should receive advancing frames and clean up."""

        class _DeterministicStream:
            def __init__(self, *args, **kwargs):
                self._counter = 0

            def start(self):
                return self

            def read(self):
                self._counter += 1
                return np.full((2, 2, 3), self._counter, dtype=np.uint8)

            def stop(self):
                pass

        monkeypatch.setattr(webcam_module, "VideoStream", _DeterministicStream)
        monkeypatch.setattr(webcam_module.time, "sleep", lambda *_args, **_kwargs: None)

        manager = webcam_module.get_webcam_manager()
        manager._warmup_time = 0.0

        results: list[list[int]] = []

        def _consume_frames() -> None:
            local_values: list[int] = []
            with manager.frame_consumer() as consumer:
                for _ in range(3):
                    frame = consumer.read(timeout=0.5)
                    assert frame is not None
                    local_values.append(int(frame[0, 0, 0]))
            results.append(local_values)

        threads = [threading.Thread(target=_consume_frames) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results, "Consumers should have recorded frame ids."
        for sequence in results:
            assert sequence == sorted(sequence)
            assert len(sequence) == len(set(sequence))
        assert manager._consumer_count == 0

    def test_camera_initialization(self, monkeypatch):
        """The shared webcam manager should only start the stream once per lifetime."""

        start_calls = []
        stop_calls = []

        class _InspectableStream:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                start_calls.append("start")
                return self

            def read(self):
                return np.zeros((2, 2, 3), dtype=np.uint8)

            def stop(self):
                stop_calls.append("stop")

        monkeypatch.setattr(webcam_module, "VideoStream", _InspectableStream)
        monkeypatch.setattr(webcam_module.time, "sleep", lambda *_args, **_kwargs: None)

        manager = webcam_module.get_webcam_manager()
        manager._warmup_time = 0.0

        manager.start()
        manager.start()
        assert start_calls == ["start"], "Video stream should only be started once."

        manager.shutdown()
        assert stop_calls == ["stop"], "Video stream should be stopped during shutdown."
