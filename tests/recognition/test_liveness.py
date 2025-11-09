import os
import sys
from types import ModuleType, SimpleNamespace

import django
from django.contrib.messages.storage.fallback import FallbackStorage
from django.test import RequestFactory

import numpy as np
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")
django.setup()

_fake_cv2 = ModuleType("cv2")
_fake_cv2.FONT_HERSHEY_SIMPLEX = 0
_fake_cv2.INTER_AREA = 0
_fake_cv2.MORPH_RECT = 0
_fake_cv2.rectangle = lambda *args, **kwargs: None
_fake_cv2.putText = lambda *args, **kwargs: None
_fake_cv2.resize = lambda image, dim, interpolation=None: image
_fake_cv2.imshow = lambda *args, **kwargs: None
_fake_cv2.waitKey = lambda *args, **kwargs: 0
_fake_cv2.destroyAllWindows = lambda *args, **kwargs: None
_fake_cv2.__getattr__ = lambda name: 0
sys.modules.setdefault("cv2", _fake_cv2)

from recognition import views  # noqa: E402


class _DummyStream:
    def __init__(self, frame):
        self._frame = frame

    def start(self):
        return self

    def read(self):
        return self._frame

    def stop(self):
        pass


def _build_request():
    request = RequestFactory().get("/")
    request.session = {}
    request.user = SimpleNamespace(is_authenticated=True)
    storage = FallbackStorage(request)
    setattr(request, "_messages", storage)
    return request, storage


def test_evaluate_match_blocks_spoof(monkeypatch):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    match = pd.Series(
        {
            "identity": "/tmp/train/alice/1.jpg",
            "distance": 0.2,
            "source_x": 5,
            "source_y": 6,
            "source_w": 20,
            "source_h": 20,
        }
    )

    monkeypatch.setattr(views, "_passes_liveness_check", lambda frame, face_region=None: False)

    username, spoofed, region = views._evaluate_recognition_match(frame, match, 0.4)

    assert spoofed is True
    assert username == "alice"
    assert region == {"x": 5, "y": 6, "w": 20, "h": 20}


def test_evaluate_match_accepts_live_face(monkeypatch):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    match = pd.Series(
        {
            "identity": "/tmp/train/bob/1.jpg",
            "distance": 0.1,
            "source_x": 2,
            "source_y": 3,
            "source_w": 15,
            "source_h": 16,
        }
    )

    monkeypatch.setattr(views, "_passes_liveness_check", lambda frame, face_region=None: True)

    username, spoofed, region = views._evaluate_recognition_match(frame, match, 0.4)

    assert spoofed is False
    assert username == "bob"
    assert region == {"x": 2, "y": 3, "w": 15, "h": 16}


class _PredictingModel:
    def __init__(self, indices):
        self._indices = indices

    def predict(self, embeddings):
        return np.array(self._indices)


def test_predict_identity_blocks_spoof(monkeypatch):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    model = _PredictingModel([0])
    class_names = ["alice"]

    monkeypatch.setattr(views, "_passes_liveness_check", lambda frame, face_region=None: False)

    name, spoofed, region = views._predict_identity_from_embedding(
        frame,
        [0.1, 0.2],
        {"x": 1, "y": 2, "w": 4, "h": 5},
        model,
        class_names,
        "in",
    )

    assert name is None
    assert spoofed is True
    assert region == {"x": 1, "y": 2, "w": 4, "h": 5}


def test_predict_identity_returns_live_name(monkeypatch):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    model = _PredictingModel([0])
    class_names = ["alice"]

    monkeypatch.setattr(views, "_passes_liveness_check", lambda frame, face_region=None: True)

    name, spoofed, region = views._predict_identity_from_embedding(
        frame,
        [0.1, 0.2],
        {"x": 1, "y": 2, "w": 4, "h": 5},
        model,
        class_names,
        "in",
    )

    assert name == "alice"
    assert spoofed is False
    assert region == {"x": 1, "y": 2, "w": 4, "h": 5}
