import base64
import io
import json
import sys
from unittest.mock import MagicMock, patch

from django.test import override_settings
from django.urls import reverse

import pytest
from PIL import Image

# Mock cv2 to avoid dependency issues in test environment if needed
sys.modules.setdefault("cv2", MagicMock())


def get_minimal_jpeg():
    img = Image.new("RGB", (1, 1))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("test-key",),
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_large_image_rejection(client):
    url = reverse("face-recognition-api")
    valid_b64 = get_minimal_jpeg()

    payload = json.dumps({"image": valid_b64})

    # We patch PIL.Image.open in recognition.views_legacy
    # Note: We must patch it where it is imported.
    with patch("recognition.views_legacy.Image.open") as mock_open:
        # Simulate a huge image
        mock_img = MagicMock()
        mock_img.size = (10000, 10000)  # 100 MP
        mock_img.format = "JPEG"
        mock_open.return_value.__enter__.return_value = mock_img

        response = client.post(
            url, data=payload, content_type="application/json", HTTP_X_API_KEY="test-key"
        )

        # Expect 400 Bad Request due to rejection
        assert response.status_code == 400
        # The generic error message for "returned None" is "Unable to decode the supplied image."
        assert "Unable to decode the supplied image" in response.json().get("error", "")
