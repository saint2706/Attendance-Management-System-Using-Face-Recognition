import json
import base64
import io
import pytest
from unittest.mock import MagicMock, patch
from django.urls import reverse
from django.test import override_settings

@pytest.mark.django_db
@override_settings(RECOGNITION_API_KEYS=("test-key",))
def test_liveness_frames_dos(client):
    """
    Simulate a request with a large number of liveness frames to verify DoS potential.
    We count how many times _decode_image_bytes is called.
    """
    url = reverse("face-recognition-api")

    # Create a minimal valid base64 image
    img_data = b"fake_image_data"
    b64_img = base64.b64encode(img_data).decode('utf-8')

    # Create a payload with many frames
    num_frames = 100
    liveness_frames = [b64_img] * num_frames

    payload = {
        "embedding": [0.1] * 128,  # Provide embedding to pass initial checks
        "liveness_frames": liveness_frames
    }

    # Mock _decode_image_bytes to count calls
    # We mock it in views_legacy module
    with patch("recognition.views_legacy._decode_image_bytes") as mock_decode:
        # Return a fake frame so it proceeds
        mock_decode.return_value = MagicMock()

        response = client.post(
            url,
            data=json.dumps(payload),
            content_type="application/json",
            HTTP_X_API_KEY="test-key"
        )

        # In the vulnerable version, it should be called num_frames times
        assert mock_decode.call_count == num_frames
