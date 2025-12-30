import json
import base64
import pytest
from unittest.mock import MagicMock, patch
from django.urls import reverse
from django.test import override_settings


@pytest.mark.django_db
@override_settings(RECOGNITION_API_KEYS=("test-key",))
def test_liveness_frames_limit_enforced(client):
    """
    Verify that the DoS vulnerability is mitigated by rejecting requests
    exceeding the liveness frames limit (default: 20).
    """
    url = reverse("face-recognition-api")

    # Create a minimal valid base64 image
    img_data = b"fake_image_data"
    b64_img = base64.b64encode(img_data).decode('utf-8')

    # Create a payload with too many frames (exceeding the limit of 20)
    num_frames = 100
    liveness_frames = [b64_img] * num_frames

    payload = {
        "embedding": [0.1] * 128,  # Provide embedding to pass initial checks
        "liveness_frames": liveness_frames
    }

    # Mock _decode_image_bytes in the same module as FaceRecognitionAPI to count calls
    with patch("recognition.views_legacy._decode_image_bytes") as mock_decode:
        # Return a fake frame so it would proceed if not rejected
        mock_decode.return_value = MagicMock()

        client.post(
            url,
            data=json.dumps(payload),
            content_type="application/json",
            HTTP_X_API_KEY="test-key"
        )

        # The fix should reject the request before any image decoding occurs
        # or at most decode frames up to the limit check
        assert mock_decode.call_count == 0, (
            f"Expected no image decoding for requests exceeding limit, "
            f"but _decode_image_bytes was called {mock_decode.call_count} times"
        )


@pytest.mark.django_db
@override_settings(RECOGNITION_API_KEYS=("test-key",))
def test_liveness_frames_limit_rejection(client):
    """
    Verify that requests exceeding the liveness frames limit are rejected
    with a 400 Bad Request response.
    """
    url = reverse("face-recognition-api")

    # Create a minimal valid base64 image
    img_data = b"fake_image_data"
    b64_img = base64.b64encode(img_data).decode('utf-8')

    # Create a payload with too many frames (exceeding the limit of 20)
    num_frames = 100
    liveness_frames = [b64_img] * num_frames

    payload = {
        "embedding": [0.1] * 128,
        "liveness_frames": liveness_frames
    }

    response = client.post(
        url,
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_X_API_KEY="test-key"
    )

    # Should return 400 Bad Request
    assert response.status_code == 400
    response_data = response.json()
    assert "error" in response_data
    assert "Too many liveness frames" in response_data["error"]


@pytest.mark.django_db
@override_settings(RECOGNITION_API_KEYS=("test-key",))
def test_liveness_frames_within_limit(client):
    """
    Verify that requests with liveness frames at or below the limit
    are processed successfully without being rejected.
    """
    url = reverse("face-recognition-api")

    # Create a minimal valid base64 image
    img_data = b"fake_image_data"
    b64_img = base64.b64encode(img_data).decode('utf-8')

    # Create a payload with frames within the limit (20 frames)
    num_frames = 20
    liveness_frames = [b64_img] * num_frames

    payload = {
        "embedding": [0.1] * 128,
        "liveness_frames": liveness_frames
    }

    # Mock _decode_image_bytes to avoid actual image processing
    with patch("recognition.views_legacy._decode_image_bytes") as mock_decode:
        mock_decode.return_value = MagicMock()

        response = client.post(
            url,
            data=json.dumps(payload),
            content_type="application/json",
            HTTP_X_API_KEY="test-key"
        )

        # Should not be rejected for having too many frames
        # The request may fail for other reasons (e.g., invalid image data),
        # but it should not be rejected with the "Too many liveness frames" error
        if response.status_code == 400:
            response_data = response.json()
            assert "Too many liveness frames" not in response_data.get("error", "")


@pytest.mark.django_db
@override_settings(RECOGNITION_API_KEYS=("test-key",))
def test_liveness_frames_single_frame(client):
    """
    Verify that requests with a single liveness frame are processed successfully.
    """
    url = reverse("face-recognition-api")

    # Create a minimal valid base64 image
    img_data = b"fake_image_data"
    b64_img = base64.b64encode(img_data).decode('utf-8')

    payload = {
        "embedding": [0.1] * 128,
        "liveness_frames": [b64_img]
    }

    # Mock _decode_image_bytes to avoid actual image processing
    with patch("recognition.views_legacy._decode_image_bytes") as mock_decode:
        mock_decode.return_value = MagicMock()

        response = client.post(
            url,
            data=json.dumps(payload),
            content_type="application/json",
            HTTP_X_API_KEY="test-key"
        )

        # Should not be rejected for having too many frames
        if response.status_code == 400:
            response_data = response.json()
            assert "Too many liveness frames" not in response_data.get("error", "")
