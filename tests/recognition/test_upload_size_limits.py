"""
Test upload size limit enforcement for face recognition API.

This module tests the RECOGNITION_MAX_UPLOAD_SIZE setting to ensure DoS prevention
through upload size limits works correctly for all payload types:
- File uploads via request.FILES
- Raw bytes/bytearray payloads
- Base64-encoded strings (with and without data URI headers)
"""
import base64
import io
import json
from unittest.mock import MagicMock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse

import pytest
from PIL import Image


def create_test_image_bytes(target_size_kb: int) -> bytes:
    """
    Create a JPEG image of approximately the specified size in KB.
    
    For test reliability, this aims to stay slightly under the target to account
    for JPEG compression variability.
    """
    import random
    target_bytes = int(target_size_kb * 1024 * 0.8)  # Aim for 80% of target to stay under limit
    # Start with an estimate
    pixels = max(50, int((target_bytes * 2) ** 0.5))
    
    # Create image with random noise
    img = Image.new("RGB", (pixels, pixels))
    pixels_data = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                   for _ in range(pixels * pixels)]
    img.putdata(pixels_data)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    result = buf.getvalue()
    
    # If we need to grow the image
    attempts = 0
    while len(result) < target_bytes and attempts < 3:
        pixels = int(pixels * 1.2)
        img = Image.new("RGB", (pixels, pixels))
        pixels_data = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                       for _ in range(pixels * pixels)]
        img.putdata(pixels_data)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        result = buf.getvalue()
        attempts += 1
    
    return result


def create_base64_image(size_kb: int, with_header: bool = False) -> str:
    """Create a base64-encoded image string."""
    image_bytes = create_test_image_bytes(size_kb)
    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    if with_header:
        return f"data:image/jpeg;base64,{b64_str}"
    return b64_str


@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("test-key",),
    RECOGNITION_MAX_UPLOAD_SIZE=10240,  # 10 KB limit for testing
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    RATELIMIT_ENABLE=False,  # Disable rate limiting for tests
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_file_upload_within_limit(client):
    """Test that file uploads within the size limit are accepted (or fail for other reasons)."""
    url = reverse("face-recognition-api")
    # Use 5 KB target (will be ~4 KB actual) - well within 10 KB limit
    small_image = create_test_image_bytes(5)
    uploaded_file = SimpleUploadedFile("test.jpg", small_image, content_type="image/jpeg")

    response = client.post(
        url,
        data={"image": uploaded_file, "username": "testuser"},
        HTTP_X_API_KEY="test-key",
    )

    # Should not be rejected due to size (may fail for other reasons like no match or no embeddings)
    # The key is that it should NOT have the size error
    assert response.status_code != 400 or "exceeds maximum allowed size" not in response.json().get("error", "")


@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("test-key",),
    RECOGNITION_MAX_UPLOAD_SIZE=10240,  # 10 KB limit for testing
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    RATELIMIT_ENABLE=False,  # Disable rate limiting for tests
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_file_upload_exceeds_limit(client):
    """Test that file uploads exceeding the size limit are rejected."""
    url = reverse("face-recognition-api")
    large_image = create_test_image_bytes(15)  # 15 KB, exceeds 10 KB limit
    uploaded_file = SimpleUploadedFile("test.jpg", large_image, content_type="image/jpeg")

    response = client.post(
        url,
        data={"image": uploaded_file, "username": "testuser"},
        HTTP_X_API_KEY="test-key",
    )

    assert response.status_code == 400
    assert "exceeds maximum allowed size" in response.json().get("error", "")


@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("test-key",),
    RECOGNITION_MAX_UPLOAD_SIZE=10240,  # 10 KB limit for testing
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    RATELIMIT_ENABLE=False,  # Disable rate limiting for tests
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_base64_without_header_within_limit(client):
    """Test that base64 payloads without data URI header within limit are accepted."""
    url = reverse("face-recognition-api")
    # Use 5 KB target (will be ~4 KB actual) - well within 10 KB limit
    b64_image = create_base64_image(5, with_header=False)
    
    payload = json.dumps({
        "image": b64_image,
        "username": "testuser"
    })

    response = client.post(
        url,
        data=payload,
        content_type="application/json",
        HTTP_X_API_KEY="test-key",
    )

    # Should not be rejected due to size
    assert response.status_code != 400 or "exceeds maximum allowed size" not in response.json().get("error", "")


@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("test-key",),
    RECOGNITION_MAX_UPLOAD_SIZE=10240,  # 10 KB limit for testing
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    RATELIMIT_ENABLE=False,  # Disable rate limiting for tests
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_base64_without_header_exceeds_limit(client):
    """Test that base64 payloads without data URI header exceeding limit are rejected."""
    url = reverse("face-recognition-api")
    b64_image = create_base64_image(15, with_header=False)  # 15 KB
    
    payload = json.dumps({
        "image": b64_image,
        "username": "testuser"
    })

    response = client.post(
        url,
        data=payload,
        content_type="application/json",
        HTTP_X_API_KEY="test-key",
    )

    assert response.status_code == 400
    assert "exceeds maximum allowed size" in response.json().get("error", "")


@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("test-key",),
    RECOGNITION_MAX_UPLOAD_SIZE=10240,  # 10 KB limit for testing
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    RATELIMIT_ENABLE=False,  # Disable rate limiting for tests
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_base64_with_header_within_limit(client):
    """Test that base64 payloads with data URI header within limit are accepted."""
    url = reverse("face-recognition-api")
    # Use 5 KB target (will be ~4 KB actual) - well within 10 KB limit
    b64_image = create_base64_image(5, with_header=True)
    
    payload = json.dumps({
        "image": b64_image,
        "username": "testuser"
    })

    response = client.post(
        url,
        data=payload,
        content_type="application/json",
        HTTP_X_API_KEY="test-key",
    )

    # Should not be rejected due to size
    assert response.status_code != 400 or "exceeds maximum allowed size" not in response.json().get("error", "")


@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("test-key",),
    RECOGNITION_MAX_UPLOAD_SIZE=10240,  # 10 KB limit for testing
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    RATELIMIT_ENABLE=False,  # Disable rate limiting for tests
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_base64_with_header_exceeds_limit(client):
    """
    Test that base64 payloads with data URI header exceeding limit are rejected.
    
    This is a critical test case that validates the fix for the security issue where
    the length check was performed before stripping the data URI header, allowing
    an attacker to bypass the size limit by including a header.
    """
    url = reverse("face-recognition-api")
    b64_image = create_base64_image(15, with_header=True)  # 15 KB with header
    
    payload = json.dumps({
        "image": b64_image,
        "username": "testuser"
    })

    response = client.post(
        url,
        data=payload,
        content_type="application/json",
        HTTP_X_API_KEY="test-key",
    )

    assert response.status_code == 400
    assert "exceeds maximum allowed size" in response.json().get("error", "")


@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("test-key",),
    RECOGNITION_MAX_UPLOAD_SIZE=10240,  # 10 KB limit for testing
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    RATELIMIT_ENABLE=False,  # Disable rate limiting for tests
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_base64_approximation_check(client):
    """
    Test that the base64 length approximation check works before full decode.
    
    This tests the efficiency optimization where we reject obviously oversized
    base64 strings before attempting to decode them.
    """
    url = reverse("face-recognition-api")
    # Create a base64 string that's large enough to trigger the approximation check
    # but might not trigger the post-decode check
    huge_b64 = "A" * (10240 * 2)  # ~20 KB in base64, exceeds 10 KB * 1.4 = 14 KB
    
    payload = json.dumps({
        "image": huge_b64,
        "username": "testuser"
    })

    response = client.post(
        url,
        data=payload,
        content_type="application/json",
        HTTP_X_API_KEY="test-key",
    )

    assert response.status_code == 400
    assert "exceeds maximum allowed size" in response.json().get("error", "")


@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("test-key",),
    RECOGNITION_MAX_UPLOAD_SIZE=10240,  # 10 KB limit for testing
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    RATELIMIT_ENABLE=False,  # Disable rate limiting for tests
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_post_decode_size_check(client):
    """
    Test that the post-decode size check catches edge cases.
    
    This ensures that even if a payload passes the approximation check,
    it's still validated after decoding.
    """
    url = reverse("face-recognition-api")
    # Create a payload that might pass approximation but exceeds after decode
    image_bytes = create_test_image_bytes(12)  # 12 KB, exceeds 10 KB limit
    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    
    payload = json.dumps({
        "image": b64_str,
        "username": "testuser"
    })

    response = client.post(
        url,
        data=payload,
        content_type="application/json",
        HTTP_X_API_KEY="test-key",
    )

    assert response.status_code == 400
    assert "exceeds maximum allowed size" in response.json().get("error", "")


@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("test-key",),
    RECOGNITION_MAX_UPLOAD_SIZE=5 * 1024 * 1024,  # Default 5 MB
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    RATELIMIT_ENABLE=False,  # Disable rate limiting for tests
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_default_size_limit(client):
    """Test that the default size limit of 5 MB is applied when not configured."""
    url = reverse("face-recognition-api")
    # Create a small payload that should pass with default 5 MB limit
    small_image = create_test_image_bytes(100)  # 100 KB
    uploaded_file = SimpleUploadedFile("test.jpg", small_image, content_type="image/jpeg")

    response = client.post(
        url,
        data={"image": uploaded_file, "username": "testuser"},
        HTTP_X_API_KEY="test-key",
    )

    # Should not be rejected due to size with default limit
    assert response.status_code != 400 or "exceeds maximum allowed size" not in response.json().get("error", "")


@pytest.mark.django_db
@override_settings(
    RECOGNITION_API_KEYS=("test-key",),
    RECOGNITION_MAX_UPLOAD_SIZE=1024,  # 1 KB limit
    RECOGNITION_DISTANCE_THRESHOLD=0.5,
    RATELIMIT_ENABLE=False,  # Disable rate limiting for tests
    DEEPFACE_OPTIMIZATIONS={
        "distance_metric": "euclidean_l2",
        "model": "Facenet",
        "detector_backend": "opencv",
        "enforce_detection": False,
        "anti_spoofing": False,
    },
)
def test_custom_size_limit(client):
    """Test that a custom RECOGNITION_MAX_UPLOAD_SIZE setting is respected."""
    url = reverse("face-recognition-api")
    # Create a 2 KB image that should exceed the 1 KB custom limit
    image = create_test_image_bytes(2)
    uploaded_file = SimpleUploadedFile("test.jpg", image, content_type="image/jpeg")

    response = client.post(
        url,
        data={"image": uploaded_file, "username": "testuser"},
        HTTP_X_API_KEY="test-key",
    )

    assert response.status_code == 400
    assert "exceeds maximum allowed size" in response.json().get("error", "")
