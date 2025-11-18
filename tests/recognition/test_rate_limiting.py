import os
import sys
from unittest.mock import MagicMock, patch

import django
from django.core.cache import cache
from django.http import HttpResponse
from django.urls import reverse

import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")

sys.modules.setdefault("cv2", MagicMock())

# Only setup Django if it hasn't been configured yet (e.g., running standalone)
if not django.apps.apps.ready:
    django.setup()

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def clear_rate_limit_cache():
    """Clear the rate-limit cache before and after each test to prevent state leakage."""
    cache.clear()
    yield
    cache.clear()


def _exercise_rate_limit(client, settings, url_name: str) -> tuple[int, int]:
    cache.clear()
    settings.RECOGNITION_ATTENDANCE_RATE_LIMIT = "2/m"
    url = reverse(url_name)
    with patch("recognition.views._mark_attendance", return_value=HttpResponse("ok")) as mocked:
        for _ in range(2):
            ok_response = client.post(url)
            assert ok_response.status_code == 200
        limited_response = client.post(url)
    cache.clear()
    return mocked.call_count, limited_response.status_code


def test_mark_attendance_in_rate_limit_blocks_after_threshold(client, django_user_model, settings):
    user = django_user_model.objects.create_user("rate", password="password")
    client.force_login(user)

    call_count, status_code = _exercise_rate_limit(client, settings, "mark-your-attendance")

    assert call_count == 2
    assert status_code == 429


def test_mark_attendance_out_rate_limit_blocks_after_threshold(client, django_user_model, settings):
    user = django_user_model.objects.create_user("rate-out", password="password")
    client.force_login(user)

    call_count, status_code = _exercise_rate_limit(client, settings, "mark-your-attendance-out")

    assert call_count == 2
    assert status_code == 429
