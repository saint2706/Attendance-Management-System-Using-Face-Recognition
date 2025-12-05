import os

from django.utils import timezone

import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")

import django  # noqa: E402

# Only setup Django if it hasn't been configured yet (e.g., running standalone)
if not django.apps.apps.ready:
    django.setup()

from recognition.tasks import process_attendance_batch  # noqa: E402
from users.models import Direction, Present, RecognitionAttempt, Time  # noqa: E402

# Mark tests as slow since they use Celery eager mode with DB transactions
pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.mark.django_db(transaction=True)
def test_process_attendance_batch_creates_records(settings, django_user_model):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    username = "celery-user"
    user = django_user_model.objects.create_user(username=username, password="pass1234")

    attempt_in = RecognitionAttempt.objects.create(
        username=username,
        direction=Direction.IN,
        site="lab",
        source="celery-test",
        successful=True,
    )
    attempt_out = RecognitionAttempt.objects.create(
        username=username,
        direction=Direction.OUT,
        site="lab",
        source="celery-test",
        successful=True,
    )

    records = [
        {
            "direction": "in",
            "present": {username: True},
            "attempt_ids": {username: attempt_in.id},
        },
        {
            "direction": "out",
            "present": {username: True},
            "attempt_ids": {username: attempt_out.id},
        },
    ]

    async_result = process_attendance_batch.delay(records)
    payload = async_result.get(timeout=5)

    assert payload["total"] == 2
    assert len(payload["results"]) == 2
    assert all(entry["status"] == "success" for entry in payload["results"])

    today = timezone.localdate()
    present_record = Present.objects.get(user=user, date=today)
    assert present_record.present is True

    times = Time.objects.filter(user=user, date=today).order_by("time")
    assert times.count() == 2
    assert times.first().out is False
    assert times.last().out is True

    attempt_in.refresh_from_db()
    attempt_out.refresh_from_db()
    assert attempt_in.user == user
    assert attempt_in.time_record is not None
    assert attempt_out.user == user
    assert attempt_out.time_record is not None
