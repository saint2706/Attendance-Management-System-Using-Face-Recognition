import os

import pytest
from django.utils import timezone

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")

import django

django.setup()

from recognition.tasks import process_attendance_batch
from users.models import Present, Time


@pytest.mark.django_db(transaction=True)
def test_process_attendance_batch_creates_records(settings, django_user_model):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    username = "celery-user"
    user = django_user_model.objects.create_user(username=username, password="pass1234")

    records = [
        {"direction": "in", "present": {username: True}},
        {"direction": "out", "present": {username: True}},
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
