"""Unit tests for the attendance analytics helpers."""

import datetime
import os

import pytest
from django.utils import timezone

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings"
)

import django  # noqa: E402

django.setup()

from django.contrib.auth import get_user_model  # noqa: E402
from django.contrib.auth.models import Group  # noqa: E402

from recognition.analysis import AttendanceAnalytics  # noqa: E402
from users.models import Present, Time  # noqa: E402


def _aware(dt: datetime.datetime) -> datetime.datetime:
    tz = timezone.get_current_timezone()
    if timezone.is_naive(dt):
        return timezone.make_aware(dt, tz)
    return dt


User = get_user_model()


@pytest.mark.django_db
def test_get_daily_trends_with_breaks_and_filters():
    analytics = AttendanceAnalytics()
    user = User.objects.create_user("alice", password="test1234")

    day1 = datetime.date(2024, 1, 2)
    day2 = datetime.date(2024, 1, 3)

    Present.objects.create(user=user, date=day1, present=True)
    Present.objects.create(user=user, date=day2, present=True)

    Time.objects.create(user=user, date=day1, time=_aware(datetime.datetime(2024, 1, 2, 8, 45)), out=False)
    Time.objects.create(user=user, date=day1, time=_aware(datetime.datetime(2024, 1, 2, 12, 0)), out=True)
    Time.objects.create(user=user, date=day1, time=_aware(datetime.datetime(2024, 1, 2, 13, 0)), out=False)
    Time.objects.create(user=user, date=day1, time=_aware(datetime.datetime(2024, 1, 2, 17, 5)), out=True)

    Time.objects.create(user=user, date=day2, time=_aware(datetime.datetime(2024, 1, 3, 9, 25)), out=False)
    Time.objects.create(user=user, date=day2, time=_aware(datetime.datetime(2024, 1, 3, 17, 0)), out=True)

    result = analytics.get_daily_trends(start_date=day1, end_date=day2)
    assert result["start_date"] == day1
    assert result["end_date"] == day2

    days = result["days"]
    assert len(days) == 2

    first_day, second_day = days
    assert first_day.date == day1
    assert first_day.present == 1
    assert first_day.early == 1
    assert first_day.late == 0
    assert first_day.on_time == 0
    assert first_day.average_break_hours == pytest.approx(1.0)

    assert second_day.date == day2
    assert second_day.late == 1
    assert second_day.early == 0
    assert second_day.on_time == 0
    assert second_day.average_break_hours == pytest.approx(0.0)

    filtered = analytics.get_daily_trends(employee_id=user.id)
    assert len(filtered["days"]) == 2
    assert all(entry.present == 1 for entry in filtered["days"])


@pytest.mark.django_db
def test_get_department_summary_handles_unassigned_department():
    analytics = AttendanceAnalytics()
    group = Group.objects.create(name="Sales")

    sales_user = User.objects.create_user("bob", password="pass1234")
    sales_user.groups.add(group)
    other_user = User.objects.create_user("charlie", password="pass1234")

    Present.objects.create(user=sales_user, date=datetime.date(2024, 2, 1), present=True)
    Present.objects.create(user=sales_user, date=datetime.date(2024, 2, 2), present=False)

    Present.objects.create(user=other_user, date=datetime.date(2024, 2, 1), present=True)
    Present.objects.create(user=other_user, date=datetime.date(2024, 2, 2), present=True)

    summary = analytics.get_department_summary()
    assert summary["overall_rate"] == pytest.approx(0.75, abs=1e-3)

    departments = {entry["department"]: entry for entry in summary["departments"]}
    assert "Sales" in departments
    assert "Unassigned" in departments

    sales_stats = departments["Sales"]
    assert sales_stats["attendance_rate"] == pytest.approx(0.5, abs=1e-3)
    assert sales_stats["relative_to_overall"] == pytest.approx(-0.25, abs=1e-3)

    other_stats = departments["Unassigned"]
    assert other_stats["attendance_rate"] == pytest.approx(1.0, abs=1e-3)
    assert other_stats["relative_to_overall"] == pytest.approx(0.25, abs=1e-3)


@pytest.mark.django_db
def test_get_attendance_prediction_with_and_without_history():
    analytics = AttendanceAnalytics()
    user = User.objects.create_user("dana", password="pass1234")

    empty_prediction = analytics.get_attendance_prediction(employee_id=user.id)
    assert empty_prediction["prediction"] is None
    assert empty_prediction["confidence"] == 0.0

    today = timezone.localdate()
    for offset, present in enumerate([True, True, False, True]):
        Present.objects.create(
            user=user,
            date=today - datetime.timedelta(days=offset),
            present=present,
        )

    prediction = analytics.get_attendance_prediction(employee_id=user.id, window=4)
    assert prediction["prediction"] is True
    assert prediction["confidence"] == pytest.approx(0.75, abs=1e-3)
    assert len(prediction["history"]) == 4
