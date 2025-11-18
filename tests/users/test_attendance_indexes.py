"""Tests for attendance-related database indexes and query patterns."""

from django.contrib.auth import get_user_model
from django.db import connection
from django.utils import timezone

import pytest

from users.models import Present, Time


def _collect_index_names(model) -> set[str]:
    """Return the set of index names defined for the given model's table."""

    with connection.cursor() as cursor:
        constraints = connection.introspection.get_constraints(
            cursor, model._meta.db_table
        )
    return {name for name, info in constraints.items() if info.get("index")}


@pytest.mark.django_db
def test_present_lookups_by_user_and_date_succeed():
    """Filtering present records by user and date should return expected rows."""

    user = get_user_model().objects.create_user(
        username="present-user", password="password123"
    )
    today = timezone.localdate()

    Present.objects.create(user=user, date=today, present=True)

    assert Present.objects.filter(user=user, date=today).count() == 1
    # The reverse ordering mimics ORM usage that can leverage the alternate composite index.
    assert Present.objects.filter(date=today, user=user).exists()


@pytest.mark.django_db
def test_time_lookups_by_user_and_date_succeed():
    """Filtering time records by user and date should return expected rows."""

    user = get_user_model().objects.create_user(
        username="time-user", password="password123"
    )
    now = timezone.now()
    event_date = now.date()

    Time.objects.create(user=user, date=event_date, time=now, out=False)

    assert Time.objects.filter(user=user, date=event_date).exists()
    assert Time.objects.filter(date=event_date, user=user).exists()


@pytest.mark.django_db
def test_attendance_indexes_are_installed():
    """The named composite indexes from the migration should exist in the database."""

    present_indexes = _collect_index_names(Present)
    time_indexes = _collect_index_names(Time)

    assert "users_present_user_date_idx" in present_indexes
    assert "users_present_date_user_idx" in present_indexes
    assert "users_time_user_date_idx" in time_indexes
    assert "users_time_date_user_idx" in time_indexes
