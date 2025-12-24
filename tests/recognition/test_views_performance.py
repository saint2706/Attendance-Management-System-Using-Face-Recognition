import datetime
from unittest.mock import patch

from django.contrib.auth.models import User
from django.utils import timezone

import pytest

from recognition.views_legacy import hours_vs_date_given_employee, hours_vs_employee_given_date
from users.models import Direction, Present, Time


@pytest.mark.django_db
def test_n_plus_one_hours_vs_employee_given_date(django_assert_num_queries):
    # Setup
    date = timezone.localdate()
    users = []
    # Create 5 users
    for i in range(5):
        u = User.objects.create_user(username=f"user{i}", password="password")
        users.append(u)
        Present.objects.create(user=u, date=date, present=True)
        # Add some time entries
        t1 = timezone.now()
        t2 = t1 + datetime.timedelta(hours=8)
        Time.objects.create(user=u, date=date, time=t1, direction=Direction.IN)
        Time.objects.create(user=u, date=date, time=t2, direction=Direction.OUT)

    present_qs = Present.objects.filter(date=date)
    time_qs = Time.objects.filter(date=date)

    with patch("recognition.views_legacy._save_plot_to_media") as mock_save:
        mock_save.return_value = "/media/fake.png"
        with patch("recognition.views_legacy.plt"):
            with patch("recognition.views_legacy.sns"):
                # Optimized implementation:
                # 1 query for present_qs
                # 1 query for time_qs (list(time_qs))
                # Note: EXPLAIN queries are environment specific and not present in SQLite/Standard Django Test default.
                # Total expected: 2
                with django_assert_num_queries(2):
                    hours_vs_employee_given_date(present_qs, time_qs)


@pytest.mark.django_db
def test_n_plus_one_hours_vs_date_given_employee(django_assert_num_queries):
    # Setup
    user = User.objects.create_user(username="user_date", password="password")
    today = timezone.localdate()
    days = 5

    for i in range(days):
        d = today - datetime.timedelta(days=i)
        Present.objects.create(user=user, date=d, present=True)
        t1 = timezone.now()
        t2 = t1 + datetime.timedelta(hours=8)
        Time.objects.create(user=user, date=d, time=t1, direction=Direction.IN)
        Time.objects.create(user=user, date=d, time=t2, direction=Direction.OUT)

    present_qs = Present.objects.filter(user=user)
    time_qs = Time.objects.filter(user=user)

    with patch("recognition.views_legacy._save_plot_to_media") as mock_save:
        mock_save.return_value = "/media/fake.png"
        with patch("recognition.views_legacy.plt"):
            with patch("recognition.views_legacy.sns"):
                # Optimized: 2 queries
                with django_assert_num_queries(2):
                    hours_vs_date_given_employee(present_qs, time_qs)
