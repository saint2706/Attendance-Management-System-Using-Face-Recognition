"""Attendance analytics helpers for the recognition application."""

from __future__ import annotations

import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from django.contrib.auth.models import Group
from django.db.models import Max, Min, Q
from django.utils import timezone

from recognition.utils import check_validity_times
from users.models import Present, Time


@dataclass(slots=True)
class DailyTrend:
    """Container for summarised daily attendance information."""

    date: datetime.date
    total_records: int
    present: int
    late: int
    early: int
    on_time: int
    average_break_hours: float


class AttendanceAnalytics:
    """High level analytics utilities for attendance data.

    The analytics make the following assumptions:

    * The standard working day starts at 09:00 and ends at 17:00 local time.
    * Arrivals at or before 15 minutes prior to the scheduled start are treated
      as *early*.
    * Arrivals 15 minutes after the scheduled start (or later) are treated as
      *late*. Anything in-between is considered *on time*.
    * Break durations are derived from the existing ``check_validity_times``
      helper which validates the in/out sequence.

    ``start_hour``/``end_hour`` as well as the ``early_grace_minutes`` and
    ``late_grace_minutes`` thresholds can be overridden when instantiating the
    class.
    """

    def __init__(
        self,
        *,
        start_hour: int = 9,
        end_hour: int = 17,
        early_grace_minutes: int = 15,
        late_grace_minutes: int = 15,
    ) -> None:
        self.workday_start = datetime.time(hour=start_hour, minute=0)
        self.workday_end = datetime.time(hour=end_hour, minute=0)
        self.early_grace = datetime.timedelta(minutes=early_grace_minutes)
        self.late_grace = datetime.timedelta(minutes=late_grace_minutes)

    # ------------------------------------------------------------------
    # Daily trends
    # ------------------------------------------------------------------
    def get_daily_trends(
        self,
        *,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        employee_id: Optional[int] = None,
    ) -> Dict[str, Iterable[DailyTrend]]:
        """Aggregate punctuality trends on a per-day basis.

        Args:
            start_date: Optional start date filter (inclusive).
            end_date: Optional end date filter (inclusive).
            employee_id: If provided, only data for this employee is returned.

        Returns:
            A dictionary containing the filters that were applied and a list of
            :class:`DailyTrend` entries ordered chronologically.
        """

        filters: Dict[str, object] = {}
        time_filters: Dict[str, object] = {}
        if start_date is not None:
            filters["date__gte"] = start_date
            time_filters["date__gte"] = start_date
        if end_date is not None:
            filters["date__lte"] = end_date
            time_filters["date__lte"] = end_date
        if employee_id is not None:
            filters["user_id"] = employee_id
            time_filters["user_id"] = employee_id

        present_records = (
            Present.objects.filter(**filters).select_related("user").order_by("date", "user_id")
        )

        if not present_records.exists():
            return {
                "start_date": start_date,
                "end_date": end_date,
                "employee_id": employee_id,
                "days": [],
            }

        # Pre-compute first-in and last-out timestamps for efficiency.
        time_summary = {
            (entry["user_id"], entry["date"]): entry
            for entry in (
                Time.objects.filter(**time_filters)
                .values("user_id", "date")
                .annotate(
                    first_in=Min("time", filter=Q(out=False)),
                    last_out=Max("time", filter=Q(out=True)),
                )
            )
        }

        daily_metrics: Dict[datetime.date, Dict[str, object]] = defaultdict(
            lambda: {
                "present": 0,
                "total": 0,
                "late": 0,
                "early": 0,
                "on_time": 0,
                "break_hours": [],
            }
        )

        current_tz = timezone.get_current_timezone()

        for record in present_records:
            day_data = daily_metrics[record.date]
            day_data["total"] += 1
            if record.present:
                day_data["present"] += 1

            key = (record.user_id, record.date)
            summary = time_summary.get(key, {})
            first_in = summary.get("first_in")

            workday_start = datetime.datetime.combine(record.date, self.workday_start)
            if timezone.is_naive(workday_start):
                workday_start = timezone.make_aware(workday_start, current_tz)

            early_threshold = workday_start - self.early_grace
            late_threshold = workday_start + self.late_grace

            if first_in is not None:
                if first_in <= early_threshold:
                    day_data["early"] += 1
                elif first_in >= late_threshold:
                    day_data["late"] += 1
                else:
                    day_data["on_time"] += 1

            times_qs = Time.objects.filter(user_id=record.user_id, date=record.date).order_by(
                "time"
            )
            is_valid, break_hours = check_validity_times(times_qs)
            if is_valid:
                day_data["break_hours"].append(break_hours)

        trends: List[DailyTrend] = []
        for date_key in sorted(daily_metrics):
            data = daily_metrics[date_key]
            break_hours_list: List[float] = data["break_hours"]
            avg_break = sum(break_hours_list) / len(break_hours_list) if break_hours_list else 0.0
            trends.append(
                DailyTrend(
                    date=date_key,
                    total_records=data["total"],
                    present=data["present"],
                    late=data["late"],
                    early=data["early"],
                    on_time=data["on_time"],
                    average_break_hours=round(avg_break, 2),
                )
            )

        return {
            "start_date": start_date,
            "end_date": end_date,
            "employee_id": employee_id,
            "days": trends,
        }

    # ------------------------------------------------------------------
    # Department summary
    # ------------------------------------------------------------------
    def get_department_summary(
        self,
        *,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> Dict[str, object]:
        """Summarise attendance rates grouped by department.

        Departments are derived from Django ``Group`` memberships. When a user
        does not belong to any group, the department is reported as
        ``"Unassigned"``. The summary contains totals and compares the
        department rate to the overall attendance rate for the provided
        timeframe.
        """

        filters: Dict[str, object] = {}
        if start_date is not None:
            filters["date__gte"] = start_date
        if end_date is not None:
            filters["date__lte"] = end_date

        present_qs = Present.objects.filter(**filters).select_related("user")

        if not present_qs.exists():
            return {
                "start_date": start_date,
                "end_date": end_date,
                "departments": [],
                "overall_rate": 0.0,
            }

        # Build a mapping from user id to department name once.
        user_departments: Dict[int, str] = {}
        user_ids = present_qs.values_list("user_id", flat=True).distinct()
        for user_id in user_ids:
            group = (
                Group.objects.filter(user__id=user_id)
                .order_by("name")
                .values_list("name", flat=True)
                .first()
            )
            user_departments[user_id] = group or "Unassigned"

        department_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"total": 0, "present": 0}
        )

        for record in present_qs:
            department = user_departments.get(record.user_id, "Unassigned")
            stats = department_stats[department]
            stats["total"] += 1
            if record.present:
                stats["present"] += 1

        overall_total = sum(stats["total"] for stats in department_stats.values())
        overall_present = sum(stats["present"] for stats in department_stats.values())
        overall_rate = overall_present / overall_total if overall_total else 0.0

        departments: List[Dict[str, object]] = []
        for name, stats in sorted(department_stats.items()):
            dept_rate = stats["present"] / stats["total"] if stats["total"] else 0.0
            comparative = dept_rate - overall_rate
            departments.append(
                {
                    "department": name,
                    "present": int(stats["present"]),
                    "total": int(stats["total"]),
                    "attendance_rate": round(dept_rate, 3),
                    "relative_to_overall": round(comparative, 3),
                }
            )

        return {
            "start_date": start_date,
            "end_date": end_date,
            "departments": departments,
            "overall_rate": round(overall_rate, 3),
        }

    # ------------------------------------------------------------------
    # Attendance prediction
    # ------------------------------------------------------------------
    def get_attendance_prediction(
        self,
        *,
        employee_id: int,
        window: int = 5,
    ) -> Dict[str, object]:
        """Estimate future attendance for the provided employee.

        The predictor currently implements a simple moving average across the
        ``window`` most recent :class:`users.models.Present` records, producing
        a probability that the employee will be present on the next work day.

        The approach assumes that recent attendance patterns are indicative of
        near-term behaviour and does not incorporate seasonality or external
        factors. This method is intentionally lightweight so that it can be
        swapped for a more sophisticated model (such as logistic regression or
        an ML pipeline) in the future without changing the API.
        """

        if window <= 0:
            raise ValueError("window must be greater than zero")

        recent_records = list(
            Present.objects.filter(user_id=employee_id).order_by("-date")[:window]
        )

        if not recent_records:
            return {
                "employee_id": employee_id,
                "window": window,
                "method": "moving_average",
                "assumptions": (
                    "Prediction unavailable because the employee has no " "attendance history."
                ),
                "prediction": None,
                "confidence": 0.0,
            }

        values = [1 if record.present else 0 for record in recent_records]
        probability = sum(values) / len(values)
        prediction = probability >= 0.5

        return {
            "employee_id": employee_id,
            "window": window,
            "method": "moving_average",
            "assumptions": ("Recent attendance behaviour is representative of the near " "future."),
            "prediction": prediction,
            "confidence": round(probability, 3),
            "history": [
                {
                    "date": record.date,
                    "present": record.present,
                }
                for record in recent_records
            ],
        }
