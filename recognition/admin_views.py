"""Admin views for evaluation metrics, reports, and health dashboards."""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Callable

from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate, TruncWeek
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone

from users.models import RecognitionAttempt

from . import health, monitoring
from .forms import AttendanceSessionFilterForm
from .models import RecognitionOutcome


@staff_member_required
def evaluation_dashboard(request: HttpRequest) -> HttpResponse:
    """Render evaluation metrics, confidence intervals, and related figures."""
    reports_dir = Path(settings.BASE_DIR) / "reports"
    metrics_file = reports_dir / "metrics_with_ci.json"
    threshold_file = reports_dir / "selected_threshold.json"
    split_summary_file = reports_dir / "split_summary.json"

    context = {
        "metrics_available": False,
        "threshold_available": False,
        "splits_available": False,
        "figures_available": False,
    }

    # Load metrics
    if metrics_file.exists():
        with open(metrics_file) as f:
            data = json.load(f)
            context["metrics"] = data.get("metrics", {})
            context["confidence_intervals"] = data.get("confidence_intervals", {})
            context["metrics_available"] = True

    # Load threshold
    if threshold_file.exists():
        with open(threshold_file) as f:
            context["threshold_info"] = json.load(f)
            context["threshold_available"] = True

    # Load split summary
    if split_summary_file.exists():
        with open(split_summary_file) as f:
            context["split_info"] = json.load(f)
            context["splits_available"] = True

    # Check for figures
    figures_dir = reports_dir / "figures"
    if figures_dir.exists():
        figures = ["roc.png", "pr.png", "det.png", "calibration.png"]
        available_figures = [f for f in figures if (figures_dir / f).exists()]
        context["available_figures"] = available_figures
        context["figures_available"] = len(available_figures) > 0

    return render(request, "recognition/admin/evaluation_dashboard.html", context)


def _prepare_accuracy_trend(
    time_trunc_func: Callable[[str], Any],
) -> list[dict[str, Any]]:
    """Aggregate accuracy information using the provided truncation function."""

    aggregates = (
        RecognitionOutcome.objects.annotate(period=time_trunc_func("created_at"))
        .values("period")
        .order_by("period")
        .annotate(
            total=Count("id"),
            accepted=Count("id", filter=Q(accepted=True)),
            average_confidence=Avg("confidence"),
        )
    )

    trend = []
    for entry in aggregates:
        total = entry.get("total") or 0
        accepted = entry.get("accepted") or 0
        accuracy = float(accepted) / float(total) if total else 0.0
        trend.append(
            {
                "period": entry.get("period"),
                "total": total,
                "accepted": accepted,
                "rejected": total - accepted,
                "accuracy": accuracy,
                "average_confidence": entry.get("average_confidence"),
            }
        )
    return trend


@staff_member_required
def recognition_accuracy_trends(request: HttpRequest) -> HttpResponse:
    """Display daily and weekly accuracy aggregates for recognition outcomes."""

    retention_setting = getattr(settings, "RECOGNITION_OUTCOME_RETENTION_DAYS", 30)
    retention_days_numeric = None
    retention_label = "indefinitely"
    if retention_setting not in (None, "none", ""):
        try:
            retention_days_numeric = int(retention_setting)
            if retention_days_numeric <= 0:
                retention_days_numeric = None
            else:
                retention_label = str(retention_days_numeric)
        except (TypeError, ValueError):  # pragma: no cover - defensive casting
            retention_label = str(retention_setting)

    context = {
        "daily_trend": _prepare_accuracy_trend(TruncDate),
        "weekly_trend": _prepare_accuracy_trend(TruncWeek),
        "retention_days": retention_label,
        "retention_days_numeric": retention_days_numeric,
    }
    return render(request, "recognition/admin/recognition_trends.html", context)


@staff_member_required
def ablation_results(request: HttpRequest) -> HttpResponse:
    """Render ablation experiment results when report files are present."""
    reports_dir = Path(settings.BASE_DIR) / "reports"
    ablation_csv = reports_dir / "ablation_results.csv"

    context = {
        "ablation_available": False,
    }

    if ablation_csv.exists():
        import pandas as pd

        df = pd.read_csv(ablation_csv)
        context["ablation_results"] = df.to_dict("records")
        context["ablation_available"] = True

        # Find best configuration
        if not df.empty:
            best_idx = df["accuracy"].idxmax()
            context["best_config"] = df.iloc[best_idx].to_dict()

    return render(request, "recognition/admin/ablation_results.html", context)


@staff_member_required
def failure_analysis(request: HttpRequest) -> HttpResponse:
    """Present false accept/reject breakdowns and subgroup metrics."""
    reports_dir = Path(settings.BASE_DIR) / "reports"
    failure_cases_csv = reports_dir / "failure_cases.csv"
    subgroup_csv = reports_dir / "subgroup_metrics.csv"

    context = {
        "failures_available": False,
        "subgroups_available": False,
    }

    if failure_cases_csv.exists():
        import pandas as pd

        df = pd.read_csv(failure_cases_csv)

        # Separate false accepts and false rejects
        fa_df = df[df["failure_type"] == "false_accept"]
        fr_df = df[df["failure_type"] == "false_reject"]

        context["false_accepts"] = fa_df.to_dict("records") if not fa_df.empty else []
        context["false_rejects"] = fr_df.to_dict("records") if not fr_df.empty else []
        context["failures_available"] = True

    if subgroup_csv.exists():
        import pandas as pd

        df = pd.read_csv(subgroup_csv)
        context["subgroup_metrics"] = df.to_dict("records")
        context["subgroups_available"] = True

    return render(request, "recognition/admin/failure_analysis.html", context)


@staff_member_required
def system_health_dashboard(request: HttpRequest) -> HttpResponse:
    """Render current webcam and recognition health signals for admins."""

    snapshot = monitoring.get_health_snapshot()
    dataset = health.dataset_health()
    model = health.model_health(dataset_last_updated=dataset.get("last_updated"))
    recognition_state = health.recognition_activity()
    worker_state = health.worker_health()
    context = {
        "snapshot": snapshot,
        "dataset": dataset,
        "model": model,
        "recognition_state": recognition_state,
        "worker_state": worker_state,
        "metrics_url": request.build_absolute_uri(reverse("monitoring-metrics")),
    }
    return render(request, "recognition/admin/system_health.html", context)


# Reusable filter for unknown face attempts
def _unknown_face_filter() -> Q:
    """Return a Q object to filter for unknown face attempts."""
    return (
        Q(successful=False)
        & ~Q(spoof_detected=True)
        & (Q(username="") | Q(username__isnull=True))
    )


# Default limit for dashboard record display
DASHBOARD_RECORD_LIMIT = 100


def _get_chart_data_for_period(days: int = 7) -> dict[str, Any]:
    """Aggregate recognition outcomes for the specified period for chart display."""
    since = timezone.now() - datetime.timedelta(days=days)

    # Use simpler aggregation to avoid SQLite compatibility issues
    outcomes = list(
        RecognitionOutcome.objects.filter(created_at__gte=since)
        .values("created_at", "accepted")
    )

    # Get liveness failure data from RecognitionAttempt
    liveness_attempts = list(
        RecognitionAttempt.objects.filter(created_at__gte=since, spoof_detected=True)
        .values("created_at")
    )

    # Get unknown face attempts (failed attempts with no username)
    unknown_attempts = list(
        RecognitionAttempt.objects.filter(created_at__gte=since)
        .filter(_unknown_face_filter())
        .values("created_at")
    )

    # Aggregate by day in Python to avoid SQLite EXPLAIN issues with aggregates
    daily_check_ins: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.get("accepted"):
            day = outcome["created_at"].date().isoformat()
            daily_check_ins[day] = daily_check_ins.get(day, 0) + 1

    daily_liveness: dict[str, int] = {}
    for attempt in liveness_attempts:
        day = attempt["created_at"].date().isoformat()
        daily_liveness[day] = daily_liveness.get(day, 0) + 1

    daily_unknown: dict[str, int] = {}
    for attempt in unknown_attempts:
        day = attempt["created_at"].date().isoformat()
        daily_unknown[day] = daily_unknown.get(day, 0) + 1

    # Collect all unique days and sort them
    all_days = set(daily_check_ins.keys()) | set(daily_liveness.keys()) | set(daily_unknown.keys())
    sorted_days = sorted(all_days)

    return {
        "labels": sorted_days,
        "check_ins": [daily_check_ins.get(day, 0) for day in sorted_days],
        "liveness_failures": [daily_liveness.get(day, 0) for day in sorted_days],
        "unknown_faces": [daily_unknown.get(day, 0) for day in sorted_days],
    }


def _get_summary_stats(date_from: datetime.date | None, date_to: datetime.date | None) -> dict:
    """Calculate summary statistics for the given date range."""
    filters = Q()

    if date_from:
        filters &= Q(created_at__date__gte=date_from)
    if date_to:
        filters &= Q(created_at__date__lte=date_to)

    outcomes = RecognitionOutcome.objects.filter(filters)
    attempts = RecognitionAttempt.objects.filter(filters)

    total_outcomes = outcomes.count()
    accepted = outcomes.filter(accepted=True).count()
    rejected = outcomes.filter(accepted=False).count()
    liveness_failures = attempts.filter(spoof_detected=True).count()
    unknown_faces = attempts.filter(_unknown_face_filter()).count()

    acceptance_rate = 0
    if total_outcomes:
        acceptance_rate = round((accepted or 0) / total_outcomes * 100, 1)

    return {
        "total_outcomes": total_outcomes,
        "accepted": accepted,
        "rejected": rejected,
        "liveness_failures": liveness_failures,
        "unknown_faces": unknown_faces,
        "acceptance_rate": acceptance_rate,
    }


@staff_member_required(login_url="login")
def attendance_dashboard(request: HttpRequest) -> HttpResponse:
    """Render attendance dashboard with filters, search, and visual charts."""
    form = AttendanceSessionFilterForm(request.GET or None)

    # Default to last 7 days if no filter is provided
    date_from = None
    date_to = None
    employee = None
    outcome_filter = None

    if form.is_valid():
        date_from = form.cleaned_data.get("date_from")
        date_to = form.cleaned_data.get("date_to")
        employee = form.cleaned_data.get("employee")
        outcome_filter = form.cleaned_data.get("outcome")

    # Build queryset filters
    outcome_filters = Q()
    attempt_filters = Q()

    if date_from:
        outcome_filters &= Q(created_at__date__gte=date_from)
        attempt_filters &= Q(created_at__date__gte=date_from)
    if date_to:
        outcome_filters &= Q(created_at__date__lte=date_to)
        attempt_filters &= Q(created_at__date__lte=date_to)
    if employee:
        outcome_filters &= Q(username__icontains=employee)
        attempt_filters &= Q(username__icontains=employee) | Q(user__username__icontains=employee)

    # Apply outcome filter - track if we should skip outcomes entirely
    skip_outcomes = False
    if outcome_filter == "success":
        outcome_filters &= Q(accepted=True)
        attempt_filters &= Q(successful=True)
    elif outcome_filter == "liveness_fail":
        attempt_filters &= Q(spoof_detected=True)
        # Liveness failures are only tracked in attempts, not outcomes
        skip_outcomes = True
    elif outcome_filter == "low_confidence":
        outcome_filters &= Q(accepted=False)
        attempt_filters &= Q(successful=False) & Q(spoof_detected=False)

    # Fetch filtered data (limit configurable via DASHBOARD_RECORD_LIMIT)
    if skip_outcomes:
        outcomes = RecognitionOutcome.objects.none()
    else:
        outcomes = (
            RecognitionOutcome.objects.filter(outcome_filters)
            .order_by("-created_at")[:DASHBOARD_RECORD_LIMIT]
        )
    attempts = (
        RecognitionAttempt.objects.filter(attempt_filters)
        .order_by("-created_at")[:DASHBOARD_RECORD_LIMIT]
    )

    # Get chart data
    chart_data = _get_chart_data_for_period(days=7)
    summary_stats = _get_summary_stats(date_from, date_to)

    context = {
        "form": form,
        "outcomes": outcomes,
        "attempts": attempts,
        "chart_data_json": json.dumps(chart_data),
        "summary_stats": summary_stats,
    }
    return render(request, "recognition/admin/attendance_dashboard.html", context)


@staff_member_required(login_url="login")
def export_attendance_csv(request: HttpRequest) -> HttpResponse:
    """Export filtered attendance data as CSV."""
    form = AttendanceSessionFilterForm(request.GET or None)

    date_from = None
    date_to = None
    employee = None
    outcome_filter = None

    if form.is_valid():
        date_from = form.cleaned_data.get("date_from")
        date_to = form.cleaned_data.get("date_to")
        employee = form.cleaned_data.get("employee")
        outcome_filter = form.cleaned_data.get("outcome")

    # Build queryset filters for outcomes
    outcome_filters = Q()
    if date_from:
        outcome_filters &= Q(created_at__date__gte=date_from)
    if date_to:
        outcome_filters &= Q(created_at__date__lte=date_to)
    if employee:
        outcome_filters &= Q(username__icontains=employee)
    if outcome_filter == "success":
        outcome_filters &= Q(accepted=True)
    elif outcome_filter == "low_confidence":
        outcome_filters &= Q(accepted=False)

    outcomes = RecognitionOutcome.objects.filter(outcome_filters).order_by("-created_at")

    # Build queryset filters for attempts
    attempt_filters = Q()
    if date_from:
        attempt_filters &= Q(created_at__date__gte=date_from)
    if date_to:
        attempt_filters &= Q(created_at__date__lte=date_to)
    if employee:
        attempt_filters &= Q(username__icontains=employee) | Q(user__username__icontains=employee)
    if outcome_filter == "success":
        attempt_filters &= Q(successful=True)
    elif outcome_filter == "liveness_fail":
        attempt_filters &= Q(spoof_detected=True)
    elif outcome_filter == "low_confidence":
        attempt_filters &= Q(successful=False) & Q(spoof_detected=False)

    attempts = RecognitionAttempt.objects.filter(attempt_filters).order_by("-created_at")

    # Create CSV response
    response = HttpResponse(content_type="text/csv")
    filename = f"attendance_report_{timezone.now().strftime('%Y%m%d_%H%M%S')}.csv"
    response["Content-Disposition"] = f'attachment; filename="{filename}"'

    writer = csv.writer(response)
    writer.writerow(
        [
            "Timestamp",
            "Username",
            "Direction",
            "Status",
            "Confidence",
            "Distance",
            "Threshold",
            "Liveness",
            "Source",
            "Error Message",
        ]
    )

    # Write outcomes
    for outcome in outcomes:
        writer.writerow(
            [
                outcome.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                outcome.username,
                outcome.direction,
                "Accepted" if outcome.accepted else "Rejected",
                f"{outcome.confidence:.4f}" if outcome.confidence is not None else "",
                f"{outcome.distance:.4f}" if outcome.distance is not None else "",
                f"{outcome.threshold:.4f}" if outcome.threshold is not None else "",
                "",
                outcome.source,
                "",
            ]
        )

    # Write attempts (only those not already covered by outcomes)
    for attempt in attempts:
        liveness_status = "Failed" if attempt.spoof_detected else "Passed"
        writer.writerow(
            [
                attempt.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                attempt.username or (attempt.user.username if attempt.user else ""),
                attempt.direction,
                "Success" if attempt.successful else "Failed",
                "",
                "",
                "",
                liveness_status,
                attempt.source,
                attempt.error_message,
            ]
        )

    return response
