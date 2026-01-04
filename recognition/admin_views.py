"""Admin views for evaluation metrics, reports, and health dashboards."""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Callable

from django.conf import settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate, TruncWeek
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone

from users.models import RecognitionAttempt

from . import health, monitoring
from .forms import AttendanceSessionFilterForm, ThresholdImportForm, ThresholdProfileForm
from .models import LivenessResult, RecognitionOutcome, ThresholdProfile


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
    evaluation_state = health.evaluation_health()
    context = {
        "snapshot": snapshot,
        "dataset": dataset,
        "model": model,
        "recognition_state": recognition_state,
        "worker_state": worker_state,
        "evaluation_state": evaluation_state,
        "metrics_url": request.build_absolute_uri(reverse("monitoring-metrics")),
    }
    return render(request, "recognition/admin/system_health.html", context)


# Reusable filter for unknown face attempts
def _unknown_face_filter() -> Q:
    """Return a Q object to filter for unknown face attempts."""
    return (
        Q(successful=False) & ~Q(spoof_detected=True) & (Q(username="") | Q(username__isnull=True))
    )


# Default limit for dashboard record display
DASHBOARD_RECORD_LIMIT = 100


def _sanitize_csv_value(value: str) -> str:
    """Sanitize a value to prevent CSV injection (formula injection)."""
    if not value:
        return ""
    value = str(value)
    if value.startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def _get_chart_data_for_period(days: int = 7) -> dict[str, Any]:
    """Aggregate recognition outcomes for the specified period for chart display."""
    since = timezone.now() - datetime.timedelta(days=days)

    # Use simpler aggregation to avoid SQLite compatibility issues
    outcomes = list(
        RecognitionOutcome.objects.filter(created_at__gte=since).values("created_at", "accepted")
    )

    # Get liveness failure data from RecognitionAttempt
    liveness_attempts = list(
        RecognitionAttempt.objects.filter(created_at__gte=since, spoof_detected=True).values(
            "created_at"
        )
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
        outcomes = RecognitionOutcome.objects.filter(outcome_filters).order_by("-created_at")[
            :DASHBOARD_RECORD_LIMIT
        ]
    attempts = RecognitionAttempt.objects.filter(attempt_filters).order_by("-created_at")[
        :DASHBOARD_RECORD_LIMIT
    ]

    # Get chart data
    chart_data = _get_chart_data_for_period(days=7)
    summary_stats = _get_summary_stats(date_from, date_to)

    context = {
        "form": form,
        "outcomes": outcomes,
        "attempts": attempts,
        "chart_data_json": chart_data,
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
                _sanitize_csv_value(outcome.username),
                outcome.direction,
                "Accepted" if outcome.accepted else "Rejected",
                f"{outcome.confidence:.4f}" if outcome.confidence is not None else "",
                f"{outcome.distance:.4f}" if outcome.distance is not None else "",
                f"{outcome.threshold:.4f}" if outcome.threshold is not None else "",
                "",
                _sanitize_csv_value(outcome.source),
                "",
            ]
        )

    # Write attempts (only those not already covered by outcomes)
    for attempt in attempts:
        liveness_status = "Failed" if attempt.spoof_detected else "Passed"
        writer.writerow(
            [
                attempt.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                _sanitize_csv_value(attempt.username or (attempt.user.username if attempt.user else "")),
                attempt.direction,
                "Success" if attempt.successful else "Failed",
                "",
                "",
                "",
                liveness_status,
                _sanitize_csv_value(attempt.source),
                _sanitize_csv_value(attempt.error_message),
            ]
        )

    return response


@staff_member_required(login_url="login")
def fairness_dashboard(request: HttpRequest) -> HttpResponse:
    """Render model fairness & limitations dashboard for admins."""
    reports_dir = Path(settings.BASE_DIR) / "reports" / "fairness"

    context: dict[str, Any] = {
        "fairness_available": False,
        "last_audit_date": None,
        "summary_content": None,
        "group_metrics": {},
        "flagged_groups": [],
        "overall_metrics": {},
    }

    summary_path = reports_dir / "summary.md"
    if summary_path.exists():
        context["fairness_available"] = True

        # Get last audit date from file modification time
        mtime = summary_path.stat().st_mtime
        context["last_audit_date"] = datetime.datetime.fromtimestamp(
            mtime, tz=datetime.timezone.utc
        )

        # Parse summary.md for overall metrics
        summary_content = summary_path.read_text(encoding="utf-8")
        context["summary_content"] = summary_content

        # Extract overall metrics from markdown table
        overall_metrics = {}
        in_overall_section = False
        for line in summary_content.split("\n"):
            if "## Overall evaluation" in line:
                in_overall_section = True
                continue
            if in_overall_section and line.startswith("| ") and " | " in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) == 2 and parts[0] not in ("Metric", "---"):
                    overall_metrics[parts[0]] = parts[1]
            if in_overall_section and line.startswith("## ") and "Overall" not in line:
                in_overall_section = False
        context["overall_metrics"] = overall_metrics

    # Load group metrics from CSV files
    csv_files = {
        "by_role": "metrics_by_role.csv",
        "by_site": "metrics_by_site.csv",
        "by_source": "metrics_by_source.csv",
        "by_lighting": "metrics_by_lighting.csv",
    }

    flagged_groups = []
    far_threshold = 0.10  # Flag groups with FAR > 10%
    frr_threshold = 0.15  # Flag groups with FRR > 15%

    for key, filename in csv_files.items():
        csv_path = reports_dir / filename
        if csv_path.exists():
            rows = []
            with open(csv_path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
                    # Check for flagged groups (high FAR or FRR)
                    try:
                        far = float(row.get("far", 0))
                        frr = float(row.get("frr", 0))
                        group_name = row.get("group", "unknown")
                        if far > far_threshold or frr > frr_threshold:
                            flagged_groups.append(
                                {
                                    "category": key.replace("by_", "").title(),
                                    "group": group_name,
                                    "far": f"{far:.4f}",
                                    "frr": f"{frr:.4f}",
                                    "reason": ("High FAR" if far > far_threshold else "")
                                    + (
                                        " & High FRR"
                                        if far > far_threshold and frr > frr_threshold
                                        else ("High FRR" if frr > frr_threshold else "")
                                    ),
                                }
                            )
                    except (ValueError, TypeError):
                        pass
            context["group_metrics"][key] = rows

    context["flagged_groups"] = flagged_groups

    # Check for report files to link
    report_files = []
    if summary_path.exists():
        report_files.append({"name": "Summary Report", "path": "summary.md"})
    for key, filename in csv_files.items():
        csv_path = reports_dir / filename
        if csv_path.exists():
            report_files.append(
                {
                    "name": f"Metrics by {key.replace('by_', '').title()}",
                    "path": filename,
                }
            )
    context["report_files"] = report_files

    return render(request, "recognition/admin/fairness_dashboard.html", context)


@staff_member_required(login_url="login")
def threshold_profiles(request: HttpRequest) -> HttpResponse:
    """List and manage threshold profiles."""
    profiles = ThresholdProfile.objects.all()

    # Get the current system default threshold
    system_default = getattr(settings, "RECOGNITION_DISTANCE_THRESHOLD", 0.4)

    # Load threshold sweep data if available
    reports_dir = Path(settings.BASE_DIR) / "reports"
    threshold_file = reports_dir / "selected_threshold.json"
    sweep_data = None

    if threshold_file.exists():
        try:
            with open(threshold_file) as f:
                sweep_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    context = {
        "profiles": profiles,
        "system_default": system_default,
        "sweep_data": sweep_data,
        "has_sweep_data": sweep_data is not None,
    }

    return render(request, "recognition/admin/threshold_profiles.html", context)


@staff_member_required(login_url="login")
def threshold_profile_create(request: HttpRequest) -> HttpResponse:
    """Create a new threshold profile."""
    if request.method == "POST":
        form = ThresholdProfileForm(request.POST)
        if form.is_valid():
            profile = form.save()
            messages.success(
                request,
                f"Created threshold profile '{profile.name}' with threshold {profile.distance_threshold:.4f}",
            )
            return redirect("admin_threshold_profiles")
    else:
        form = ThresholdProfileForm()

    context = {"form": form, "action": "Create"}
    return render(request, "recognition/admin/threshold_profile_form.html", context)


@staff_member_required(login_url="login")
def threshold_profile_edit(request: HttpRequest, profile_id: int) -> HttpResponse:
    """Edit an existing threshold profile."""
    try:
        profile = ThresholdProfile.objects.get(pk=profile_id)
    except ThresholdProfile.DoesNotExist:
        messages.error(request, "Profile not found.")
        return redirect("admin_threshold_profiles")

    if request.method == "POST":
        form = ThresholdProfileForm(request.POST, instance=profile)
        if form.is_valid():
            profile = form.save()
            messages.success(request, f"Updated threshold profile '{profile.name}'")
            return redirect("admin_threshold_profiles")
    else:
        form = ThresholdProfileForm(instance=profile)

    context = {"form": form, "action": "Edit", "profile": profile}
    return render(request, "recognition/admin/threshold_profile_form.html", context)


@staff_member_required(login_url="login")
def threshold_profile_delete(request: HttpRequest, profile_id: int) -> HttpResponse:
    """Delete a threshold profile."""
    try:
        profile = ThresholdProfile.objects.get(pk=profile_id)
    except ThresholdProfile.DoesNotExist:
        messages.error(request, "Profile not found.")
        return redirect("admin_threshold_profiles")

    if request.method == "POST":
        name = profile.name
        profile.delete()
        messages.success(request, f"Deleted threshold profile '{name}'")
        return redirect("admin_threshold_profiles")

    context = {"profile": profile}
    return render(request, "recognition/admin/threshold_profile_delete.html", context)


@staff_member_required(login_url="login")
def threshold_profile_set_default(request: HttpRequest, profile_id: int) -> HttpResponse:
    """Set a profile as the default."""
    try:
        profile = ThresholdProfile.objects.get(pk=profile_id)
    except ThresholdProfile.DoesNotExist:
        messages.error(request, "Profile not found.")
        return redirect("admin_threshold_profiles")

    profile.is_default = True
    profile.save()
    messages.success(request, f"Set '{profile.name}' as the default profile")
    return redirect("admin_threshold_profiles")


@staff_member_required(login_url="login")
def threshold_profile_import(request: HttpRequest) -> HttpResponse:
    """Import a threshold from evaluation artifacts."""
    reports_dir = Path(settings.BASE_DIR) / "reports"
    threshold_file = reports_dir / "selected_threshold.json"

    sweep_data = None
    if threshold_file.exists():
        try:
            with open(threshold_file) as f:
                sweep_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    if request.method == "POST":
        form = ThresholdImportForm(request.POST)
        if form.is_valid():
            method = form.cleaned_data["import_method"]
            name = form.cleaned_data["profile_name"]
            sites = form.cleaned_data.get("sites", "")
            set_default = form.cleaned_data.get("set_as_default", False)

            # Check if profile name already exists
            if ThresholdProfile.objects.filter(name=name).exists():
                messages.error(request, f"Profile '{name}' already exists.")
                return redirect("admin_threshold_profile_import")

            # Get threshold from sweep data
            if sweep_data is None:
                messages.error(request, "No threshold sweep data available. Run evaluation first.")
                return redirect("admin_threshold_profile_import")

            threshold = sweep_data.get("threshold")
            if threshold is None:
                messages.error(request, "Invalid threshold data.")
                return redirect("admin_threshold_profile_import")

            profile = ThresholdProfile.objects.create(
                name=name,
                description=f"Imported from evaluation using {method} method",
                distance_threshold=threshold,
                target_far=sweep_data.get("actual_far") or sweep_data.get("target_far"),
                target_frr=sweep_data.get("frr"),
                selection_method=method,
                sites=sites,
                is_default=set_default,
            )

            messages.success(
                request,
                f"Imported profile '{profile.name}' with threshold {profile.distance_threshold:.4f}",
            )
            return redirect("admin_threshold_profiles")
    else:
        form = ThresholdImportForm()

    context = {
        "form": form,
        "sweep_data": sweep_data,
        "has_sweep_data": sweep_data is not None,
    }
    return render(request, "recognition/admin/threshold_profile_import.html", context)


@staff_member_required(login_url="login")
def threshold_profile_api(request: HttpRequest) -> JsonResponse:
    """API endpoint for getting threshold by site code."""
    site_code = request.GET.get("site", "")

    profile = ThresholdProfile.get_for_site(site_code)
    threshold = ThresholdProfile.get_threshold_for_site(site_code)

    response_data = {
        "site": site_code,
        "threshold": threshold,
        "profile": None,
    }

    if profile:
        response_data["profile"] = {
            "id": profile.id,
            "name": profile.name,
            "distance_threshold": profile.distance_threshold,
            "is_default": profile.is_default,
        }

    return JsonResponse(response_data)


@staff_member_required(login_url="login")
def liveness_results_dashboard(request: HttpRequest) -> HttpResponse:
    """Dashboard for liveness check results and analytics."""
    # Get date range filter
    days = int(request.GET.get("days", 7))
    since = timezone.now() - datetime.timedelta(days=days)

    # Get liveness results
    results = LivenessResult.objects.filter(created_at__gte=since).order_by("-created_at")[:100]

    # Aggregate statistics
    total_checks = results.count()
    passed_count = results.filter(challenge_status="passed").count()
    failed_count = results.filter(challenge_status="failed").count()

    # Aggregate by challenge type with pass rates computed
    by_challenge_raw = (
        LivenessResult.objects.filter(created_at__gte=since)
        .values("challenge_type")
        .annotate(
            total=Count("id"),
            passed=Count("id", filter=Q(challenge_status="passed")),
            avg_confidence=Avg("liveness_confidence"),
            avg_motion=Avg("motion_score"),
        )
    )
    by_challenge = []
    for item in by_challenge_raw:
        item["pass_rate"] = (item["passed"] / item["total"] * 100) if item["total"] else 0
        by_challenge.append(item)

    # Daily trend with pass rates
    daily_trend_raw = (
        LivenessResult.objects.filter(created_at__gte=since)
        .annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(
            total=Count("id"),
            passed=Count("id", filter=Q(challenge_status="passed")),
            failed=Count("id", filter=Q(challenge_status="failed")),
        )
        .order_by("day")
    )
    daily_trend = []
    for item in daily_trend_raw:
        item["pass_rate"] = (item["passed"] / item["total"] * 100) if item["total"] else 0
        daily_trend.append(item)

    context = {
        "results": results,
        "total_checks": total_checks,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "pass_rate": (passed_count / total_checks * 100) if total_checks else 0,
        "by_challenge": by_challenge,
        "daily_trend": daily_trend,
        "days": days,
    }

    return render(request, "recognition/admin/liveness_results.html", context)
