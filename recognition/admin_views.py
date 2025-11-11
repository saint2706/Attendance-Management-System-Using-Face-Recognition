"""
Custom admin views for evaluation metrics and reports.
"""

import json
from pathlib import Path

from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate, TruncWeek
from django.shortcuts import render
from django.urls import reverse


from . import monitoring

from .models import RecognitionOutcome

from users.models import RecognitionAttempt


@staff_member_required
def evaluation_dashboard(request):
    """
    Display evaluation metrics with confidence intervals and links to figures.
    """
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


def _prepare_accuracy_trend(time_trunc_func):
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
def recognition_accuracy_trends(request):
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
def ablation_results(request):
    """
    Display ablation experiment results.
    """
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
def failure_analysis(request):
    """
    Display failure analysis results.
    """
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
def system_health_dashboard(request):
    """Render current webcam and recognition health signals for admins."""

    snapshot = monitoring.get_health_snapshot()
    context = {
        "snapshot": snapshot,
        "metrics_url": request.build_absolute_uri(reverse("monitoring-metrics")),
    }
    return render(request, "recognition/admin/system_health.html", context)
