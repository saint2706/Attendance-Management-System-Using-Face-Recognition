"""
Custom admin views for evaluation metrics and reports.
"""

import json
from pathlib import Path

from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Count, Q
from django.db.models.functions import Coalesce
from django.shortcuts import render

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
def recognition_attempt_summary(request):
    """Display aggregated recognition attempt metrics per site and employee."""

    attempts = RecognitionAttempt.objects.all()

    site_summary = list(
        attempts.values("site")
        .annotate(
            total=Count("id"),
            successes=Count("id", filter=Q(successful=True)),
            failures=Count("id", filter=Q(successful=False)),
            spoofed=Count("id", filter=Q(spoof_detected=True)),
        )
        .order_by("site")
    )
    for entry in site_summary:
        total = entry.get("total") or 0
        entry["success_rate"] = (
            (entry.get("successes", 0) / total) * 100 if total else None
        )

    employee_summary = list(
        attempts.annotate(
            resolved_username=Coalesce("user__username", "username")
        )
        .values("site", "resolved_username")
        .annotate(
            total=Count("id"),
            successes=Count("id", filter=Q(successful=True)),
            failures=Count("id", filter=Q(successful=False)),
            spoofed=Count("id", filter=Q(spoof_detected=True)),
        )
        .order_by("site", "resolved_username")
    )
    for entry in employee_summary:
        total = entry.get("total") or 0
        entry["success_rate"] = (
            (entry.get("successes", 0) / total) * 100 if total else None
        )

    context = {
        "site_summary": site_summary,
        "employee_summary": employee_summary,
    }

    return render(request, "recognition/admin/recognition_attempt_summary.html", context)
