"""
Django management command to export all reports.
"""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Export all generated reports to a summary"

    def handle(self, *args, **options):
        self.stdout.write("Exporting reports...")

        reports_dir = Path(settings.BASE_DIR) / "reports"

        if not reports_dir.exists():
            self.stdout.write(
                self.style.WARNING("Reports directory not found. Run evaluation first.")
            )
            return

        # List all reports
        report_files = [
            "splits.csv",
            "split_summary.json",
            "metrics_with_ci.json",
            "metrics_with_ci.md",
            "ablation_results.csv",
            "ABLATIONS.md",
            "failure_cases.csv",
            "FAILURES.md",
            "subgroup_metrics.csv",
            "selected_threshold.json",
        ]

        self.stdout.write(self.style.SUCCESS("\nAvailable Reports:"))
        found_count = 0

        for report_file in report_files:
            report_path = reports_dir / report_file
            if report_path.exists():
                size_kb = report_path.stat().st_size / 1024
                self.stdout.write(f"  ✓ {report_file} ({size_kb:.1f} KB)")
                found_count += 1
            else:
                self.stdout.write(self.style.WARNING(f"  ✗ {report_file} (not found)"))

        # Check figures
        figures_dir = reports_dir / "figures"
        if figures_dir.exists():
            self.stdout.write("\nFigures:")
            for fig_file in ["roc.png", "pr.png", "det.png", "calibration.png"]:
                fig_path = figures_dir / fig_file
                if fig_path.exists():
                    size_kb = fig_path.stat().st_size / 1024
                    self.stdout.write(f"  ✓ {fig_file} ({size_kb:.1f} KB)")
                    found_count += 1

        self.stdout.write(f"\n{found_count} reports/figures found in {reports_dir}")
