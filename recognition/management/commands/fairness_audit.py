"""Management command that runs the fairness & robustness audit."""

from pathlib import Path

from django.core.management.base import BaseCommand

from src.common.seeding import set_global_seed
from src.evaluation.fairness import (
    FairnessAuditConfig,
    compute_threshold_recommendations,
    run_fairness_audit,
    write_threshold_recommendations_csv,
)


class Command(BaseCommand):
    help = "Run fairness and robustness metrics across the evaluation set"

    def add_arguments(self, parser):
        parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
        parser.add_argument(
            "--reports-dir",
            type=Path,
            default=None,
            help="Directory that will store fairness outputs (defaults to reports/fairness)",
        )
        parser.add_argument(
            "--dataset-root",
            type=Path,
            default=None,
            help="Optional override for the encrypted dataset root",
        )
        parser.add_argument(
            "--split-csv",
            type=Path,
            default=None,
            help="CSV produced by prepare_splits.py to lock the audit to the test split",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=None,
            help="Distance threshold override (defaults to the project setting)",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            default=None,
            help="Limit the number of samples processed (useful for smoke tests)",
        )
        parser.add_argument(
            "--recommend-thresholds",
            action="store_true",
            help="Generate per-group threshold recommendations based on FAR/FRR metrics",
        )
        parser.add_argument(
            "--frr-threshold",
            type=float,
            default=0.15,
            help="FRR above this value triggers a looser threshold recommendation (default: 0.15)",
        )
        parser.add_argument(
            "--far-threshold",
            type=float,
            default=0.05,
            help="FAR above this value triggers a stricter threshold recommendation (default: 0.05)",
        )

    def handle(self, *args, **options):
        set_global_seed(options["seed"])

        self.stdout.write("Running fairness & robustness audit...")
        config = FairnessAuditConfig(
            reports_dir=options["reports_dir"],
            dataset_root=options["dataset_root"],
            test_split_csv=options["split_csv"],
            threshold=options["threshold"],
            limit_samples=options["max_samples"],
        )

        result = run_fairness_audit(config)

        self.stdout.write(self.style.SUCCESS("\n✓ Fairness audit complete"))
        self.stdout.write(f"Summary: {result.summary_path}")
        self.stdout.write("Group metrics:")
        for name, metrics in result.group_metrics.items():
            self.stdout.write(f"  - {name}: {metrics.csv_path}")
        self.stdout.write(f"Evaluation artifacts: {config.evaluation_reports_dir}")

        # Generate threshold recommendations if requested
        if options["recommend_thresholds"]:
            threshold = options["threshold"] or 0.4  # Use specified or default
            recommendations = compute_threshold_recommendations(
                result.group_metrics,
                current_threshold=threshold,
                frr_threshold=options["frr_threshold"],
                far_threshold=options["far_threshold"],
            )

            if recommendations:
                csv_path = config.reports_dir / "threshold_recommendations.csv"
                write_threshold_recommendations_csv(recommendations, csv_path)
                self.stdout.write(
                    self.style.SUCCESS(f"\n✓ Generated {len(recommendations)} threshold recommendations")
                )
                self.stdout.write(f"Recommendations: {csv_path}")

                self.stdout.write("\nRecommended adjustments:")
                for rec in recommendations:
                    self.stdout.write(
                        f"  - {rec.group_type}:{rec.group_value} → "
                        f"{rec.recommended_threshold:.4f} ({rec.adjustment_reason})"
                    )
            else:
                self.stdout.write(
                    self.style.SUCCESS("\n✓ No threshold adjustments needed - all groups within acceptable ranges")
                )

