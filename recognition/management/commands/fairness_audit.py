"""Management command that runs the fairness & robustness audit."""

from pathlib import Path

from django.core.management.base import BaseCommand

from src.common.seeding import set_global_seed
from src.evaluation.fairness import FairnessAuditConfig, run_fairness_audit


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
