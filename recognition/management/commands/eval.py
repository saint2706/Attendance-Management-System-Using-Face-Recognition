"""Django management command that runs the face-recognition evaluation suite."""

from pathlib import Path

from django.core.management.base import BaseCommand

from src.common.seeding import set_global_seed
from src.evaluation.face_recognition_eval import (
    EvaluationConfig,
    run_face_recognition_evaluation,
)


class Command(BaseCommand):
    help = "Run the face recognition evaluation pipeline"

    def add_arguments(self, parser):
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=None,
            help="Distance threshold override (defaults to project settings)",
        )
        parser.add_argument(
            "--reports-dir",
            type=Path,
            default=None,
            help="Directory that will store evaluation outputs",
        )
        parser.add_argument(
            "--split-csv",
            type=Path,
            default=None,
            help="CSV produced by prepare_splits.py to limit evaluation to the test split",
        )
        parser.add_argument(
            "--dataset-root",
            type=Path,
            default=None,
            help="Override the dataset root. Defaults to the encrypted training dataset",
        )
        parser.add_argument("--threshold-start", type=float, default=0.2)
        parser.add_argument("--threshold-stop", type=float, default=1.0)
        parser.add_argument("--threshold-step", type=float, default=0.05)
        parser.add_argument(
            "--max-samples",
            type=int,
            default=None,
            help="Limit the number of samples processed (useful for smoke tests)",
        )

    def handle(self, *args, **options):
        set_global_seed(options["seed"])

        self.stdout.write("Running face recognition evaluation...")
        config = EvaluationConfig(
            reports_dir=options["reports_dir"],
            test_split_csv=options["split_csv"],
            dataset_root=options["dataset_root"],
            threshold=options["threshold"],
            threshold_start=options["threshold_start"],
            threshold_stop=options["threshold_stop"],
            threshold_step=options["threshold_step"],
            limit_samples=options["max_samples"],
        )

        summary = run_face_recognition_evaluation(config)

        metrics = summary.metrics
        self.stdout.write(self.style.SUCCESS("\n✓ Evaluation complete"))
        self.stdout.write("\nMetrics:")
        self.stdout.write(f"  - Accuracy: {metrics['accuracy']:.4f}")
        self.stdout.write(f"  - Precision (macro): {metrics['precision']:.4f}")
        self.stdout.write(f"  - Recall (macro): {metrics['recall']:.4f}")
        self.stdout.write(f"  - F1 (macro): {metrics['f1']:.4f}")
        self.stdout.write(f"  - FAR: {metrics['far']:.4f}")
        self.stdout.write(f"  - FRR: {metrics['frr']:.4f}")

        self.stdout.write("\nArtifacts:")
        for label, path in summary.artifact_paths.items():
            self.stdout.write(f"  - {label}: {path}")
