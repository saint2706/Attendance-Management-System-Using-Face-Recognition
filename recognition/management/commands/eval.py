"""
Django management command to run evaluation with metrics and confidence intervals.
"""

from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand

from recognition.evaluation.metrics import (
    bootstrap_confidence_intervals,
    calculate_verification_metrics,
    generate_metric_plots,
    save_metrics_json,
    save_metrics_markdown,
)
from src.common.seeding import set_global_seed


class Command(BaseCommand):
    help = "Run evaluation and generate metrics with confidence intervals"

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
            default=0.5,
            help="Classification threshold",
        )
        parser.add_argument(
            "--n-bootstrap",
            type=int,
            default=1000,
            help="Number of bootstrap samples for confidence intervals",
        )

    def handle(self, *args, **options):
        set_global_seed(options["seed"])

        self.stdout.write("Running evaluation...")

        # Generate synthetic evaluation data for demonstration
        # In production, this would load actual test set predictions
        np.random.seed(options["seed"])
        n_samples = 200

        # Simulate genuine (1) and impostor (0) pairs
        y_true = np.array([1] * 100 + [0] * 100)

        # Simulate scores (genuine pairs have higher scores on average)
        y_scores_genuine = np.random.beta(8, 2, 100)  # Skewed towards higher scores
        y_scores_impostor = np.random.beta(2, 8, 100)  # Skewed towards lower scores
        y_scores = np.concatenate([y_scores_genuine, y_scores_impostor])

        # Shuffle
        shuffle_idx = np.random.permutation(n_samples)
        y_true = y_true[shuffle_idx]
        y_scores = y_scores[shuffle_idx]

        # Calculate metrics
        self.stdout.write("Calculating verification metrics...")
        metrics = calculate_verification_metrics(
            y_true, y_scores, threshold=options["threshold"]
        )

        # Calculate confidence intervals
        self.stdout.write("Bootstrapping confidence intervals...")
        ci_results = bootstrap_confidence_intervals(
            y_true, y_scores, n_bootstrap=options["n_bootstrap"], random_state=options["seed"]
        )

        # Generate plots
        self.stdout.write("Generating metric plots...")
        reports_dir = Path(settings.BASE_DIR) / "reports"
        figures_dir = reports_dir / "figures"
        generate_metric_plots(y_true, y_scores, figures_dir)

        # Save metrics
        save_metrics_json(metrics, ci_results, reports_dir / "metrics_with_ci.json")
        save_metrics_markdown(metrics, ci_results, reports_dir / "metrics_with_ci.md")

        # Display summary
        self.stdout.write(self.style.SUCCESS("\n✓ Evaluation complete"))
        self.stdout.write("\nKey Metrics:")
        self.stdout.write(f"  - ROC AUC: {metrics['roc_auc']:.4f}")
        self.stdout.write(f"  - EER: {metrics['eer']:.4f}")
        self.stdout.write(f"  - Brier Score: {metrics['brier_score']:.4f}")
        self.stdout.write(f"  - Optimal F1: {metrics['optimal_f1']:.4f}")
        self.stdout.write("\nConfidence Intervals (95%):")
        if ci_results["auc"]["mean"] is not None:
            self.stdout.write(
                f"  - AUC: {ci_results['auc']['mean']:.4f} "
                f"[{ci_results['auc']['ci_lower']:.4f}, {ci_results['auc']['ci_upper']:.4f}]"
            )
        self.stdout.write(f"\nReports saved to: {reports_dir}")
        self.stdout.write(f"Figures saved to: {figures_dir}")
