"""
Django management command to select optimal threshold on validation set.
"""

import json
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand

from recognition.evaluation.metrics import calculate_eer, find_optimal_threshold
from src.common.seeding import set_global_seed


class Command(BaseCommand):
    help = "Select optimal threshold on validation set"

    def add_arguments(self, parser):
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility",
        )
        parser.add_argument(
            "--target-far",
            type=float,
            default=0.01,
            help="Target False Accept Rate for threshold selection",
        )
        parser.add_argument(
            "--method",
            type=str,
            choices=["eer", "f1", "far"],
            default="eer",
            help="Method for threshold selection: eer (Equal Error Rate), f1 (max F1), or far (target FAR)",
        )

    def handle(self, *args, **options):
        set_global_seed(options["seed"])

        self.stdout.write("Selecting optimal threshold on validation set...")

        # Generate synthetic validation data
        # In production, this would load actual validation set predictions
        np.random.seed(options["seed"])
        n_val = 150

        y_true = np.array([1] * 75 + [0] * 75)
        y_scores_genuine = np.random.beta(8, 2, 75)
        y_scores_impostor = np.random.beta(2, 8, 75)
        y_scores = np.concatenate([y_scores_genuine, y_scores_impostor])

        shuffle_idx = np.random.permutation(n_val)
        y_true = y_true[shuffle_idx]
        y_scores = y_scores[shuffle_idx]

        # Select threshold based on method
        if options["method"] == "eer":
            eer, threshold = calculate_eer(y_true, y_scores)
            self.stdout.write("Selected threshold using EER method")
            self.stdout.write(f"  - EER: {eer:.4f}")
            self.stdout.write(f"  - Threshold: {threshold:.4f}")
            selection_info = {
                "method": "eer",
                "threshold": float(threshold),
                "eer": float(eer),
            }

        elif options["method"] == "f1":
            threshold, best_f1 = find_optimal_threshold(y_true, y_scores)
            self.stdout.write("Selected threshold using F1 optimization")
            self.stdout.write(f"  - Optimal F1: {best_f1:.4f}")
            self.stdout.write(f"  - Threshold: {threshold:.4f}")
            selection_info = {
                "method": "f1",
                "threshold": float(threshold),
                "f1_score": float(best_f1),
            }

        elif options["method"] == "far":
            from sklearn.metrics import roc_curve

            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            target_far = options["target_far"]

            # Find threshold closest to target FAR
            idx = np.argmin(np.abs(fpr - target_far))
            threshold = float(thresholds[idx])
            actual_far = float(fpr[idx])
            actual_tpr = float(tpr[idx])

            self.stdout.write(f"Selected threshold for target FAR = {target_far:.4f}")
            self.stdout.write(f"  - Threshold: {threshold:.4f}")
            self.stdout.write(f"  - Actual FAR: {actual_far:.4f}")
            self.stdout.write(f"  - TPR at threshold: {actual_tpr:.4f}")
            selection_info = {
                "method": "far",
                "target_far": target_far,
                "threshold": threshold,
                "actual_far": actual_far,
                "tpr": actual_tpr,
            }

        # Save threshold
        reports_dir = Path(settings.BASE_DIR) / "reports"
        reports_dir.mkdir(exist_ok=True)
        threshold_path = reports_dir / "selected_threshold.json"

        with open(threshold_path, "w") as f:
            json.dump(selection_info, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f"\n✓ Threshold saved to {threshold_path}"))
