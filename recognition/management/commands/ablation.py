"""
Django management command to run ablation experiments.
"""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from recognition.ablation import run_ablation_study
from src.common.seeding import set_global_seed


class Command(BaseCommand):
    help = "Run ablation experiments on face recognition components"

    def add_arguments(self, parser):
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility",
        )

    def handle(self, *args, **options):
        set_global_seed(options["seed"])

        self.stdout.write("Running ablation experiments...")

        # Get test dataset (in production, this would load actual test data)
        data_root = (
            Path(settings.BASE_DIR) / "face_recognition_data" / "training_dataset"
        )

        if not data_root.exists():
            self.stdout.write(
                self.style.WARNING(
                    "Training dataset not found. Creating minimal sample for ablation..."
                )
            )
            data_root.mkdir(parents=True, exist_ok=True)
            # Create sample structure
            for person in ["user1", "user2", "user3"]:
                person_dir = data_root / person
                person_dir.mkdir(exist_ok=True)
                for i in range(5):
                    (person_dir / f"{i}.jpg").touch()

        # Collect image paths and labels
        image_paths = list(data_root.glob("*/*.jpg"))
        labels = [p.parent.name for p in image_paths]

        if len(image_paths) == 0:
            self.stdout.write(
                self.style.ERROR("No images found. Cannot run ablation study.")
            )
            return

        self.stdout.write(f"Running ablation on {len(image_paths)} images...")

        # Run ablation study
        reports_dir = Path(settings.BASE_DIR) / "reports"
        results_df = run_ablation_study(
            image_paths, labels, reports_dir, random_state=options["seed"]
        )

        # Display summary
        self.stdout.write(self.style.SUCCESS("\n✓ Ablation experiments complete"))
        self.stdout.write("\nResults summary:")
        self.stdout.write(results_df.to_string(index=False))
        self.stdout.write(
            f"\nDetailed results saved to: {reports_dir / 'ablation_results.csv'}"
        )
        self.stdout.write(f"Narrative report: {reports_dir / 'ABLATIONS.md'}")
