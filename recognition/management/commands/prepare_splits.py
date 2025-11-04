"""
Django management command to prepare dataset splits.
"""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from recognition.data_splits import (
    create_stratified_splits,
    save_split_summary_json,
    save_splits_to_csv,
)
from src.common.seeding import set_global_seed


class Command(BaseCommand):
    help = "Prepare stratified train/val/test splits with leakage prevention"

    def add_arguments(self, parser):
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility",
        )
        parser.add_argument(
            "--train-ratio",
            type=float,
            default=0.70,
            help="Proportion of data for training",
        )
        parser.add_argument(
            "--val-ratio",
            type=float,
            default=0.15,
            help="Proportion of data for validation",
        )
        parser.add_argument(
            "--test-ratio",
            type=float,
            default=0.15,
            help="Proportion of data for testing",
        )

    def handle(self, *args, **options):
        set_global_seed(options["seed"])

        self.stdout.write("Preparing dataset splits...")

        # Get training dataset root
        data_root = Path(settings.BASE_DIR) / "face_recognition_data" / "training_dataset"

        if not data_root.exists():
            self.stdout.write(
                self.style.WARNING(
                    f"Training dataset not found at {data_root}. Creating sample structure..."
                )
            )
            # Create a minimal sample structure for testing
            data_root.mkdir(parents=True, exist_ok=True)
            for person in ["user1", "user2", "user3"]:
                person_dir = data_root / person
                person_dir.mkdir(exist_ok=True)
                # Create dummy files
                for i in range(10):
                    (person_dir / f"{i}.jpg").touch()

        # Collect all image paths
        image_paths = list(data_root.glob("*/*.jpg"))

        if len(image_paths) == 0:
            self.stdout.write(
                self.style.ERROR("No images found in training dataset. Cannot create splits.")
            )
            return

        self.stdout.write(f"Found {len(image_paths)} images")

        # Create splits
        try:
            train_paths, val_paths, test_paths, split_info = create_stratified_splits(
                image_paths,
                train_ratio=options["train_ratio"],
                val_ratio=options["val_ratio"],
                test_ratio=options["test_ratio"],
                random_state=options["seed"],
            )

            # Save splits
            reports_dir = Path(settings.BASE_DIR) / "reports"
            reports_dir.mkdir(exist_ok=True)

            save_splits_to_csv(train_paths, val_paths, test_paths, reports_dir / "splits.csv")
            save_split_summary_json(split_info, reports_dir / "split_summary.json")

            self.stdout.write(self.style.SUCCESS("✓ Splits saved successfully"))
            self.stdout.write(f"  - Train: {len(train_paths)} images")
            self.stdout.write(f"  - Val: {len(val_paths)} images")
            self.stdout.write(f"  - Test: {len(test_paths)} images")
            self.stdout.write(f"  - Output: {reports_dir}")

        except ValueError as e:
            self.stdout.write(self.style.ERROR(f"Error creating splits: {e}"))
