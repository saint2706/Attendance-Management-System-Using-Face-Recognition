"""Management command to calibrate camera profiles for domain adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

from django.core.management.base import BaseCommand, CommandError

import numpy as np

if TYPE_CHECKING:
    from recognition.domain_adaptation import CameraProfile


class Command(BaseCommand):
    help = "Calibrate a camera by analyzing reference images to create a domain profile"

    def add_arguments(self, parser):
        parser.add_argument(
            "camera_name",
            type=str,
            help="Unique identifier for this camera (e.g., 'lobby_kiosk', 'reception_webcam')",
        )
        parser.add_argument(
            "--source-type",
            type=str,
            default="webcam",
            choices=["webcam", "kiosk", "mobile", "api", "other"],
            help="Type of camera source (default: webcam)",
        )
        parser.add_argument(
            "--images-dir",
            type=Path,
            default=None,
            help="Directory containing reference images for calibration",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory to save the camera profile (default: configs/camera_profiles/)",
        )
        parser.add_argument(
            "--compare-to",
            type=str,
            default=None,
            help="Name of existing camera profile to compare against",
        )

    def handle(self, *args, **options):
        from recognition.domain_adaptation import (
            estimate_camera_characteristics,
            get_camera_profiles_dir,
            save_camera_profile,
        )

        camera_name = options["camera_name"]
        source_type = options["source_type"]
        images_dir = options["images_dir"]
        output_dir = options["output_dir"] or get_camera_profiles_dir()
        compare_to = options["compare_to"]

        self.stdout.write(f"Calibrating camera: {camera_name}")

        # Load reference images
        images = self._load_images(images_dir)
        if not images:
            raise CommandError("No valid images found. Provide --images-dir with reference images.")

        self.stdout.write(f"Loaded {len(images)} reference images")

        # Estimate camera characteristics
        profile = estimate_camera_characteristics(
            images=images,
            camera_name=camera_name,
            source_type=source_type,
        )

        # Display profile summary
        self._display_profile(profile)

        # Save profile
        profile_path = Path(output_dir) / f"{camera_name}.json"
        save_camera_profile(profile, profile_path)
        self.stdout.write(self.style.SUCCESS(f"\n✓ Saved camera profile to {profile_path}"))

        # Compare to another profile if requested
        if compare_to:
            self._compare_profiles(profile, compare_to, output_dir)

    def _load_images(self, images_dir: Path | None) -> List[np.ndarray]:
        """Load images from directory or create synthetic test images."""
        images = []

        if images_dir and images_dir.exists():
            try:
                import cv2
            except ImportError:
                self.stderr.write(
                    self.style.WARNING("cv2 not available, using PIL for image loading")
                )
                cv2 = None

            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                for img_path in images_dir.glob(ext):
                    try:
                        if cv2:
                            img = cv2.imread(str(img_path))
                        else:
                            from PIL import Image

                            pil_img = Image.open(img_path)
                            img = np.array(pil_img)
                        if img is not None:
                            images.append(img)
                    except Exception as exc:
                        self.stderr.write(f"Failed to load {img_path}: {exc}")

        return images

    def _display_profile(self, profile: "CameraProfile") -> None:
        """Display camera profile details."""
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write("Camera Profile Summary")
        self.stdout.write("=" * 50)
        self.stdout.write(f"Name:              {profile.name}")
        self.stdout.write(f"Source Type:       {profile.source_type}")
        self.stdout.write(f"Mean Brightness:   {profile.mean_brightness:.1f}")
        self.stdout.write(f"Std Brightness:    {profile.std_brightness:.1f}")
        self.stdout.write(f"Mean Contrast:     {profile.mean_contrast:.1f}")
        self.stdout.write(f"Color Temperature: {profile.color_temperature}")
        self.stdout.write(f"Resolution:        {profile.resolution[0]}x{profile.resolution[1]}")
        self.stdout.write(f"Samples Analyzed:  {profile.calibration_samples}")
        self.stdout.write("=" * 50)

    def _compare_profiles(
        self, profile: "CameraProfile", compare_to: str, profiles_dir: Path
    ) -> None:
        """Compare the new profile to an existing one."""
        from recognition.domain_adaptation import assess_domain_gap, load_camera_profile

        compare_path = Path(profiles_dir) / f"{compare_to}.json"
        if not compare_path.exists():
            self.stderr.write(
                self.style.WARNING(f"Profile '{compare_to}' not found at {compare_path}")
            )
            return

        try:
            reference_profile = load_camera_profile(compare_path)
        except Exception as exc:
            self.stderr.write(f"Failed to load comparison profile: {exc}")
            return

        gap_result = assess_domain_gap(reference_profile, profile)

        self.stdout.write("\n" + "-" * 50)
        self.stdout.write(f"Domain Gap Analysis: {reference_profile.name} → {profile.name}")
        self.stdout.write("-" * 50)
        self.stdout.write(f"Brightness Shift:  {gap_result.brightness_shift:.1f}")
        self.stdout.write(f"Contrast Ratio:    {gap_result.contrast_ratio:.2f}x")
        self.stdout.write(
            f"Overall Gap Score: {gap_result.overall_gap_score:.2f} (0=identical, 1=very different)"
        )

        if gap_result.recommendations:
            self.stdout.write("\nRecommendations:")
            for rec in gap_result.recommendations:
                self.stdout.write(f"  • {rec}")
        else:
            self.stdout.write(
                self.style.SUCCESS("\n✓ Camera profiles are similar, no adjustments needed")
            )
