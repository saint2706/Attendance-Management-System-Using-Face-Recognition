"""Evaluate the lightweight liveness detector on a curated dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

import cv2
import numpy as np

from recognition.liveness import is_live_face


class Command(BaseCommand):
    """Run motion-based liveness detection against genuine and spoof samples."""

    help = "Run the lightweight liveness detector on image bursts under a samples root."

    def add_arguments(self, parser) -> None:  # pragma: no cover - argparse wiring
        parser.add_argument(
            "--samples-root",
            dest="samples_root",
            default=str(Path(settings.BASE_DIR) / "liveness_samples"),
            help="Directory containing 'genuine/' and 'spoof/' sub-folders with frame sequences.",
        )
        parser.add_argument(
            "--threshold",
            dest="threshold",
            type=float,
            default=None,
            help="Override the motion threshold (defaults to settings.RECOGNITION_LIVENESS_MOTION_THRESHOLD).",
        )
        parser.add_argument(
            "--min-frames",
            dest="min_frames",
            type=int,
            default=None,
            help="Override the minimum frames required per sequence.",
        )

    def handle(self, *args, **options):
        samples_root = Path(options["samples_root"]).expanduser()
        threshold = options.get("threshold")
        min_frames = options.get("min_frames")

        if threshold is None:
            threshold = float(getattr(settings, "RECOGNITION_LIVENESS_MOTION_THRESHOLD", 1.1))
        if min_frames is None:
            min_frames = int(getattr(settings, "RECOGNITION_LIVENESS_MIN_FRAMES", 3))

        if not samples_root.exists():
            raise CommandError(
                f"Samples root '{samples_root}' does not exist. Capture a few sequences first."
            )

        categories = (("genuine", "accepted"), ("spoof", "rejected"))
        aggregate = {label: {"total": 0, "passes": 0} for label, _ in categories}

        for label, _ in categories:
            label_dir = samples_root / label
            if not label_dir.exists():
                self.stderr.write(f"Skipping '{label}' — directory missing at {label_dir}.")
                continue

            for sample_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                frames = list(self._load_frames(sample_dir))
                if not frames:
                    self.stderr.write(f"No frames found under {sample_dir} — skipping.")
                    continue

                score = is_live_face(
                    frames,
                    min_frames=min_frames,
                    motion_threshold=threshold,
                    return_score=True,
                )
                aggregate[label]["total"] += 1

                if score is None:
                    self.stderr.write(
                        f"Could not compute liveness score for {sample_dir} (insufficient motion)."
                    )
                    continue

                if label == "genuine":
                    if score >= threshold:
                        aggregate[label]["passes"] += 1
                else:
                    if score < threshold:
                        aggregate[label]["passes"] += 1

        if all(values["total"] == 0 for values in aggregate.values()):
            raise CommandError(
                "No evaluation samples were processed. Ensure the directory contains frame bursts."
            )

        self.stdout.write(self.style.SUCCESS("Liveness evaluation summary:"))
        for label, context in categories:
            total = aggregate[label]["total"]
            hits = aggregate[label]["passes"]
            rate = (hits / total * 100.0) if total else 0.0
            baseline = "100.0" if label == "genuine" else "0.0"
            self.stdout.write(
                f"- {label.title():<7}: {hits}/{total} {context} ({rate:.1f}% vs {baseline}% without liveness)"
            )

    def _load_frames(self, sample_dir: Path) -> Iterable[np.ndarray]:
        for image_path in sorted(sample_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            image = cv2.imread(str(image_path))
            if image is not None:
                yield image
