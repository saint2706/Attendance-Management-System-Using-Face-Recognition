#!/usr/bin/env python3
"""Reproduce face-recognition metrics using the bundled synthetic dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths() -> tuple[Path, Path, Path]:
    base = _project_root()
    dataset_root = base / "sample_data" / "face_recognition_data" / "training_dataset"
    split_csv = base / "sample_data" / "reports" / "sample_splits.csv"
    reports_dir = base / "reports" / "sample_repro"
    return dataset_root, split_csv, reports_dir


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    dataset_root, split_csv, reports_dir = _default_paths()
    parser = argparse.ArgumentParser(
        description="Run the evaluation pipeline against the synthetic sample dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=dataset_root,
        help="Root directory that contains per-user folders with JPEG frames.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=split_csv,
        help="Optional CSV describing the evaluation split (defaults to sample_splits.csv).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=reports_dir,
        help="Directory where the evaluation artifacts will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed forwarded to the evaluation helpers.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional distance threshold override.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of evaluation samples (handy for smoke tests).",
    )
    return parser.parse_args(argv)


def _ensure_django_ready() -> None:
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")
    import django

    django.setup()


def _patch_dataset_root(dataset_root: Path) -> None:
    from recognition import views as recognition_views

    recognition_views.DATA_ROOT = dataset_root.parent
    recognition_views.TRAINING_DATASET_ROOT = dataset_root
    recognition_views._dataset_embedding_cache = recognition_views.DatasetEmbeddingCache(
        recognition_views.TRAINING_DATASET_ROOT,
        recognition_views.DATA_ROOT,
    )


def _validate_dataset(dataset_root: Path) -> int:
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root {dataset_root} does not exist. Run from the repository root or "
            "provide --dataset-root."
        )

    jpeg_count = len(list(dataset_root.glob("*/*.jpg")))
    if jpeg_count == 0:
        raise RuntimeError(
            f"Dataset root {dataset_root} does not contain any .jpg files. "
            "Verify that sample_data is intact."
        )
    return jpeg_count


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    split_csv = args.split_csv.resolve()
    reports_dir = args.reports_dir.resolve()

    try:
        sample_count = _validate_dataset(dataset_root)
    except Exception as exc:  # pragma: no cover - CLI helper
        print(f"[reproduce] {exc}", file=sys.stderr)
        return 1

    if split_csv.exists():
        split_path: Optional[Path] = split_csv
    else:
        split_path = None
        print(
            f"[reproduce] Split CSV {split_csv} not found. Falling back to scanning the dataset root.",
            file=sys.stderr,
        )

    reports_dir.mkdir(parents=True, exist_ok=True)

    _ensure_django_ready()
    _patch_dataset_root(dataset_root)

    from src.common.seeding import set_global_seed
    from src.evaluation.face_recognition_eval import (
        EvaluationConfig,
        run_face_recognition_evaluation,
    )

    set_global_seed(args.seed)

    print("=== Attendance sample reproducibility run ===")
    print(f"Dataset root : {dataset_root}")
    print(f"Split CSV    : {split_path or 'scan entire dataset'}")
    print(f"Reports dir  : {reports_dir}")
    print(f"Samples      : {sample_count}")

    config = EvaluationConfig(
        reports_dir=reports_dir,
        test_split_csv=split_path,
        dataset_root=dataset_root,
        threshold=args.threshold,
        limit_samples=args.max_samples,
    )

    summary = run_face_recognition_evaluation(config)

    print("\nMetrics (macro averages unless noted):")
    for key in ("accuracy", "precision", "recall", "f1", "far", "frr"):
        value = summary.metrics.get(key)
        if value is None:
            continue
        print(f"  - {key.title():<9}: {value:.4f}")

    print("\nArtifacts:")
    for label, path in summary.artifact_paths.items():
        print(f"  - {label:>12}: {path}")

    print("\nDone. Inspect the artifacts above for the deterministic outputs.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
