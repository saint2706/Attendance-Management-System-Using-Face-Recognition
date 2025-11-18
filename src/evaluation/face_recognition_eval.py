"""End-to-end evaluation harness for the face-recognition pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from django.conf import settings
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from recognition.pipeline import find_closest_dataset_match, is_within_distance_threshold

from . import UNKNOWN_LABEL

logger = logging.getLogger(__name__)


@dataclass
class SampleEvaluation:
    """Container describing the prediction for a single evaluation sample."""

    image_path: Path
    ground_truth: str
    match_username: Optional[str]
    distance: Optional[float]
    embedding_available: bool

    def predicted_label(self, threshold: float) -> str:
        """Return the predicted label after applying the provided threshold."""

        if (
            self.match_username
            and self.distance is not None
            and is_within_distance_threshold(self.distance, threshold)
        ):
            return self.match_username
        return UNKNOWN_LABEL

    def to_row(self, threshold: float) -> Dict[str, object]:
        """Return a serialisable representation for CSV exports."""

        return {
            "image_path": str(self.image_path),
            "ground_truth": self.ground_truth,
            "match_username": self.match_username or "",
            "distance": None if self.distance is None else float(self.distance),
            "embedding_available": self.embedding_available,
            "predicted_label": self.predicted_label(threshold),
        }


def _recognition_views():  # pragma: no cover - import helper
    from recognition import views as recognition_views

    return recognition_views


@dataclass
class EvaluationConfig:
    """Runtime configuration for the evaluation pipeline."""

    reports_dir: Optional[Path] = None
    test_split_csv: Optional[Path] = None
    dataset_root: Optional[Path] = None
    threshold: Optional[float] = None
    threshold_start: float = 0.2
    threshold_stop: float = 1.0
    threshold_step: float = 0.05
    limit_samples: Optional[int] = None

    def __post_init__(self) -> None:
        base_dir = Path(getattr(settings, "BASE_DIR", Path.cwd()))
        if self.reports_dir is None:
            self.reports_dir = base_dir / "reports" / "evaluation"
        else:
            self.reports_dir = Path(self.reports_dir)

        views_module = _recognition_views()

        if self.dataset_root is None:
            self.dataset_root = Path(views_module.TRAINING_DATASET_ROOT)
        else:
            self.dataset_root = Path(self.dataset_root)

        if self.test_split_csv is not None:
            self.test_split_csv = Path(self.test_split_csv)

        if self.threshold is None:
            default_threshold = getattr(
                settings,
                "RECOGNITION_DISTANCE_THRESHOLD",
                views_module.DEFAULT_DISTANCE_THRESHOLD,
            )
            self.threshold = float(default_threshold)
        else:
            self.threshold = float(self.threshold)

        if self.threshold_step <= 0:
            raise ValueError("threshold_step must be positive")

        if self.threshold_stop < self.threshold_start:
            raise ValueError("threshold_stop must be >= threshold_start")

        if self.limit_samples is not None and self.limit_samples <= 0:
            raise ValueError("limit_samples must be positive when provided")

    @property
    def threshold_values(self) -> List[float]:
        """Return the inclusive list of threshold values for sweep analysis."""

        values: List[float] = []
        current = self.threshold_start
        while current <= self.threshold_stop + 1e-9:
            values.append(round(float(current), 6))
            current += self.threshold_step
        return values


@dataclass
class EvaluationSummary:
    """Aggregate outputs from the evaluation run."""

    metrics: Dict[str, float]
    threshold_sweep: List[Dict[str, float]]
    samples: List[SampleEvaluation]
    y_true: List[str]
    y_pred: List[str]
    labels: List[str]
    artifact_paths: Dict[str, Path]


def _resolve_image_paths(config: EvaluationConfig) -> List[Path]:
    """Return ordered image paths for evaluation based on the provided config."""

    dataset_root = config.dataset_root
    assert dataset_root is not None  # for mypy
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist")

    image_paths: List[Path] = []
    allowed_suffixes = {".jpg", ".jpeg", ".png"}

    if config.test_split_csv and config.test_split_csv.exists():
        logger.info("Loading evaluation split from %s", config.test_split_csv)
        with config.test_split_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if str(row.get("split", "")).strip().lower() != "test":
                    continue
                raw_path = row.get("image_path")
                if not raw_path:
                    continue
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = dataset_root / candidate
                if candidate.suffix.lower() not in allowed_suffixes:
                    continue
                if candidate.exists():
                    image_paths.append(candidate)
                else:
                    logger.warning("Split entry %s does not exist on disk", candidate)

    if not image_paths:
        logger.info("Falling back to scanning dataset root %s", dataset_root)
        for suffix in allowed_suffixes:
            image_paths.extend(dataset_root.glob(f"*/*{suffix}"))

    image_paths = sorted(set(image_paths))

    if config.limit_samples is not None:
        image_paths = image_paths[: config.limit_samples]

    if not image_paths:
        raise RuntimeError("No evaluation images found. Ensure the dataset exists or split CSV is populated.")

    logger.info("Prepared %s evaluation samples", len(image_paths))
    return image_paths


def _load_dataset_index(views_module, model_name: str, detector_backend: str, enforce_detection: bool):
    """Return the cached dataset index used during recognition."""

    return views_module._load_dataset_embeddings_for_matching(  # type: ignore[attr-defined]
        model_name,
        detector_backend,
        enforce_detection,
    )


def _infer_samples(
    image_paths: Sequence[Path],
    dataset_index,
    *,
    views_module,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    distance_metric: str,
) -> List[SampleEvaluation]:
    """Run recognition for each evaluation image and return per-sample metadata."""

    results: List[SampleEvaluation] = []
    for path in image_paths:
        ground_truth = path.parent.name or UNKNOWN_LABEL
        embedding = views_module._get_or_compute_cached_embedding(  # type: ignore[attr-defined]
            path,
            model_name,
            detector_backend,
            enforce_detection,
        )

        embedding_available = embedding is not None
        match_username: Optional[str] = None
        distance: Optional[float] = None

        if embedding is not None:
            match = find_closest_dataset_match(embedding, dataset_index, distance_metric)
            if match is not None:
                match_username = match[0] or None
                distance = float(match[1]) if match[1] is not None else None

        results.append(
            SampleEvaluation(
                image_path=path,
                ground_truth=ground_truth or UNKNOWN_LABEL,
                match_username=match_username,
                distance=distance,
                embedding_available=embedding_available,
            )
        )

    return results


def _sorted_labels(y_true: Sequence[str], y_pred: Sequence[str]) -> List[str]:
    """Return deterministic label ordering with UNKNOWN at the end."""

    label_set = {label for label in y_true if label != UNKNOWN_LABEL}
    label_set.update(label for label in y_pred if label != UNKNOWN_LABEL)
    labels = sorted(label_set)
    if any(label == UNKNOWN_LABEL for label in [*y_true, *y_pred]):
        labels.append(UNKNOWN_LABEL)
    return labels


def _calculate_far_frr(y_true: Sequence[str], y_pred: Sequence[str]) -> Tuple[float, float]:
    """Calculate the False Acceptance and False Rejection rates."""

    total_genuine = sum(1 for truth in y_true if truth != UNKNOWN_LABEL)
    false_rejects = sum(
        1 for truth, pred in zip(y_true, y_pred) if truth != UNKNOWN_LABEL and truth != pred
    )
    frr = false_rejects / total_genuine if total_genuine else 0.0

    total_impostor = sum(1 for truth in y_true if truth == UNKNOWN_LABEL)
    false_accepts = sum(
        1 for truth, pred in zip(y_true, y_pred) if truth == UNKNOWN_LABEL and pred != UNKNOWN_LABEL
    )
    far = false_accepts / total_impostor if total_impostor else 0.0

    return float(far), float(frr)


def compute_basic_metrics(
    samples: Sequence[SampleEvaluation], threshold: float
) -> Tuple[Dict[str, float], List[str], List[str], List[str]]:
    """Compute core classification metrics for the provided samples."""

    if not samples:
        raise ValueError("No samples provided for metric computation")

    y_true = [sample.ground_truth or UNKNOWN_LABEL for sample in samples]
    y_pred = [sample.predicted_label(threshold) for sample in samples]
    labels = _sorted_labels(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    far, frr = _calculate_far_frr(y_true, y_pred)
    unknown_predictions = sum(1 for label in y_pred if label == UNKNOWN_LABEL)
    missing_embeddings = sum(1 for sample in samples if not sample.embedding_available)

    metrics = {
        "threshold": float(threshold),
        "samples": len(samples),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "far": float(far),
        "frr": float(frr),
        "unknown_predictions": int(unknown_predictions),
        "samples_without_embedding": int(missing_embeddings),
    }

    return metrics, y_true, y_pred, labels


def compute_threshold_sweep(
    samples: Sequence[SampleEvaluation], thresholds: Iterable[float]
) -> List[Dict[str, float]]:
    """Compute metrics for each threshold in ``thresholds``."""

    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        metrics, _, _, _ = compute_basic_metrics(samples, threshold)
        rows.append(metrics)
    return rows


def _save_metrics_json(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"generated_at": datetime.utcnow().isoformat(), "metrics": metrics}, handle, indent=2)


def _save_samples_csv(samples: Sequence[SampleEvaluation], threshold: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [sample.to_row(threshold) for sample in samples]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_confusion_outputs(
    cm: np.ndarray,
    labels: Sequence[str],
    csv_path: Path,
    png_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + list(labels))
        for idx, row in enumerate(cm.tolist()):
            writer.writerow([labels[idx]] + row)

    plt.figure(figsize=(max(6, len(labels)), max(6, len(labels))))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path)
    plt.close()


def _save_threshold_sweep_outputs(rows: Sequence[Dict[str, float]], csv_path: Path, png_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    thresholds = [row["threshold"] for row in rows]
    far_values = [row["far"] for row in rows]
    frr_values = [row["frr"] for row in rows]
    f1_values = [row["f1"] for row in rows]
    accuracy_values = [row["accuracy"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, far_values, label="FAR", marker="o")
    plt.plot(thresholds, frr_values, label="FRR", marker="o")
    plt.plot(thresholds, f1_values, label="F1", linestyle="--")
    plt.plot(thresholds, accuracy_values, label="Accuracy", linestyle=":")
    plt.xlabel("Distance threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold sweep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path)
    plt.close()


def _save_artifacts(
    *,
    config: EvaluationConfig,
    metrics: Dict[str, float],
    samples: Sequence[SampleEvaluation],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    threshold_sweep: Sequence[Dict[str, float]],
) -> Dict[str, Path]:
    """Persist artifacts and return their locations."""

    reports_dir = config.reports_dir
    assert reports_dir is not None
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics_summary.json"
    samples_path = reports_dir / "sample_predictions.csv"
    confusion_csv = reports_dir / "confusion_matrix.csv"
    confusion_png = reports_dir / "confusion_matrix.png"
    sweep_csv = reports_dir / "threshold_sweep.csv"
    sweep_png = reports_dir / "threshold_sweep.png"

    _save_metrics_json(metrics, metrics_path)
    _save_samples_csv(samples, metrics["threshold"], samples_path)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    _save_confusion_outputs(cm, labels, confusion_csv, confusion_png)
    _save_threshold_sweep_outputs(threshold_sweep, sweep_csv, sweep_png)

    return {
        "metrics": metrics_path,
        "samples": samples_path,
        "confusion_csv": confusion_csv,
        "confusion_png": confusion_png,
        "threshold_csv": sweep_csv,
        "threshold_png": sweep_png,
    }


def run_face_recognition_evaluation(config: EvaluationConfig) -> EvaluationSummary:
    """Execute the evaluation end-to-end and return the resulting summary."""

    views_module = _recognition_views()
    model_name = views_module._get_face_recognition_model()  # type: ignore[attr-defined]
    detector_backend = views_module._get_face_detection_backend()  # type: ignore[attr-defined]
    enforce_detection = views_module._should_enforce_detection()  # type: ignore[attr-defined]
    distance_metric = views_module._get_deepface_distance_metric()  # type: ignore[attr-defined]

    logger.info(
        "Running evaluation with model=%s detector=%s metric=%s threshold=%.3f",
        model_name,
        detector_backend,
        distance_metric,
        config.threshold,
    )

    image_paths = _resolve_image_paths(config)
    dataset_index = _load_dataset_index(views_module, model_name, detector_backend, enforce_detection)
    samples = _infer_samples(
        image_paths,
        dataset_index,
        views_module=views_module,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        distance_metric=distance_metric,
    )

    assert config.threshold is not None
    metrics, y_true, y_pred, labels = compute_basic_metrics(samples, config.threshold)
    threshold_sweep = compute_threshold_sweep(samples, config.threshold_values)
    artifact_paths = _save_artifacts(
        config=config,
        metrics=metrics,
        samples=samples,
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        threshold_sweep=threshold_sweep,
    )

    return EvaluationSummary(
        metrics=metrics,
        threshold_sweep=threshold_sweep,
        samples=samples,
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        artifact_paths=artifact_paths,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Return an argument parser for CLI execution."""

    parser = argparse.ArgumentParser(description="Evaluate the face-recognition pipeline")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Optional CSV generated by prepare_splits.py containing the test split.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the dataset root. Defaults to the encrypted training dataset.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory where evaluation artifacts will be stored.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Distance threshold used for the main metric computation.",
    )
    parser.add_argument("--threshold-start", type=float, default=0.2)
    parser.add_argument("--threshold-stop", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of evaluation samples (useful for smoke tests).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> EvaluationSummary:
    """CLI entry-point that configures Django and executes the evaluation."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    # Lazy import to avoid Django setup during module import time
    import django

    django.setup()

    config = EvaluationConfig(
        reports_dir=args.reports_dir,
        test_split_csv=args.split_csv,
        dataset_root=args.dataset_root,
        threshold=args.threshold,
        threshold_start=args.threshold_start,
        threshold_stop=args.threshold_stop,
        threshold_step=args.threshold_step,
        limit_samples=args.max_samples,
    )

    summary = run_face_recognition_evaluation(config)
    logger.info("Evaluation metrics saved to %s", summary.artifact_paths.get("metrics"))
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
