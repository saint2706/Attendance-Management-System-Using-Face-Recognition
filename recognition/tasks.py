"""Background jobs and helpers for incremental face training."""

from __future__ import annotations

import asyncio
import io
import logging
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from asgiref.sync import sync_to_async
from celery import shared_task
from django_rq import job
from sklearn.linear_model import SGDClassifier

from src.common import decrypt_bytes, encrypt_bytes
from src.common.face_data_encryption import FaceDataEncryption, InvalidToken

from .views import (
    DATA_ROOT,
    _dataset_embedding_cache,
    _get_face_detection_backend,
    _get_face_recognition_model,
    _get_or_compute_cached_embedding,
    _should_enforce_detection,
    update_attendance_in_db_in,
    update_attendance_in_db_out,
)

logger = logging.getLogger(__name__)

ENCODINGS_DIR = DATA_ROOT / "encodings"
MODEL_PATH = DATA_ROOT / "svc.sav"
CLASSES_PATH = DATA_ROOT / "classes.npy"


def _employee_encoding_path(employee_id: str) -> Path:
    """Return the path where the employee's encodings are stored."""

    return ENCODINGS_DIR / employee_id / "encodings.npy.enc"


def _ensure_directory(path: Path) -> None:
    """Create parent directories for the provided path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def load_existing_encodings(employee_id: str) -> np.ndarray:
    """Return the persisted encodings for the provided employee."""

    encoding_path = _employee_encoding_path(employee_id)
    if not encoding_path.exists():
        return np.empty((0, 0), dtype=np.float64)

    helper = FaceDataEncryption()
    try:
        encrypted_bytes = encoding_path.read_bytes()
        decrypted_bytes = helper.decrypt(encrypted_bytes)
    except FileNotFoundError:
        return np.empty((0, 0), dtype=np.float64)
    except InvalidToken:
        logger.warning("Failed to decrypt encodings for %s due to invalid token.", employee_id)
        return np.empty((0, 0), dtype=np.float64)
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.error("Unexpected error loading encodings for %s: %s", employee_id, exc)
        return np.empty((0, 0), dtype=np.float64)

    try:
        return np.load(io.BytesIO(decrypted_bytes))
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.warning("Failed to deserialize encodings for %s: %s", employee_id, exc)
        return np.empty((0, 0), dtype=np.float64)


def compute_face_encoding(image_path: Path) -> np.ndarray | None:
    """Compute or retrieve the DeepFace embedding for an encrypted image."""

    model_name = _get_face_recognition_model()
    detector_backend = _get_face_detection_backend()
    enforce_detection = _should_enforce_detection()

    embedding = _get_or_compute_cached_embedding(
        Path(image_path), model_name, detector_backend, enforce_detection
    )
    if embedding is None:
        return None
    return np.array(embedding, dtype=np.float64)


def save_employee_encodings(employee_id: str, encodings: Iterable[Sequence[float]]) -> None:
    """Persist the provided encodings for the employee."""

    encoding_array = np.atleast_2d(np.asarray(list(encodings), dtype=np.float64))
    encoding_path = _employee_encoding_path(employee_id)
    _ensure_directory(encoding_path)

    buffer = io.BytesIO()
    np.save(buffer, encoding_array)

    helper = FaceDataEncryption()
    encrypted_payload = helper.encrypt(buffer.getvalue())
    encoding_path.write_bytes(encrypted_payload)


def _iter_all_employee_encodings() -> Iterable[tuple[str, np.ndarray]]:
    """Yield all stored employee encodings."""

    if not ENCODINGS_DIR.exists():
        return

    for employee_dir in sorted(path for path in ENCODINGS_DIR.iterdir() if path.is_dir()):
        employee_id = employee_dir.name
        encodings = load_existing_encodings(employee_id)
        if encodings.size == 0:
            continue
        yield employee_id, np.atleast_2d(encodings).astype(np.float64)


def _load_existing_model() -> SGDClassifier | None:
    """Return the previously trained classifier if available."""

    if not MODEL_PATH.exists():
        return None

    try:
        encrypted_model = MODEL_PATH.read_bytes()
        decrypted_model = decrypt_bytes(encrypted_model)
        model = pickle.loads(decrypted_model)
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.warning("Failed to load existing model: %s", exc)
        return None

    if isinstance(model, SGDClassifier):
        return model

    logger.info("Replacing non-incremental classifier %s with SGDClassifier.", type(model).__name__)
    return None


def _train_classifier(features: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> SGDClassifier:
    """Train or update the incremental classifier on the provided dataset."""

    existing_model = _load_existing_model()
    if existing_model is None:
        classifier = SGDClassifier(loss="log_loss", random_state=42)
    else:
        classifier = existing_model

    if hasattr(classifier, "partial_fit"):
        classifier.partial_fit(features, labels, classes=classes)
    else:  # pragma: no cover - defensive programming
        classifier.fit(features, labels)

    return classifier


def _persist_model(classifier: SGDClassifier, classes: np.ndarray) -> None:
    """Persist the trained classifier and class labels to disk."""

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    model_bytes = pickle.dumps(classifier)
    MODEL_PATH.write_bytes(encrypt_bytes(model_bytes))

    buffer = io.BytesIO()
    np.save(buffer, classes)
    CLASSES_PATH.write_bytes(encrypt_bytes(buffer.getvalue()))


@job("default")
def incremental_face_training(employee_id: str, new_images: Sequence[str]) -> None:
    """Incrementally update stored encodings and classifier for the employee."""

    if not new_images:
        logger.debug("No new images supplied for %s; skipping incremental training.", employee_id)
        return

    logger.info("Starting incremental training for %s with %d images.", employee_id, len(new_images))

    new_vectors: list[np.ndarray] = []
    for image in new_images:
        embedding = compute_face_encoding(Path(image))
        if embedding is None:
            logger.debug("No embedding generated for %s; skipping.", image)
            continue
        new_vectors.append(embedding.astype(np.float64))

    if not new_vectors:
        logger.info("No embeddings produced for %s; skipping persistence and training.", employee_id)
        return

    existing = load_existing_encodings(employee_id)
    if existing.size:
        combined = np.vstack([np.atleast_2d(existing), np.vstack(new_vectors)])
    else:
        combined = np.vstack(new_vectors)

    save_employee_encodings(employee_id, combined)

    features: list[list[float]] = []
    labels: list[str] = []

    for known_employee, encodings in _iter_all_employee_encodings():
        for vector in encodings:
            features.append(vector.astype(float).tolist())
            labels.append(known_employee)

    if not features:
        logger.warning("No encodings available after update; skipping classifier training.")
        _dataset_embedding_cache.invalidate()
        return

    classes = np.array(sorted(set(labels)))
    if classes.size < 2:
        logger.info(
            "Insufficient distinct classes (%d) for training; skipping classifier update.",
            classes.size,
        )
        _dataset_embedding_cache.invalidate()
        return

    try:
        classifier = _train_classifier(
            np.array(features, dtype=np.float64), np.array(labels), classes
        )
        _persist_model(classifier, classes)
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.error("Incremental training failed for %s: %s", employee_id, exc)
    else:
        logger.info("Incremental training for %s complete.", employee_id)
    finally:
        _dataset_embedding_cache.invalidate()


async def process_single_attendance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Process a single attendance record asynchronously."""

    direction = str(record.get("direction", "in")).lower()
    payload = record.get("present") or record.get("payload") or {}

    result: dict[str, Any] = {
        "direction": direction,
        "processed": 0,
        "status": "success",
    }

    if direction not in {"in", "out"}:
        result["status"] = "error"
        result["error"] = "direction must be either 'in' or 'out'"
        return result

    if not isinstance(payload, Mapping):
        result["status"] = "error"
        result["error"] = "present payload must be a mapping"
        return result

    present_payload = dict(payload)
    result["processed"] = len(present_payload)

    if not present_payload:
        return result

    update_fn = update_attendance_in_db_in if direction == "in" else update_attendance_in_db_out
    update_async = sync_to_async(update_fn, thread_sensitive=True)

    try:
        await update_async(present_payload)
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.exception("Failed to process %s attendance payload: %s", direction, exc)
        result["status"] = "error"
        result["error"] = str(exc)

    return result


@shared_task(bind=True)
def process_attendance_batch(self, records: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Process a batch of attendance records using asyncio within a Celery task."""

    normalized_records = list(records or [])
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        if normalized_records:
            raw_results = loop.run_until_complete(
                asyncio.gather(
                    *(process_single_attendance(record) for record in normalized_records),
                    return_exceptions=True,
                )
            )
        else:
            raw_results = []

        results: list[dict[str, Any]] = []
        for index, outcome in enumerate(raw_results):
            if isinstance(outcome, Exception):
                logger.exception(
                    "Attendance batch entry %d raised an exception.",
                    index,
                    exc_info=outcome,
                )
                record_direction = str(
                    normalized_records[index].get("direction", "unknown")
                ).lower()
                results.append(
                    {
                        "direction": record_direction,
                        "processed": 0,
                        "status": "error",
                        "error": str(outcome),
                    }
                )
            else:
                results.append(outcome)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:  # pragma: no cover - defensive programming
            logger.debug("Async generator shutdown encountered an issue.")
        asyncio.set_event_loop(None)
        loop.close()

    return {
        "results": results,
        "total": len(normalized_records),
    }


__all__ = [
    "incremental_face_training",
    "compute_face_encoding",
    "load_existing_encodings",
    "save_employee_encodings",
    "process_single_attendance",
    "process_attendance_batch",
]

