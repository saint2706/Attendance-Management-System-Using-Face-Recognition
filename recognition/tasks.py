"""Background jobs and helpers for incremental face training."""

from __future__ import annotations

import asyncio
import io
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from django.conf import settings

import cv2
import imutils
import numpy as np
from asgiref.sync import sync_to_async
from celery import shared_task
from imutils.video import VideoStream
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.common import (
    InvalidToken,
    decrypt_bytes,
    decrypt_face_bytes,
    encrypt_bytes,
    encrypt_face_bytes,
)

from . import embedding_cache
from .views import (
    DATA_ROOT,
    TRAINING_DATASET_ROOT,
    _dataset_embedding_cache,
    _get_face_detection_backend,
    _get_face_recognition_model,
    _get_or_compute_cached_embedding,
    _get_recognition_training_seed,
    _get_recognition_training_test_split_ratio,
    _is_headless_environment,
    _should_enforce_detection,
    update_attendance_in_db_in,
    update_attendance_in_db_out,
)

logger = logging.getLogger(__name__)

ENCODINGS_DIR = DATA_ROOT / "encodings"
MODEL_PATH = DATA_ROOT / "svc.sav"
CLASSES_PATH = DATA_ROOT / "classes.npy"
FAISS_INDEX_PATH = DATA_ROOT / "faiss_index.bin.enc"


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

    try:
        encrypted_bytes = encoding_path.read_bytes()
        decrypted_bytes = decrypt_face_bytes(encrypted_bytes)
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

    encrypted_payload = encrypt_face_bytes(buffer.getvalue())
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

    logger.info(
        "Replacing non-incremental classifier %s with SGDClassifier.",
        type(model).__name__,
    )
    return None


def _train_classifier(
    features: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> SGDClassifier:
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


class TrainingPreconditionError(RuntimeError):
    """Raised when the training workflow cannot proceed due to missing data."""


def capture_dataset_sync(
    username: str,
    *,
    max_frames: int | None = None,
    frame_pause: float | None = None,
    enqueue_training: bool = True,
) -> dict[str, Any]:
    """Capture encrypted frames for the provided ``username`` synchronously."""

    dataset_directory = TRAINING_DATASET_ROOT / username
    dataset_directory.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing video stream to capture images for %s", username)

    video_stream = VideoStream(src=0).start()

    headless = _is_headless_environment()
    max_frames = (
        int(getattr(settings, "RECOGNITION_HEADLESS_DATASET_FRAMES", 50))
        if max_frames is None
        else max_frames
    )
    frame_pause = (
        float(getattr(settings, "RECOGNITION_HEADLESS_FRAME_SLEEP", 0.01))
        if frame_pause is None
        else frame_pause
    )

    sample_number = 0
    saved_paths: list[Path] = []

    try:
        while True:
            frame = video_stream.read()
            if frame is None:
                continue

            frame = imutils.resize(frame, width=800)
            sample_number += 1

            output_path = dataset_directory / f"{sample_number}.jpg"
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                logger.warning(
                    "Failed to encode captured frame %s for %s",
                    sample_number,
                    username,
                )
                continue

            try:
                encrypted_frame = encrypt_bytes(buffer.tobytes())
            except Exception as exc:  # pragma: no cover - defensive programming
                logger.error(
                    "Failed to encrypt frame %s for %s: %s",
                    sample_number,
                    username,
                    exc,
                )
                continue

            try:
                with output_path.open("wb") as image_file:
                    image_file.write(encrypted_frame)
                saved_paths.append(output_path)
            except OSError as exc:
                logger.error(
                    "Failed to persist encrypted frame %s for %s: %s",
                    sample_number,
                    username,
                    exc,
                )
                continue

            if not headless:
                cv2.imshow("Add Images - Press 'q' to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if frame_pause:
                    time.sleep(frame_pause)

            if sample_number >= max_frames:
                break
    finally:
        logger.info("Finished capturing images for %s.", username)
        video_stream.stop()
        if not headless:
            cv2.destroyAllWindows()

    if enqueue_training and saved_paths:
        try:
            incremental_face_training.delay(username, [str(path) for path in saved_paths])
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.error("Failed to enqueue incremental training for %s: %s", username, exc)

    return {
        "username": username,
        "frames_captured": sample_number,
        "images_saved": len(saved_paths),
        "saved_paths": [str(path) for path in saved_paths],
    }


def train_model_sync(*, initiated_by: str | None = None) -> dict[str, Any]:
    """Run the full training workflow synchronously and return evaluation metrics."""

    logger.info("Training workflow started%s.", f" by {initiated_by}" if initiated_by else "")

    image_paths = sorted(TRAINING_DATASET_ROOT.glob("*/*.jpg"))
    if not image_paths:
        raise TrainingPreconditionError("No training data found. Add photos before training.")

    embedding_vectors: list[list[float]] = []
    class_names: list[str] = []

    model_name = _get_face_recognition_model()
    detector_backend = _get_face_detection_backend()

    for image_path in image_paths:
        embedding_array = _get_or_compute_cached_embedding(image_path, model_name, detector_backend)
        if embedding_array is None:
            logger.debug("Skipping image %s because no embedding was produced.", image_path)
            continue

        embedding_vectors.append(embedding_array.tolist())
        class_names.append(image_path.parent.name)

    if not embedding_vectors:
        raise TrainingPreconditionError(
            "No usable training data found after decrypting images. Please recreate the dataset."
        )

    unique_classes = sorted(set(class_names))
    if len(unique_classes) < 2:
        raise TrainingPreconditionError(
            "Training requires at least two different users with photos."
        )

    test_split_ratio = _get_recognition_training_test_split_ratio()
    random_seed = _get_recognition_training_seed()

    X_train, X_test, y_train, y_test = train_test_split(
        embedding_vectors,
        class_names,
        test_size=test_split_ratio,
        random_state=random_seed,
        stratify=class_names,
    )
    logger.info("Data split: %d training samples, %d test samples.", len(X_train), len(X_test))

    model = SVC(gamma="auto", probability=True, random_state=random_seed)
    model.fit(X_train, y_train)
    logger.info("SVC training complete.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
    logger.info("Model Evaluation Report:\n%s", report)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    model_path = DATA_ROOT / "svc.sav"
    model_bytes = pickle.dumps(model)
    model_path.write_bytes(encrypt_bytes(model_bytes))

    classes_path = DATA_ROOT / "classes.npy"
    buffer = io.BytesIO()
    np.save(buffer, unique_classes)
    classes_path.write_bytes(encrypt_bytes(buffer.getvalue()))

    report_path = DATA_ROOT / "classification_report.txt"
    with report_path.open("w") as handle:
        handle.write("Model Evaluation Report\n")
        handle.write("=========================\n\n")
        handle.write(f"Accuracy: {accuracy:.2%}\n")
        handle.write(f"Precision (weighted): {precision:.2f}\n")
        handle.write(f"Recall (weighted): {recall:.2f}\n")
        handle.write(f"F1-Score (weighted): {f1:.2f}\n\n")
        handle.write(str(report))

    logger.info("Successfully saved model, classes, and evaluation report.")

    # Build and save FAISS index for optimised search
    try:
        from .faiss_index import FAISSIndex

        embedding_array = np.array(embedding_vectors, dtype=np.float32)
        faiss_index = FAISSIndex(dimension=embedding_array.shape[1])
        faiss_index.add_embeddings(embedding_array, class_names)
        faiss_index.save(FAISS_INDEX_PATH)
        logger.info(
            "FAISS index built with %d embeddings (%d classes).",
            faiss_index.size,
            len(unique_classes),
        )
    except Exception as exc:
        logger.warning("Failed to build FAISS index (non-fatal): %s", exc)

    # Invalidate embedding caches after training
    _dataset_embedding_cache.invalidate()
    embedding_cache.invalidate_all_embeddings()
    logger.info("Embedding caches invalidated after training.")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "unique_classes": unique_classes,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


@shared_task(bind=True, name="recognition.capture_dataset")
def capture_dataset(self, username: str) -> dict[str, Any]:
    """Celery task wrapper for :func:`capture_dataset_sync`."""

    self.update_state(state="STARTED", meta={"username": username, "frames_captured": 0})
    try:
        result = capture_dataset_sync(username)
    except Exception:
        logger.exception("Dataset capture failed for %s", username)
        raise

    return result


@shared_task(bind=True, name="recognition.train_recognition_model")
def train_recognition_model(self, initiated_by: str | None = None) -> dict[str, Any]:
    """Celery task that executes the synchronous training workflow."""

    self.update_state(state="STARTED", meta={"initiated_by": initiated_by})
    try:
        result = train_model_sync(initiated_by=initiated_by)
    except TrainingPreconditionError:
        logger.warning("Training aborted due to missing prerequisites.", exc_info=True)
        raise
    except Exception:
        logger.exception("Training workflow failed.")
        raise

    return result


@shared_task(bind=True, name="recognition.incremental_face_training")
def incremental_face_training(self, employee_id: str, new_images: Sequence[str]) -> dict[str, Any]:
    """Incrementally update stored encodings and classifier for the employee."""

    if not new_images:
        logger.debug("No new images supplied for %s; skipping incremental training.", employee_id)
        return {
            "employee_id": employee_id,
            "images_provided": 0,
            "status": "skipped",
        }

    logger.info(
        "Starting incremental training for %s with %d images.",
        employee_id,
        len(new_images),
    )

    new_vectors: list[np.ndarray] = []
    for image in new_images:
        embedding = compute_face_encoding(Path(image))
        if embedding is None:
            logger.debug("No embedding generated for %s; skipping.", image)
            continue
        new_vectors.append(embedding.astype(np.float64))

    if not new_vectors:
        logger.info(
            "No embeddings produced for %s; skipping persistence and training.",
            employee_id,
        )
        return {
            "employee_id": employee_id,
            "images_provided": len(new_images),
            "status": "no-embeddings",
        }

    existing = load_existing_encodings(employee_id)
    if existing.size:
        combined = np.vstack([np.atleast_2d(existing), np.vstack(new_vectors)])
    else:
        combined = np.vstack(new_vectors)

    save_employee_encodings(employee_id, combined)

    self.update_state(
        state="PROGRESS",
        meta={
            "employee_id": employee_id,
            "images_provided": len(new_images),
            "encodings_total": int(combined.shape[0]),
        },
    )

    features: list[list[float]] = []
    labels: list[str] = []

    for known_employee, encodings in _iter_all_employee_encodings():
        for vector in encodings:
            features.append(vector.astype(float).tolist())
            labels.append(known_employee)

    if not features:
        logger.warning("No encodings available after update; skipping classifier training.")
        _dataset_embedding_cache.invalidate()
        return {
            "employee_id": employee_id,
            "images_provided": len(new_images),
            "encodings_total": int(combined.shape[0]),
            "status": "no-encodings",
        }

    classes = np.array(sorted(set(labels)))
    if classes.size < 2:
        logger.info(
            "Insufficient distinct classes (%d) for training; skipping classifier update.",
            classes.size,
        )
        _dataset_embedding_cache.invalidate()
        return {
            "employee_id": employee_id,
            "images_provided": len(new_images),
            "encodings_total": int(combined.shape[0]),
            "status": "insufficient-classes",
        }

    try:
        classifier = _train_classifier(
            np.array(features, dtype=np.float64), np.array(labels), classes
        )
        _persist_model(classifier, classes)

        # Rebuild FAISS index with updated embeddings
        from .faiss_index import FAISSIndex

        embedding_array = np.array(features, dtype=np.float32)
        faiss_index = FAISSIndex(dimension=embedding_array.shape[1])
        faiss_index.add_embeddings(embedding_array, labels)
        faiss_index.save(FAISS_INDEX_PATH)
        logger.debug("FAISS index updated with %d embeddings.", faiss_index.size)

    except Exception as exc:  # pragma: no cover - defensive programming
        logger.error("Incremental training failed for %s: %s", employee_id, exc)
        _dataset_embedding_cache.invalidate()
        raise

    logger.info("Incremental training for %s complete.", employee_id)
    _dataset_embedding_cache.invalidate()
    embedding_cache.invalidate_user_embeddings(employee_id)
    embedding_cache.invalidate_all_embeddings()  # Clear dataset index cache

    return {
        "employee_id": employee_id,
        "images_provided": len(new_images),
        "encodings_total": int(combined.shape[0]),
        "status": "trained",
        "classes": sorted(set(labels)),
    }


async def process_single_attendance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Process a single attendance record asynchronously."""

    direction = str(record.get("direction", "in")).lower()
    payload = record.get("present") or record.get("payload") or {}
    attempt_ids = record.get("attempt_ids")
    if not isinstance(attempt_ids, Mapping):
        attempt_ids = {}

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
        await update_async(present_payload, attempt_ids=attempt_ids)
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.exception("Failed to process %s attendance payload: %s", direction, exc)
        result["status"] = "error"
        result["error"] = str(exc)

    return result


@shared_task(bind=True)
def process_attendance_batch(
    self, records: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
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
    "TrainingPreconditionError",
    "capture_dataset",
    "capture_dataset_sync",
    "incremental_face_training",
    "train_model_sync",
    "train_recognition_model",
    "compute_face_encoding",
    "load_existing_encodings",
    "save_employee_encodings",
    "process_single_attendance",
    "process_attendance_batch",
]
