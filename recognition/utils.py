"""Utility functions for the recognition app."""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from django.conf import settings
from django.core.cache import cache

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from users.models import Time

import cv2
import numpy as np
from deepface import DeepFace

from src.common import InvalidToken, decrypt_bytes
from src.common.face_data_encryption import FaceDataEncryption

from .pipeline import extract_embedding

logger = logging.getLogger(__name__)

DATA_ROOT: Path = Path(settings.BASE_DIR) / "face_recognition_data"
TRAINING_DATASET_ROOT: Path = DATA_ROOT / "training_dataset"


class DatasetEmbeddingCache:
    """Cache DeepFace embeddings for the encrypted training dataset."""

    def __init__(
        self,
        dataset_root: Path,
        cache_root: Path,
        *,
        encryption: FaceDataEncryption | None = None,
    ) -> None:
        """Initialise the cache."""
        self._dataset_root = dataset_root
        self._cache_root = cache_root
        self._lock = threading.RLock()
        self._memory_cache: dict[
            tuple[str, str, bool], tuple[tuple[tuple[str, int, int], ...], list[dict[str, Any]]]
        ] = {}
        self._encryption = encryption or FaceDataEncryption()

    def _cache_file_path(
        self, model_name: str, detector_backend: str, enforce_detection: bool
    ) -> Path:
        """Generate a unique cache file path for the given parameters."""
        safe_model = model_name.replace("/", "_").replace(" ", "_")
        safe_detector = detector_backend.replace("/", "_").replace(" ", "_")
        enforcement_suffix = "enf" if enforce_detection else "noenf"
        filename = (
            f"representations_{safe_model.lower()}_{safe_detector.lower()}_"
            f"{enforcement_suffix}.pkl"
        )
        return self._cache_root / filename

    def _current_dataset_state(self) -> tuple[tuple[str, int, int], ...]:
        """Return a tuple representing the current state of the dataset."""
        entries: list[tuple[str, int, int]] = []
        if not self._dataset_root.exists():
            return tuple()

        for image_path in sorted(self._dataset_root.glob("*/*.jpg")):
            try:
                stat_result = image_path.stat()
            except FileNotFoundError:
                continue
            relative = image_path.relative_to(self._dataset_root).as_posix()
            entries.append((relative, int(stat_result.st_mtime_ns), int(stat_result.st_size)))
        return tuple(entries)

    def _load_from_disk(
        self, model_name: str, detector_backend: str, enforce_detection: bool
    ) -> tuple[tuple[tuple[str, int, int], ...], list[dict[str, Any]]] | None:
        """Load cached embeddings from a pickle file on disk."""
        cache_file = self._cache_file_path(model_name, detector_backend, enforce_detection)
        if not cache_file.exists():
            return None

        try:
            encrypted_bytes = cache_file.read_bytes()
            decrypted_bytes = self._encryption.decrypt(encrypted_bytes)
            payload = pickle.loads(decrypted_bytes)
        except (IOError, pickle.UnpicklingError, InvalidToken) as exc:
            logger.warning("Failed to load cached embeddings from %s: %s", cache_file, exc)
            return None

        dataset_state = payload.get("dataset_state")
        dataset_index = payload.get("dataset_index")
        if not isinstance(dataset_state, tuple) or not isinstance(dataset_index, list):
            return None

        return dataset_state, dataset_index

    def _save_to_disk(
        self,
        model_name: str,
        detector_backend: str,
        enforce_detection: bool,
        dataset_state: tuple[tuple[str, int, int], ...],
        dataset_index: list[dict[str, Any]],
    ) -> None:
        """Save embeddings to a pickle file on disk."""
        cache_file = self._cache_file_path(model_name, detector_backend, enforce_detection)
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "dataset_state": dataset_state,
                "dataset_index": dataset_index,
            }
            serialized = pickle.dumps(payload)
            encrypted = self._encryption.encrypt(serialized)
            cache_file.write_bytes(encrypted)
        except (IOError, pickle.PicklingError) as exc:
            logger.warning("Failed to save cached embeddings to %s: %s", cache_file, exc)

    def get_dataset_index(
        self,
        model_name: str,
        detector_backend: str,
        enforce_detection: bool,
        builder: Callable[[], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Return cached embeddings, building them if the dataset has changed."""
        key = (model_name, detector_backend, enforce_detection)
        current_state = self._current_dataset_state()

        with self._lock:
            memory_entry = self._memory_cache.get(key)
            if memory_entry and memory_entry[0] == current_state:
                return memory_entry[1]

            disk_entry = self._load_from_disk(model_name, detector_backend, enforce_detection)
            if disk_entry and disk_entry[0] == current_state:
                self._memory_cache[key] = disk_entry
                return disk_entry[1]

        dataset_index = builder()
        refreshed_state = self._current_dataset_state()

        with self._lock:
            entry = (refreshed_state, dataset_index)
            self._memory_cache[key] = entry
            self._save_to_disk(
                model_name,
                detector_backend,
                enforce_detection,
                refreshed_state,
                dataset_index,
            )

        return dataset_index

    def invalidate(self) -> None:
        """Clear the in-memory cache and remove cached files from disk."""
        with self._lock:
            self._memory_cache.clear()

        for cache_file in self._cache_root.glob("representations_*.pkl"):
            try:
                cache_file.unlink()
            except (IOError, FileNotFoundError) as exc:
                logger.warning("Failed to delete cache file %s: %s", cache_file, exc)


_dataset_embedding_cache = DatasetEmbeddingCache(TRAINING_DATASET_ROOT, DATA_ROOT)


def _get_deepface_options() -> dict[str, Any]:
    """Return the configured DeepFace options merged with defaults."""
    defaults: dict[str, Any] = {
        "model": "Facenet",
        "detector_backend": "ssd",
        "distance_metric": "euclidean_l2",
    }
    configured = getattr(settings, "DEEPFACE_OPTIMIZATIONS", {})
    return defaults | (configured if isinstance(configured, dict) else {})


def get_face_detection_backend() -> str:
    """Return the configured face detection backend for DeepFace."""
    return str(_get_deepface_options().get("detector_backend", "opencv"))


def get_face_recognition_model() -> str:
    """Return the face recognition model name configured in Django settings."""
    return str(_get_deepface_options().get("model", "VGG-Face"))


def should_enforce_detection() -> bool:
    """Return whether DeepFace should enforce detection failures."""
    return bool(_get_deepface_options().get("enforce_detection", False))


def get_deepface_distance_metric() -> str:
    """Return the configured embedding distance metric."""
    return str(_get_deepface_options().get("distance_metric", "euclidean_l2"))


def is_liveness_enabled() -> bool:
    """Return whether anti-spoofing checks are enabled."""
    return bool(_get_deepface_options().get("anti_spoofing", True))


def _decrypt_image_bytes(image_path: Path) -> bytes | None:
    """Return decrypted bytes for the encrypted image stored on disk."""
    try:
        encrypted_bytes = image_path.read_bytes()
        return decrypt_bytes(encrypted_bytes)
    except (IOError, InvalidToken) as exc:
        logger.error("Failed to decrypt image %s: %s", image_path, exc)
        return None


def _decode_image_bytes(decrypted_bytes: bytes, *, source: Path | None = None) -> np.ndarray | None:
    """Decode decrypted image bytes into a numpy array."""
    frame_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
    if frame_array.size == 0:
        logger.warning("Decrypted image %s is empty.", source or "payload")
        return None

    imread_flag = getattr(cv2, "IMREAD_COLOR", 1)
    image = cv2.imdecode(frame_array, imread_flag)
    if image is None:
        logger.warning("Failed to decode decrypted image %s", source or "payload")
        return None
    return image


def _get_or_compute_cached_embedding(
    image_path: Path, model_name: str, detector_backend: str, enforce_detection: bool = False
) -> np.ndarray | None:
    """Return the embedding for an encrypted image using Django's cache."""
    decrypted_bytes = _decrypt_image_bytes(image_path)
    if decrypted_bytes is None:
        return None

    payload_hash = hashlib.sha256(decrypted_bytes).hexdigest()
    cache_key = f"recognition:embedding:{model_name}:{detector_backend}:{payload_hash}"

    cached_embedding = cache.get(cache_key)
    if cached_embedding is not None:
        try:
            return np.array(cached_embedding, dtype=np.float64)
        except (TypeError, ValueError):
            logger.debug("Failed to coerce cached embedding for %s into an array", image_path)

    image = _decode_image_bytes(decrypted_bytes, source=image_path)
    if image is None:
        return None

    try:
        representations = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
        )
    except Exception as exc:
        logger.debug("Failed to generate embedding for %s: %s", image_path, exc)
        return None

    embedding_vector, _ = extract_embedding(representations)
    if embedding_vector is None:
        logger.debug("No embedding produced for %s", image_path)
        return None

    cache.set(cache_key, embedding_vector.tolist(), timeout=None)
    return embedding_vector


def _build_dataset_embeddings_for_matching(
    model_name: str, detector_backend: str, enforce_detection: bool = False
) -> list[dict[str, Any]]:
    """Build embeddings for the encrypted training dataset."""
    dataset_index = []
    image_paths = sorted(TRAINING_DATASET_ROOT.glob("*/*.jpg"))

    for image_path in image_paths:
        embedding_array = _get_or_compute_cached_embedding(
            image_path, model_name, detector_backend, enforce_detection
        )
        if embedding_array is None:
            continue

        dataset_index.append(
            {
                "identity": str(image_path),
                "embedding": embedding_array,
                "username": image_path.parent.name,
            }
        )
    return dataset_index


def load_dataset_embeddings_for_matching(
    model_name: str, detector_backend: str, enforce_detection: bool
) -> list[dict[str, Any]]:
    """Return embeddings from the cache, computing them if necessary."""
    return _dataset_embedding_cache.get_dataset_index(
        model_name,
        detector_backend,
        enforce_detection,
        lambda: _build_dataset_embeddings_for_matching(
            model_name, detector_backend, enforce_detection
        ),
    )


def check_validity_times(times_all: "QuerySet[Time]") -> tuple[bool, float]:
    """Validate and calculate break hours from a sequence of time entries."""
    if not times_all.exists():
        return True, 0.0

    first_entry = times_all.first()
    if first_entry is None or first_entry.time is None:
        return True, 0.0

    if first_entry.out:
        return False, 0.0

    if times_all.filter(out=False).count() != times_all.filter(out=True).count():
        return False, 0.0

    break_hours = 0.0
    prev_time = None
    is_break = False

    for entry in times_all:
        if not entry.out:
            if is_break and prev_time is not None and entry.time is not None:
                break_duration = (entry.time - prev_time).total_seconds() / 3600
                break_hours += break_duration
            is_break = False
        else:
            is_break = True
        prev_time = entry.time

    return True, break_hours


def update_attendance_in_db_in(
    present: dict[str, bool], *, attempt_ids: "Mapping[str, int]" | None = None
) -> None:
    """Persist check-in attendance information for the provided users."""
    from django.contrib.auth.models import User
    from django.utils import timezone

    from users.models import Present, RecognitionAttempt, Time

    today = timezone.localdate()
    current_time = timezone.now()
    attempt_ids = attempt_ids or {}
    for person, is_present in present.items():
        user = User.objects.filter(username=person).first()
        if not user:
            logger.warning("Skipping check-in for unknown user '%s'", person)
            continue

        qs, created = Present.objects.get_or_create(
            user=user, date=today, defaults={"present": is_present}
        )
        if not created and is_present and not qs.present:
            qs.present = True
            qs.save(update_fields=["present"])

        if is_present:
            time_record = Time.objects.create(user=user, date=today, time=current_time, out=False)
            if attempt_id := attempt_ids.get(person):
                RecognitionAttempt.objects.filter(id=attempt_id).update(
                    user=user, present_record=qs, time_record=time_record
                )


def update_attendance_in_db_out(
    present: dict[str, bool], *, attempt_ids: "Mapping[str, int]" | None = None
) -> None:
    """Persist check-out attendance information for the provided users."""
    from django.contrib.auth.models import User
    from django.utils import timezone

    from users.models import Present, RecognitionAttempt, Time

    today = timezone.localdate()
    current_time = timezone.now()
    attempt_ids = attempt_ids or {}
    for person, is_present in present.items():
        if not is_present:
            continue
        user = User.objects.filter(username=person).first()
        if not user:
            logger.warning("Skipping check-out for unknown user '%s'", person)
            continue

        time_record = Time.objects.create(user=user, date=today, time=current_time, out=True)
        if attempt_id := attempt_ids.get(person):
            present_record = Present.objects.filter(user=user, date=today).first()
            RecognitionAttempt.objects.filter(id=attempt_id).update(
                user=user, time_record=time_record, present_record=present_record
            )
