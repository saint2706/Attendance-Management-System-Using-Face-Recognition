"""
Views for the recognition app.

This module contains the view functions for the face recognition-based attendance system.
It handles requests for rendering pages, processing forms, capturing images,
marking attendance, and displaying attendance data.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import logging
import math
import os
import pickle
import sys
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from urllib.parse import urljoin

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.cache import cache
from django.db.models import Count, QuerySet
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone

import cv2
import imutils
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deepface import DeepFace
from django_pandas.io import read_frame
from django_ratelimit.core import is_ratelimited
from imutils.video import VideoStream
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.common import InvalidToken, decrypt_bytes, encrypt_bytes
from src.common.face_data_encryption import FaceDataEncryption
from users.models import Present, Time

from .forms import DateForm, DateForm_2, UsernameAndDateForm, usernameForm
from .webcam_manager import get_webcam_manager

# Use 'Agg' backend for Matplotlib to avoid GUI-related issues in a web server environment
mpl.use("Agg")

# Initialize logger for the module
logger = logging.getLogger(__name__)


# Rate limit helper utilities -------------------------------------------------


def _attendance_rate_limit_methods() -> tuple[str, ...]:
    """Return the normalized HTTP methods subject to attendance rate limiting."""

    methods = getattr(settings, "RECOGNITION_ATTENDANCE_RATE_LIMIT_METHODS", ("POST",))
    if isinstance(methods, str):
        methods = (methods,)
    return tuple(method.upper() for method in methods)


def attendance_rate_limited(view_func):
    """Apply django-ratelimit protection to attendance endpoints."""

    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        rate = getattr(settings, "RECOGNITION_ATTENDANCE_RATE_LIMIT", "5/m")
        if not rate:
            return view_func(request, *args, **kwargs)

        methods = _attendance_rate_limit_methods()
        request_method = request.method.upper()
        if methods and request_method not in methods:
            return view_func(request, *args, **kwargs)

        was_limited = is_ratelimited(
            request=request,
            group="recognition.attendance",
            key="user_or_ip",
            rate=rate,
            method=methods or None,
            increment=True,
        )

        if was_limited:
            logger.warning(
                "Attendance rate limit triggered for %s via %s",
                (
                    request.user
                    if getattr(request, "user", None) and request.user.is_authenticated
                    else request.META.get("REMOTE_ADDR", "unknown")
                ),
                request_method,
            )
            return HttpResponse("Too many attendance attempts. Please wait.", status=429)

        return view_func(request, *args, **kwargs)

    return _wrapped


# Define root directories for data storage
DATA_ROOT = Path(settings.BASE_DIR) / "face_recognition_data"
TRAINING_DATASET_ROOT = DATA_ROOT / "training_dataset"

# Define paths for saving generated attendance graphs within MEDIA_ROOT
ATTENDANCE_GRAPHS_ROOT = Path(
    getattr(settings, "ATTENDANCE_GRAPHS_ROOT", Path(settings.MEDIA_ROOT) / "attendance_graphs")
)
HOURS_VS_DATE_PATH = ATTENDANCE_GRAPHS_ROOT / "hours_vs_date" / "1.png"
EMPLOYEE_LOGIN_PATH = ATTENDANCE_GRAPHS_ROOT / "employee_login" / "1.png"
HOURS_VS_EMPLOYEE_PATH = ATTENDANCE_GRAPHS_ROOT / "hours_vs_employee" / "1.png"
THIS_WEEK_PATH = ATTENDANCE_GRAPHS_ROOT / "this_week" / "1.png"
LAST_WEEK_PATH = ATTENDANCE_GRAPHS_ROOT / "last_week" / "1.png"


class DatasetEmbeddingCache:
    """Cache DeepFace embeddings for the encrypted training dataset."""

    def __init__(
        self,
        dataset_root: Path,
        cache_root: Path,
        *,
        encryption: FaceDataEncryption | None = None,
    ) -> None:
        self._dataset_root = dataset_root
        self._cache_root = cache_root
        self._lock = threading.RLock()
        self._memory_cache: Dict[
            Tuple[str, str, bool], Tuple[Tuple[Tuple[str, int, int], ...], list[dict]]
        ]
        self._memory_cache = {}
        self._encryption = encryption or FaceDataEncryption()

    def _cache_file_path(
        self, model_name: str, detector_backend: str, enforce_detection: bool
    ) -> Path:
        safe_model = model_name.replace("/", "_").replace(" ", "_")
        safe_detector = detector_backend.replace("/", "_").replace(" ", "_")
        enforcement_suffix = "enf" if enforce_detection else "noenf"
        filename = (
            f"representations_{safe_model.lower()}_{safe_detector.lower()}_"
            f"{enforcement_suffix}.pkl"
        )
        return self._cache_root / filename

    def _current_dataset_state(self) -> Tuple[Tuple[str, int, int], ...]:
        entries: list[Tuple[str, int, int]] = []
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
    ) -> Optional[Tuple[Tuple[Tuple[str, int, int], ...], list[dict]]]:
        cache_file = self._cache_file_path(model_name, detector_backend, enforce_detection)
        if not cache_file.exists():
            return None

        try:
            encrypted_bytes = cache_file.read_bytes()
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.warning("Failed to read cached embeddings %s: %s", cache_file, exc)
            return None

        try:
            decrypted_bytes = self._encryption.decrypt(encrypted_bytes)
        except InvalidToken:
            logger.warning("Invalid cache encryption token for %s", cache_file)
            return None
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.warning("Failed to decrypt cached embeddings %s: %s", cache_file, exc)
            return None

        try:
            payload = pickle.loads(decrypted_bytes)
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.warning("Failed to deserialize cached embeddings %s: %s", cache_file, exc)
            return None

        dataset_state = payload.get("dataset_state")
        dataset_index = payload.get("dataset_index")
        if not isinstance(dataset_state, tuple) or not isinstance(dataset_index, list):
            return None

        normalized_index: list[dict] = []
        for entry in dataset_index:
            if not isinstance(entry, dict):
                continue
            normalized = dict(entry)
            embedding = normalized.get("embedding")
            if embedding is not None and not isinstance(embedding, np.ndarray):
                try:
                    normalized["embedding"] = np.array(embedding, dtype=float)
                except Exception:  # pragma: no cover - defensive programming
                    logger.debug("Failed to coerce cached embedding to ndarray: %r", embedding)
                    continue
            normalized_index.append(normalized)

        if len(normalized_index) != len(dataset_index):
            return None

        return dataset_state, normalized_index

    def _save_to_disk(
        self,
        model_name: str,
        detector_backend: str,
        enforce_detection: bool,
        dataset_state: Tuple[Tuple[str, int, int], ...],
        dataset_index: list[dict],
    ) -> None:
        cache_file = self._cache_file_path(model_name, detector_backend, enforce_detection)
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload_index: list[dict] = []
            for entry in dataset_index:
                normalized = dict(entry)
                embedding = normalized.get("embedding")
                if isinstance(embedding, np.ndarray):
                    normalized["embedding"] = embedding.astype(float).tolist()
                elif isinstance(embedding, (list, tuple)):
                    normalized["embedding"] = [float(value) for value in embedding]
                payload_index.append(normalized)

            payload = {
                "dataset_state": dataset_state,
                "dataset_index": payload_index,
            }
            serialized = pickle.dumps(payload)
            encrypted = self._encryption.encrypt(serialized)
            cache_file.write_bytes(encrypted)
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.warning("Failed to persist cached embeddings %s: %s", cache_file, exc)

    def get_dataset_index(
        self,
        model_name: str,
        detector_backend: str,
        enforce_detection: bool,
        builder: Callable[[], list[dict]],
    ) -> list[dict]:
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
            except FileNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover - defensive programming
                logger.warning("Failed to delete cache file %s: %s", cache_file, exc)


_dataset_embedding_cache = DatasetEmbeddingCache(TRAINING_DATASET_ROOT, DATA_ROOT)


def _ensure_directory(path: Path) -> None:
    """
    Ensure the parent directory for the given path exists.

    If the directory does not exist, it is created. This is useful for
    ensuring that file paths for saving graphs are valid.

    Args:
        path: The file path whose parent directory needs to exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _media_url_for(path: Path) -> str:
    """Return the MEDIA_URL-relative URL for the provided file path."""

    media_root = Path(settings.MEDIA_ROOT)
    try:
        relative_path = path.resolve().relative_to(media_root.resolve())
    except ValueError:
        relative_path = path
    return urljoin(settings.MEDIA_URL, relative_path.as_posix())


def _save_plot_to_media(path: Path) -> str:
    """Persist the current Matplotlib plot and return its media URL."""

    _ensure_directory(path)
    try:
        plt.savefig(path)
    finally:
        plt.close()
    return _media_url_for(path)


def _decrypt_image_bytes(image_path: Path) -> Optional[bytes]:
    """Return decrypted bytes for the encrypted image stored on disk."""
    try:
        encrypted_bytes = image_path.read_bytes()
    except FileNotFoundError:
        logger.warning("Image path %s is missing while building dataset index.", image_path)
        return None
    except OSError as exc:
        logger.error("Failed to read encrypted image %s: %s", image_path, exc)
        return None

    try:
        decrypted_bytes = decrypt_bytes(encrypted_bytes)
    except InvalidToken:
        logger.error("Invalid encryption token encountered for %s", image_path)
        return None
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.error("Unexpected error decrypting %s: %s", image_path, exc)
        return None

    return decrypted_bytes


def _decode_image_bytes(
    decrypted_bytes: bytes, *, source: Optional[Path] = None
) -> Optional[np.ndarray]:
    """Decode decrypted image bytes into a numpy array."""

    frame_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
    if frame_array.size == 0:
        if source is not None:
            logger.warning("Decrypted image %s is empty.", source)
        else:
            logger.warning("Encountered empty decrypted image payload.")
        return None

    imread_flag = getattr(cv2, "IMREAD_COLOR", 1)
    image = cv2.imdecode(frame_array, imread_flag)
    if image is None:
        if source is not None:
            logger.warning("Failed to decode decrypted image %s", source)
        else:
            logger.warning("Failed to decode decrypted image payload.")
        return None
    return image


def _load_encrypted_image(image_path: Path) -> Optional[np.ndarray]:
    """Decrypt and load an encrypted image stored on disk."""

    decrypted_bytes = _decrypt_image_bytes(image_path)
    if decrypted_bytes is None:
        return None

    return _decode_image_bytes(decrypted_bytes, source=image_path)


def _extract_first_embedding(
    representations,
) -> Tuple[Optional[Sequence[float]], Optional[Dict[str, int]]]:
    """Normalize DeepFace representations to a single embedding and facial area."""

    embedding_vector: Optional[Sequence[float]] = None
    facial_area: Optional[Dict[str, int]] = None

    if isinstance(representations, np.ndarray):
        if representations.ndim == 2 and len(representations) > 0:
            embedding_vector = representations[0]
    elif isinstance(representations, list) and representations:
        first = representations[0]
        if isinstance(first, dict):
            embedding_vector = first.get("embedding")
            area = first.get("facial_area")
            facial_area = area if isinstance(area, dict) else None
        elif isinstance(first, (list, tuple, np.ndarray)):
            embedding_vector = first
    elif isinstance(representations, dict) and "embedding" in representations:
        embedding_vector = representations.get("embedding")
        area = representations.get("facial_area")
        facial_area = area if isinstance(area, dict) else None

    if embedding_vector is None:
        return None, facial_area

    try:
        normalized = [float(value) for value in embedding_vector]
    except (TypeError, ValueError):
        logger.debug("Unable to coerce embedding values to floats: %r", embedding_vector)
        return None, facial_area

    return normalized, facial_area


def _get_or_compute_cached_embedding(
    image_path: Path, model_name: str, detector_backend: str, enforce_detection: bool = False
) -> Optional[np.ndarray]:
    """Return the embedding for an encrypted image using Django's cache."""

    decrypted_bytes = _decrypt_image_bytes(image_path)
    if decrypted_bytes is None:
        return None

    payload_hash = hashlib.sha256(decrypted_bytes).hexdigest()
    cache_key = f"recognition:embedding:{model_name}:{detector_backend}:{payload_hash}"

    cached_embedding = cache.get(cache_key)
    if cached_embedding is not None:
        try:
            return np.array(cached_embedding, dtype=float)
        except Exception:  # pragma: no cover - defensive programming
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

    embedding_vector, _ = _extract_first_embedding(representations)
    if embedding_vector is None:
        logger.debug("No embedding produced for %s", image_path)
        return None

    embedding_array = np.array(embedding_vector, dtype=float)

    try:
        cache.set(cache_key, embedding_array.tolist(), timeout=None)
    except Exception:  # pragma: no cover - defensive programming
        logger.debug("Failed to store embedding for %s in cache", image_path)

    return embedding_array


def _build_dataset_embeddings_for_matching(
    model_name: str, detector_backend: str, enforce_detection: bool = False
):
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


def _load_dataset_embeddings_for_matching(
    model_name: str, detector_backend: str, enforce_detection: bool
):
    """Return embeddings from the cache, computing them if necessary."""

    return _dataset_embedding_cache.get_dataset_index(
        model_name,
        detector_backend,
        enforce_detection,
        lambda: _build_dataset_embeddings_for_matching(
            model_name, detector_backend, enforce_detection
        ),
    )


def _calculate_embedding_distance(
    candidate: np.ndarray, embedding_vector: np.ndarray, metric: str
) -> Optional[float]:
    """Compute a distance score between embeddings for the provided metric."""

    metric = metric.lower()
    try:
        if metric in {"cosine", "cosine_similarity"}:
            candidate_norm = float(np.linalg.norm(candidate))
            vector_norm = float(np.linalg.norm(embedding_vector))
            if candidate_norm == 0.0 or vector_norm == 0.0:
                return None
            similarity = float(np.dot(candidate, embedding_vector) / (candidate_norm * vector_norm))
            return 1.0 - similarity

        if metric in {"euclidean", "euclidean_l2", "l2"}:
            return float(np.linalg.norm(candidate - embedding_vector))

        if metric in {"manhattan", "l1", "euclidean_l1"}:
            return float(np.sum(np.abs(candidate - embedding_vector)))

    except Exception as exc:  # pragma: no cover - defensive programming
        logger.debug("Failed to compute %s distance: %s", metric, exc)
        return None

    # Default to Euclidean L2 if metric is unrecognized
    try:
        return float(np.linalg.norm(candidate - embedding_vector))
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.debug("Failed to compute fallback distance: %s", exc)
        return None


def _find_closest_dataset_match(
    embedding_vector: np.ndarray, dataset_index, metric: str
) -> Optional[Tuple[str, float, str]]:
    """Return the nearest neighbour match for the provided embedding."""

    if embedding_vector.size == 0 or not dataset_index:
        return None

    best_entry = None
    best_distance: Optional[float] = None

    for entry in dataset_index:
        candidate = entry.get("embedding")
        if candidate is None:
            continue

        distance = _calculate_embedding_distance(candidate, embedding_vector, metric)
        if distance is None:
            continue

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_entry = entry

    if best_entry is None or best_distance is None:
        return None

    return best_entry["username"], best_distance, best_entry["identity"]


def username_present(username: str) -> bool:
    """
    Return whether the given username exists in the system.

    Args:
        username: The username to check.

    Returns:
        True if the user exists, False otherwise.
    """
    return User.objects.filter(username=username).exists()


def create_dataset(username: str) -> None:
    """
    Capture and store face images for the provided username.

    This function initializes a video stream from the webcam, captures 50 frames,
    and saves them as JPEG images in a directory named after the user.

    Args:
        username: The username for whom the dataset is being created.
    """
    dataset_directory = TRAINING_DATASET_ROOT / username
    dataset_directory.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing video stream to capture images for %s", username)
    video_stream = VideoStream(src=0).start()

    headless = _is_headless_environment()
    max_frames = int(getattr(settings, "RECOGNITION_HEADLESS_DATASET_FRAMES", 50))
    frame_pause = float(getattr(settings, "RECOGNITION_HEADLESS_FRAME_SLEEP", 0.01))

    sample_number = 0
    try:
        while True:
            # Read a frame from the video stream
            frame = video_stream.read()
            if frame is None:
                continue
            frame = imutils.resize(frame, width=800)

            sample_number += 1
            output_path = dataset_directory / f"{sample_number}.jpg"
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                logger.warning("Failed to encode captured frame %s for %s", sample_number, username)
                continue

            try:
                encrypted_frame = encrypt_bytes(buffer.tobytes())
            except Exception as exc:  # pragma: no cover - defensive programming
                logger.error("Failed to encrypt frame %s for %s: %s", sample_number, username, exc)
                continue

            try:
                with output_path.open("wb") as image_file:
                    image_file.write(encrypted_frame)
            except OSError as exc:
                logger.error(
                    "Failed to persist encrypted frame %s for %s: %s", sample_number, username, exc
                )
                continue

            if not headless:
                # Display the frame to the user and allow them to quit manually
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
        _dataset_embedding_cache.invalidate()


def update_attendance_in_db_in(present: Dict[str, bool]) -> None:
    """
    Persist check-in attendance information for the provided users.

    This function records the time-in for recognized users. It creates a `Present`
    record if one doesn't exist for the day, and adds a `Time` entry for the check-in.

    Args:
        present: A dictionary mapping usernames to their presence status (True).
    """
    today = timezone.localdate()
    current_time = timezone.now()
    for person, is_present in present.items():
        user = User.objects.filter(username=person).first()
        if user is None:
            logger.warning(
                "Skipping check-in attendance update for unknown user '%s'. "
                "Training data may be stale.",
                person,
            )
            continue

        # Find or create the attendance record for the day
        qs, created = Present.objects.get_or_create(
            user=user, date=today, defaults={"present": is_present}
        )

        if not created and is_present and not qs.present:
            # Mark as present if they were previously marked absent
            qs.present = True
            qs.save(update_fields=["present"])

        if is_present:
            # Record the check-in time
            Time.objects.create(user=user, date=today, time=current_time, out=False)


def update_attendance_in_db_out(present: Dict[str, bool]) -> None:
    """
    Persist check-out attendance information for the provided users.

    This function records the time-out for recognized users by creating a `Time`
    entry with the `out` flag set to True.

    Args:
        present: A dictionary mapping usernames to their presence status (True).
    """
    today = timezone.localdate()
    current_time = timezone.now()
    for person, is_present in present.items():
        if not is_present:
            continue

        user = User.objects.filter(username=person).first()
        if user is None:
            logger.warning(
                "Skipping check-out attendance update for unknown user '%s'. "
                "Training data may be stale.",
                person,
            )
            continue
        # Record the check-out time
        Time.objects.create(user=user, date=today, time=current_time, out=True)


def check_validity_times(times_all: QuerySet[Time]) -> tuple[bool, float]:
    """
    Validate and calculate break hours from a sequence of time entries.

    This function checks if the time entries follow a valid in-out sequence and
    calculates the total break time in hours.

    Args:
        times_all: A queryset of `Time` objects for a user on a given day,
                   ordered by time.

    Returns:
        A tuple containing:
        - A boolean indicating if the time sequence is valid.
        - The total break hours as a float.
    """
    if not times_all:
        return True, 0  # No records, so no invalidity

    first_entry = times_all.first()
    if first_entry is None or first_entry.time is None:
        return True, 0

    # The first entry must be a check-in
    if first_entry.out:
        return False, 0

    # The number of check-ins must equal the number of check-outs
    if times_all.filter(out=False).count() != times_all.filter(out=True).count():
        return False, 0

    break_hours = 0
    prev_time = None
    is_break = False

    for entry in times_all:
        if not entry.out:  # This is a check-in
            if is_break and prev_time is not None and entry.time is not None:
                # Calculate time since the last check-out
                break_duration = (entry.time - prev_time).total_seconds() / 3600
                break_hours += break_duration
            is_break = False
        else:  # This is a check-out
            is_break = True
        prev_time = entry.time

    return True, break_hours


def convert_hours_to_hours_mins(hours: float) -> str:
    """
    Convert a float representing hours into a 'h hrs m mins' format.

    Args:
        hours: The total hours as a float.

    Returns:
        A formatted string (e.g., "8 hrs 30 mins").
    """
    h = int(hours)
    minutes = (hours - h) * 60
    m = math.ceil(minutes)
    return f"{h} hrs {m} mins"


def hours_vs_date_given_employee(
    present_qs: QuerySet[Present], time_qs: QuerySet[Time], admin: bool = True
) -> Tuple[QuerySet[Present], str]:
    """
    Calculate work and break hours for an employee over a date range and generate a plot.

    Args:
        present_qs: A queryset of `Present` objects for the employee.
        time_qs: A queryset of `Time` objects for the employee.
        admin: A boolean indicating if the view is for an admin (affects save path).

    Returns:
        A tuple containing the annotated queryset and the media URL of the generated plot.
    """
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []

    for obj in present_qs:
        date = obj.date
        times_all = time_qs.filter(date=date).order_by("time")
        times_in = times_all.filter(out=False)
        times_out = times_all.filter(out=True)

        # Use intermediate variables to avoid calling .time on None
        first_in = times_in.first()
        last_out = times_out.last()
        obj.time_in = first_in.time if first_in else None
        obj.time_out = last_out.time if last_out else None

        hours_val = 0.0
        if obj.time_in and obj.time_out:
            hours_val = (obj.time_out - obj.time_in).total_seconds() / 3600

        is_valid, break_hours_val = check_validity_times(times_all)
        if not is_valid:
            break_hours_val = 0.0

        df_hours.append(hours_val)
        df_break_hours.append(break_hours_val)

        # Format for display
        obj.hours = convert_hours_to_hours_mins(hours_val)
        obj.break_hours = convert_hours_to_hours_mins(break_hours_val)

    # Generate and save the plot
    df = read_frame(present_qs, fieldnames=["date"])
    df["hours"] = df_hours
    df["break_hours"] = df_break_hours
    logger.debug("Attendance dataframe for employee: %s", df)

    sns.barplot(data=df, x="date", y="hours")
    plt.xticks(rotation="vertical")
    rcParams.update({"figure.autolayout": True})
    plt.tight_layout()

    target_path = HOURS_VS_DATE_PATH if admin else EMPLOYEE_LOGIN_PATH
    chart_url = _save_plot_to_media(target_path)

    return present_qs, chart_url


def hours_vs_employee_given_date(
    present_qs: QuerySet[Present], time_qs: QuerySet[Time]
) -> Tuple[QuerySet[Present], str]:
    """
    Calculate work and break hours for all employees on a given date and generate a plot.

    Args:
        present_qs: A queryset of `Present` objects for the date.
        time_qs: A queryset of `Time` objects for the date.

    Returns:
        A tuple containing the annotated queryset and the media URL of the generated plot.
    """
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    df_username = []

    for obj in present_qs:
        user = obj.user
        times_all = time_qs.filter(user=user).order_by("time")
        times_in = times_all.filter(out=False)
        times_out = times_all.filter(out=True)

        # Use intermediate variables to avoid calling .time on None
        first_in = times_in.first()
        last_out = times_out.last()
        obj.time_in = first_in.time if first_in else None
        obj.time_out = last_out.time if last_out else None

        hours_val = 0.0
        if obj.time_in and obj.time_out:
            hours_val = (obj.time_out - obj.time_in).total_seconds() / 3600

        is_valid, break_hours_val = check_validity_times(times_all)
        if not is_valid:
            break_hours_val = 0.0

        df_hours.append(hours_val)
        df_username.append(user.username)
        df_break_hours.append(break_hours_val)

        # Format for display
        obj.hours = convert_hours_to_hours_mins(hours_val)
        obj.break_hours = convert_hours_to_hours_mins(break_hours_val)

    # Generate and save the plot
    df = read_frame(present_qs, fieldnames=["user"])
    df["hours"] = df_hours
    df["username"] = df_username
    df["break_hours"] = df_break_hours

    sns.barplot(data=df, x="username", y="hours")
    plt.xticks(rotation="vertical")
    rcParams.update({"figure.autolayout": True})
    plt.tight_layout()
    chart_url = _save_plot_to_media(HOURS_VS_EMPLOYEE_PATH)

    return present_qs, chart_url


def total_number_employees() -> int:
    """
    Return the total count of non-staff, non-superuser employees.
    """
    return User.objects.filter(is_staff=False, is_superuser=False).count()


def employees_present_today() -> int:
    """
    Return the count of employees marked as present today.
    """
    today = timezone.localdate()
    return Present.objects.filter(date=today, present=True).count()


def this_week_emp_count_vs_date() -> Optional[str]:
    """
    Generate and save a line plot of employee presence for the current week.

    Returns:
        The media URL of the generated plot, or ``None`` when no data is available.
    """
    today = timezone.localdate()
    start_of_week = today - datetime.timedelta(days=today.weekday())

    # Get attendance data for the current week
    qs = (
        Present.objects.filter(date__gte=start_of_week, date__lte=today, present=True)
        .values("date")
        .annotate(emp_count=Count("user"))
        .order_by("date")
    )

    # Create a dictionary for quick lookup
    attendance_by_date = {item["date"]: item["emp_count"] for item in qs}

    # Prepare data for the plot (Monday to Friday)
    str_dates_all = []
    emp_cnt_all = []
    for i in range(5):
        current_date = start_of_week + datetime.timedelta(days=i)
        if current_date > today:
            break
        str_dates_all.append(current_date.strftime("%Y-%m-%d"))
        emp_cnt_all.append(attendance_by_date.get(current_date, 0))

    if not str_dates_all:
        return None  # Avoid plotting if there's no data

    df = pd.DataFrame({"date": str_dates_all, "Number of employees": emp_cnt_all})

    sns.lineplot(data=df, x="date", y="Number of employees")
    return _save_plot_to_media(THIS_WEEK_PATH)


def last_week_emp_count_vs_date() -> Optional[str]:
    """
    Generate and save a line plot of employee presence for the last week.

    Returns:
        The media URL of the generated plot, or ``None`` when no data is available.
    """
    today = timezone.localdate()
    start_of_last_week = today - datetime.timedelta(days=today.weekday() + 7)
    end_of_last_week = start_of_last_week + datetime.timedelta(days=4)

    # Get attendance data for the last week
    qs = (
        Present.objects.filter(
            date__gte=start_of_last_week, date__lte=end_of_last_week, present=True
        )
        .values("date")
        .annotate(emp_count=Count("user"))
        .order_by("date")
    )

    attendance_by_date = {item["date"]: item["emp_count"] for item in qs}

    # Prepare data for the plot (Monday to Friday of last week)
    str_dates_all = []
    emp_cnt_all = []
    for i in range(5):
        current_date = start_of_last_week + datetime.timedelta(days=i)
        str_dates_all.append(current_date.strftime("%Y-%m-%d"))
        emp_cnt_all.append(attendance_by_date.get(current_date, 0))

    df = pd.DataFrame({"date": str_dates_all, "emp_count": emp_cnt_all})

    sns.lineplot(data=df, x="date", y="emp_count")
    return _save_plot_to_media(LAST_WEEK_PATH)


# ========== Main Views ==========


def home(request):
    """
    Render the home page.
    """
    return render(request, "recognition/home.html")


@login_required
def dashboard(request):
    """
    Render the dashboard, which differs for admins and regular employees.
    """
    if request.user.is_staff or request.user.is_superuser:
        logger.debug("Rendering admin dashboard for %s", request.user)
        return render(request, "recognition/admin_dashboard.html")

    logger.debug("Rendering employee dashboard for %s", request.user)
    return render(request, "recognition/employee_dashboard.html")


@login_required
def add_photos(request):
    """
    Handle the 'Add Photos' functionality for admins to create face datasets for users.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    if request.method == "POST":
        form = usernameForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            if username_present(username):
                create_dataset(username)
                messages.success(request, f"Dataset Created for {username}")
                return redirect("add-photos")

            messages.warning(request, "No such username found. Please register employee first.")
            return redirect("dashboard")
    else:
        form = usernameForm()

    return render(request, "recognition/add_photos.html", {"form": form})


# Default distance threshold for face recognition confidence
DEFAULT_DISTANCE_THRESHOLD = 0.4


LIVENESS_FAILURE_MESSAGE = (
    "Spoofing attempt detected. Please ensure a live person is present before marking "
    "attendance."
)


def _normalize_face_region(region: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
    """Normalize various facial area representations into ``x, y, w, h`` form."""

    if not isinstance(region, dict):
        return None

    keys = {key.lower(): value for key, value in region.items() if value is not None}

    def _get_value(*aliases: str) -> Optional[int]:
        for alias in aliases:
            if alias in keys:
                try:
                    return int(keys[alias])
                except (TypeError, ValueError):
                    return None
        return None

    x = _get_value("x", "left")
    y = _get_value("y", "top")
    w = _get_value("w", "width")
    h = _get_value("h", "height")

    if None not in (x, y, w, h):
        return {"x": x, "y": y, "w": w, "h": h}

    right = _get_value("right")
    bottom = _get_value("bottom")
    if None not in (x, right, y, bottom):
        width = right - x
        height = bottom - y
        if width > 0 and height > 0:
            return {"x": x, "y": y, "w": width, "h": height}

    return None


def _crop_face_region(frame: np.ndarray, region: Optional[Dict[str, int]]) -> np.ndarray:
    """Return a cropped face image given a bounding box region."""

    if frame is None or not hasattr(frame, "shape"):
        return frame

    normalized = _normalize_face_region(region)
    if not normalized:
        return frame

    height, width = frame.shape[:2]
    x = max(normalized.get("x", 0), 0)
    y = max(normalized.get("y", 0), 0)
    w = max(normalized.get("w", 0), 0)
    h = max(normalized.get("h", 0), 0)

    if w <= 0 or h <= 0:
        return frame

    x2 = min(x + w, width)
    y2 = min(y + h, height)
    if x >= x2 or y >= y2:
        return frame

    return frame[y:y2, x:x2]


def _interpret_deepface_liveness_result(result: object) -> Optional[bool]:
    """Best-effort interpretation of DeepFace anti-spoofing output."""

    if isinstance(result, (list, tuple)):
        if not result:
            return None
        return _interpret_deepface_liveness_result(result[0])

    if not isinstance(result, dict):
        return None

    bool_keys = (
        "is_real",
        "is_real_face",
        "is_live",
        "is_genuine",
        "real",
        "live",
        "genuine",
    )
    for key in bool_keys:
        value = result.get(key)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)

    score_keys = ("real", "live", "genuine")
    threshold = float(getattr(settings, "RECOGNITION_LIVENESS_SCORE_THRESHOLD", 0.5))
    for key in score_keys:
        value = result.get(key)
        if isinstance(value, (int, float)):
            return value >= threshold

    face_type = result.get("face_type")
    if isinstance(face_type, str):
        return face_type.lower() in {"real", "live", "genuine"}
    if isinstance(face_type, dict):
        label = face_type.get("type") or face_type.get("value")
        if isinstance(label, str):
            return label.lower() in {"real", "live", "genuine"}

    return None


def _deepface_liveness_check(frame: np.ndarray) -> Optional[bool]:
    """Run DeepFace anti-spoofing on the provided frame."""

    try:
        analysis = DeepFace.analyze(  # type: ignore[call-arg]
            img_path=frame,
            actions=("anti-spoof",),
            enforce_detection=False,
            detector_backend=_get_face_detection_backend(),
            silent=True,
        )
    except Exception as exc:
        logger.warning("Liveness check failed: %s", exc)
        return None

    decision = _interpret_deepface_liveness_result(analysis)
    if decision is None:
        logger.warning("Could not interpret DeepFace anti-spoofing result: %s", analysis)
    return decision


def _passes_liveness_check(
    frame: np.ndarray,
    face_region: Optional[Dict[str, int]] = None,
) -> bool:
    """Return ``True`` when the supplied frame passes the liveness gate."""

    if not _is_liveness_enabled():
        return True

    custom_checker = getattr(settings, "RECOGNITION_LIVENESS_CHECKER", None)
    if callable(custom_checker):
        try:
            decision = custom_checker(frame=frame, face_region=face_region)
            if decision is not None:
                return bool(decision)
        except Exception as exc:
            logger.warning("Custom liveness checker raised an exception: %s", exc)

    cropped = _crop_face_region(frame, face_region)
    decision = _deepface_liveness_check(cropped)
    return bool(decision)


def _evaluate_recognition_match(
    frame: np.ndarray,
    match: pd.Series,
    distance_threshold: float,
) -> Tuple[Optional[str], bool, Optional[Dict[str, int]]]:
    """Evaluate a DeepFace match result and run the liveness gate."""

    try:
        distance = float(match.get("distance", 0.0))
    except (TypeError, ValueError):
        distance = 0.0

    if distance > distance_threshold:
        return None, False, None

    identity_value = match.get("identity")
    username: Optional[str] = None
    if identity_value:
        try:
            identity_path = Path(str(identity_value))
            parent_name = identity_path.parent.name
            if parent_name:
                username = parent_name
        except Exception as exc:
            logger.debug("Could not parse identity path '%s': %s", identity_value, exc)

    face_region = {
        "x": match.get("source_x"),
        "y": match.get("source_y"),
        "w": match.get("source_w"),
        "h": match.get("source_h"),
    }
    normalized_region = _normalize_face_region(face_region)

    if not _passes_liveness_check(frame, normalized_region):
        return username, True, normalized_region

    return username, False, normalized_region


def _predict_identity_from_embedding(
    frame: np.ndarray,
    embedding_vector: Sequence[float],
    facial_area: Optional[Dict[str, int]],
    model,
    class_names: Sequence[str],
    attendance_type: str,
) -> Tuple[Optional[str], bool, Optional[Dict[str, int]]]:
    """Run liveness verification and model prediction for an embedding."""

    normalized_region = _normalize_face_region(facial_area)
    if not _passes_liveness_check(frame, normalized_region):
        logger.warning("Spoofing attempt blocked during '%s' attendance.", attendance_type)
        return None, True, normalized_region

    try:
        prediction = model.predict(np.array([embedding_vector]))
    except Exception as exc:
        logger.warning("Failed to classify embedding: %s", exc)
        return None, False, normalized_region

    if isinstance(prediction, (list, tuple, np.ndarray)) and len(prediction) > 0:
        predicted_index = int(prediction[0])
    else:
        try:
            predicted_index = int(prediction)
        except (TypeError, ValueError):
            logger.warning("Received unexpected prediction output: %s", prediction)
            return None, False, normalized_region

    if 0 <= predicted_index < len(class_names):
        predicted_name = str(class_names[predicted_index])
        return predicted_name, False, normalized_region

    logger.warning("Predicted index %s out of range", predicted_index)
    return None, False, normalized_region


def _is_headless_environment() -> bool:
    """Return ``True`` when no graphical display is available."""

    override = getattr(settings, "RECOGNITION_HEADLESS", None)
    if override is not None:
        return bool(override)

    # Windows typically has a display available when running interactively
    if sys.platform.startswith("win"):
        return False

    display_vars = ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")
    if any(os.environ.get(var) for var in display_vars):
        return False

    return True


def _mark_attendance(request, check_in: bool):
    """
    Core logic for marking attendance (both in and out) using face recognition.

    Args:
        request: The Django HttpRequest object.
        check_in: True for marking time-in, False for time-out.
    """
    manager = get_webcam_manager()
    present = {}
    headless = _is_headless_environment()
    max_frames = int(getattr(settings, "RECOGNITION_HEADLESS_ATTENDANCE_FRAMES", 30))
    frame_pause = float(getattr(settings, "RECOGNITION_HEADLESS_FRAME_SLEEP", 0.01))
    frames_processed = 0

    # Configure DeepFace settings
    model_name = _get_face_recognition_model()
    detector_backend = _get_face_detection_backend()
    enforce_detection = _should_enforce_detection()
    distance_threshold = getattr(
        settings, "RECOGNITION_DISTANCE_THRESHOLD", DEFAULT_DISTANCE_THRESHOLD
    )
    distance_metric = _get_deepface_distance_metric()

    window_title = (
        "Mark Attendance - In - Press q to exit"
        if check_in
        else "Mark Attendance- Out - Press q to exit"
    )

    dataset_index = _load_dataset_embeddings_for_matching(
        model_name, detector_backend, enforce_detection
    )
    if not dataset_index:
        messages.error(
            request,
            "No encrypted training data available for matching. Please recreate the dataset.",
        )
        return redirect("home")

    spoof_detected = False

    try:
        with manager.frame_consumer() as consumer:
            while True:
                frame = consumer.read(timeout=1.0)
                if frame is None:
                    continue
                frame = imutils.resize(frame, width=800)
                frames_processed += 1

                try:
                    # Use DeepFace to find matching faces in the database
                    representations = DeepFace.represent(
                        img_path=frame,
                        model_name=model_name,
                        detector_backend=detector_backend,
                        enforce_detection=enforce_detection,
                    )

                    embedding_vector, facial_area = _extract_first_embedding(representations)
                    if embedding_vector is None:
                        continue

                    frame_embedding = np.array(embedding_vector, dtype=float)
                    match = _find_closest_dataset_match(
                        frame_embedding, dataset_index, distance_metric
                    )
                    if match is None:
                        continue

                    username, distance_value, identity_path = match
                    normalized_region = _normalize_face_region(facial_area)
                    spoofed = not _passes_liveness_check(frame, normalized_region)

                    if distance_value > distance_threshold:
                        logger.info(
                            "Ignoring potential match for '%s' due to high distance %.4f",
                            Path(identity_path).parent.name,
                            distance_value,
                        )
                        continue

                    if spoofed:
                        spoof_detected = True
                        logger.warning(
                            "Spoofing attempt blocked while marking attendance for '%s'",
                            username or Path(identity_path).parent.name,
                        )

                        if normalized_region:
                            x = int(normalized_region["x"])
                            y = int(normalized_region["y"])
                            w = int(normalized_region["w"])
                            h = int(normalized_region["h"])
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(
                                frame,
                                "Spoof detected",
                                (x, max(y - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )
                        continue

                    if username:
                        present[username] = True
                        logger.info("Recognized %s with distance %.4f", username, distance_value)

                        if normalized_region:
                            x = int(normalized_region["x"])
                            y = int(normalized_region["y"])
                            w = int(normalized_region["w"])
                            h = int(normalized_region["h"])
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                username,
                                (x, max(y - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 255, 0),
                                2,
                            )

                except Exception as e:
                    logger.error("Error during face recognition loop: %s", e)

                if not headless:
                    cv2.imshow(window_title, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    if frame_pause:
                        time.sleep(frame_pause)
                    if frames_processed >= max_frames:
                        logger.info("Headless mode reached frame limit of %d frames", max_frames)
                        break
    finally:
        if not headless:
            cv2.destroyAllWindows()

    if spoof_detected:
        messages.error(request, LIVENESS_FAILURE_MESSAGE)

    # Update the database based on whether it's a check-in or check-out
    if check_in:
        update_attendance_in_db_in(present)
    else:
        update_attendance_in_db_out(present)

    return redirect("home")


@login_required
@attendance_rate_limited
def mark_your_attendance(request):
    """View to handle marking time-in."""
    return _mark_attendance(request, check_in=True)


@login_required
@attendance_rate_limited
def mark_your_attendance_out(request):
    """View to handle marking time-out."""
    return _mark_attendance(request, check_in=False)


@login_required
def train(request):
    """
    This view is now obsolete, as DeepFace handles recognition implicitly
    by searching through image folders. It redirects to the dashboard with an
    informational message.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    messages.info(
        request,
        "The training process is now automatic. Just add photos for new users.",
    )
    return redirect("dashboard")


@login_required
def not_authorised(request):
    """Render a page for users trying to access unauthorized areas."""
    return render(request, "recognition/not_authorised.html")


@login_required
def view_attendance_home(request):
    """
    Render the main attendance viewing page for admins.

    This view displays summary statistics and generates weekly attendance graphs.
    """
    total_num_of_emp = total_number_employees()
    emp_present_today = employees_present_today()
    this_week_graph_url = this_week_emp_count_vs_date()
    last_week_graph_url = last_week_emp_count_vs_date()
    context = {
        "total_num_of_emp": total_num_of_emp,
        "emp_present_today": emp_present_today,
        "this_week_graph_url": this_week_graph_url,
        "last_week_graph_url": last_week_graph_url,
    }
    return render(request, "recognition/view_attendance_home.html", context)


@login_required
def view_attendance_date(request):
    """
    Admin view to see attendance for all employees on a specific date.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    qs: Optional[QuerySet[Present]] = None
    chart_url: Optional[str] = None
    if request.method == "POST":
        form = DateForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data["date"]
            logger.debug("Admin %s viewing attendance for date %s", request.user, date)

            time_qs = Time.objects.filter(date=date)
            present_qs = Present.objects.filter(date=date)

            if present_qs.exists():
                qs, chart_url = hours_vs_employee_given_date(present_qs, time_qs)
            else:
                messages.warning(request, "No records for the selected date.")
    else:
        form = DateForm()

    context = {
        "form": form,
        "qs": qs,
        "hours_vs_employee_chart_url": chart_url,
    }
    return render(request, "recognition/view_attendance_date.html", context)


@login_required
def view_attendance_employee(request):
    """
    Admin view to see attendance for a specific employee over a date range.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    qs: Optional[QuerySet[Present]] = None
    chart_url: Optional[str] = None
    if request.method == "POST":
        form = UsernameAndDateForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            date_from = form.cleaned_data["date_from"]
            date_to = form.cleaned_data["date_to"]

            if date_to < date_from:
                messages.warning(
                    request, "Invalid date selection: 'To' date cannot be before 'From' date."
                )
                return redirect("view-attendance-employee")

            user = User.objects.filter(username=username).first()
            if user:
                time_qs = Time.objects.filter(
                    user=user, date__gte=date_from, date__lte=date_to
                ).order_by("-date")
                present_qs = Present.objects.filter(
                    user=user, date__gte=date_from, date__lte=date_to
                ).order_by("-date")

                if present_qs.exists():
                    qs, chart_url = hours_vs_date_given_employee(present_qs, time_qs, admin=True)
                else:
                    messages.warning(request, "No records for the selected duration.")
            else:
                messages.warning(request, "Username not found.")
    else:
        form = UsernameAndDateForm()

    context = {
        "form": form,
        "qs": qs,
        "hours_vs_date_chart_url": chart_url,
    }
    return render(request, "recognition/view_attendance_employee.html", context)


@login_required
def view_my_attendance_employee_login(request):
    """
    Employee-specific view to see their own attendance over a date range.
    """
    if request.user.is_staff or request.user.is_superuser:
        return redirect("not-authorised")

    qs: Optional[QuerySet[Present]] = None
    chart_url: Optional[str] = None
    if request.method == "POST":
        form = DateForm_2(request.POST)
        if form.is_valid():
            user = request.user
            date_from = form.cleaned_data["date_from"]
            date_to = form.cleaned_data["date_to"]

            if date_to < date_from:
                messages.warning(request, "Invalid date selection.")
                return redirect("view-my-attendance-employee-login")

            time_qs = Time.objects.filter(
                user=user, date__gte=date_from, date__lte=date_to
            ).order_by("-date")
            present_qs = Present.objects.filter(
                user=user, date__gte=date_from, date__lte=date_to
            ).order_by("-date")

            if present_qs.exists():
                qs, chart_url = hours_vs_date_given_employee(present_qs, time_qs, admin=False)
            else:
                messages.warning(request, "No records for the selected duration.")

    else:
        form = DateForm_2()

    context = {
        "form": form,
        "qs": qs,
        "hours_vs_date_chart_url": chart_url,
    }
    return render(request, "recognition/view_my_attendance_employee_login.html", context)


def _get_deepface_options() -> Dict[str, Any]:
    """Return the configured DeepFace options merged with defaults."""

    defaults: Dict[str, Any] = {
        "backend": "opencv",
        "model": "Facenet",
        "detector_backend": "ssd",
        "distance_metric": "euclidean_l2",
        "enforce_detection": False,
        "anti_spoofing": True,
    }

    configured = getattr(settings, "DEEPFACE_OPTIMIZATIONS", {})
    if not isinstance(configured, dict):
        configured = {}

    options = defaults | configured
    distance_metric = str(options.get("distance_metric", "euclidean_l2")).lower()
    options["distance_metric"] = distance_metric

    enforce_value = options.get("enforce_detection", False)
    if isinstance(enforce_value, str):
        enforce_value = enforce_value.lower() in {"1", "true", "yes", "on"}
    else:
        enforce_value = bool(enforce_value)
    options["enforce_detection"] = enforce_value

    spoof_value = options.get("anti_spoofing", True)
    if isinstance(spoof_value, str):
        spoof_value = spoof_value.lower() in {"1", "true", "yes", "on"}
    else:
        spoof_value = bool(spoof_value)
    options["anti_spoofing"] = spoof_value
    return options


def _get_face_detection_backend() -> str:
    """
    Return the configured face detection backend for DeepFace.

    Defaults to 'opencv' if not specified in Django settings.

    Returns:
        The name of the backend (e.g., 'opencv', 'ssd', 'dlib').
    """
    return str(_get_deepface_options().get("detector_backend", "opencv"))


def _get_face_recognition_model() -> str:
    """
    Return the face recognition model name configured in Django settings.

    This value is used to select the pre-trained model for face recognition.

    Returns:
        The name of the face recognition model.
    """
    return str(_get_deepface_options().get("model", "VGG-Face"))


def _should_enforce_detection() -> bool:
    """Return whether DeepFace should enforce detection failures."""

    return bool(_get_deepface_options().get("enforce_detection", False))


def _get_deepface_distance_metric() -> str:
    """Return the configured embedding distance metric."""

    return str(_get_deepface_options().get("distance_metric", "euclidean_l2"))


def _is_liveness_enabled() -> bool:
    """Return whether anti-spoofing checks are enabled."""

    return bool(_get_deepface_options().get("anti_spoofing", True))


def _get_recognition_training_test_split_ratio() -> float:
    """
    Return the train-test split ratio for model validation.

    Defaults to 0.25 if not specified in Django settings.

    Returns:
        The proportion of the dataset to be used for testing.
    """
    return float(getattr(settings, "RECOGNITION_TRAINING_TEST_SPLIT_RATIO", 0.25))


def _get_recognition_training_seed() -> int:
    """
    Return the fixed seed for reproducible train-test splits.

    Defaults to 42 if not specified in Django settings.

    Returns:
        An integer seed for random operations.
    """
    return int(getattr(settings, "RECOGNITION_TRAINING_SEED", 42))


# ========== Views ==========


@login_required
def train_view(request):
    """
    Train the face recognition model and evaluate its performance.

    This view triggers the training process, which involves:
    1. Finding all user-specific image directories.
    2. Extracting face embeddings and splitting data into train/test sets.
    3. Training a Support Vector Classifier (SVC) on the training data.
    4. Evaluating the model on the test data and generating performance metrics.
    5. Saving the trained model, class names, and evaluation report.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        messages.error(request, "Not authorised to perform this action")
        return redirect("not_authorised")

    logger.info("Training of the model has been initiated by %s.", request.user)

    # --- 1. Data Preparation ---
    image_paths = sorted(TRAINING_DATASET_ROOT.glob("*/*.jpg"))
    if not image_paths:
        messages.error(request, "No training data found. Please add photos for users.")
        return render(request, "recognition/train.html", {"trained": False})

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
        messages.error(
            request,
            "No usable training data found after decrypting images. Please recreate the dataset.",
        )
        return render(request, "recognition/train.html", {"trained": False})

    unique_classes = sorted(set(class_names))

    if len(unique_classes) < 2:
        messages.error(
            request,
            "Training requires at least two different users with photos. "
            f"Found only {len(unique_classes)}.",
        )
        return render(request, "recognition/train.html", {"trained": False})

    logger.info(
        "Successfully extracted embeddings from %d encrypted images.", len(embedding_vectors)
    )

    # --- 3. Data Splitting ---
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

    # --- 4. Model Training ---
    logger.info("Training Support Vector Classifier...")
    model = SVC(gamma="auto", probability=True, random_state=random_seed)
    model.fit(X_train, y_train)
    logger.info("SVC training complete.")

    # --- 5. Model Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
    logger.info("Model Evaluation Report:\n%s", report)

    # --- 6. Save Artifacts ---
    try:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

        # Save the trained model
        model_path = DATA_ROOT / "svc.sav"
        model_bytes = pickle.dumps(model)
        model_path.write_bytes(encrypt_bytes(model_bytes))

        # Save the class names
        classes_path = DATA_ROOT / "classes.npy"
        buffer = io.BytesIO()
        np.save(buffer, unique_classes)
        classes_path.write_bytes(encrypt_bytes(buffer.getvalue()))

        # Save the evaluation report
        report_path = DATA_ROOT / "classification_report.txt"
        with report_path.open("w") as f:
            f.write("Model Evaluation Report\n")
            f.write("=========================\n\n")
            f.write(f"Timestamp: {timezone.now()}\n")
            f.write(f"Test Split Ratio: {test_split_ratio}\n")
            f.write(f"Random Seed: {random_seed}\n\n")
            f.write(f"Accuracy: {accuracy:.2%}\n")
            f.write(f"Precision (weighted): {precision:.2f}\n")
            f.write(f"Recall (weighted): {recall:.2f}\n")
            f.write(f"F1-Score (weighted): {f1:.2f}\n\n")
            f.write("Classification Report:\n")
            # Ensure we write a string (classification_report may be str or dict-like)
            f.write(str(report))

        logger.info("Successfully saved model, classes, and evaluation report.")
        messages.success(request, "Model trained and evaluated successfully!")

    except Exception as e:
        logger.error("Failed to save training artifacts: %s", e)
        messages.error(request, "Failed to save the trained model. Check permissions.")
        return render(request, "recognition/train.html", {"trained": False})

    context = {
        "trained": True,
        "accuracy": f"{accuracy:.2%}",
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "f1_score": f"{f1:.2f}",
        "class_report": str(report).replace("\n", "<br>"),
    }
    return render(request, "recognition/train.html", context)


@login_required
def mark_attendance_view(request, attendance_type):
    """
    View to handle marking attendance (check-in or check-out) using the trained model.

    Args:
        request: The Django HttpRequest object.
        attendance_type: A string indicating the attendance type ('in' or 'out').
    """
    if not (request.user.is_staff or request.user.is_superuser):
        messages.error(request, "Not authorised to perform this action")
        return redirect("not_authorised")

    logger.info(
        "Attendance marking process ('%s') initiated by %s.",
        attendance_type,
        request.user,
    )

    # --- Load cached embeddings to avoid rebuilding them per frame ---
    model_name = _get_face_recognition_model()
    detector_backend = _get_face_detection_backend()
    enforce_detection = _should_enforce_detection()
    dataset_index = _load_dataset_embeddings_for_matching(
        model_name, detector_backend, enforce_detection
    )
    if not dataset_index:
        logger.warning(
            "Cached embeddings are empty while marking attendance via SVC; continuing with model predictions."
        )

    # --- Load Model ---
    try:
        model_path = DATA_ROOT / "svc.sav"
        classes_path = DATA_ROOT / "classes.npy"
        encrypted_model = model_path.read_bytes()
        model = pickle.loads(decrypt_bytes(encrypted_model))  # noqa: F841

        encrypted_classes = classes_path.read_bytes()
        class_names = np.load(io.BytesIO(decrypt_bytes(encrypted_classes)), allow_pickle=True)
    except FileNotFoundError:
        messages.error(
            request, "Model not found. Please train the model before marking attendance."
        )
        return redirect("train")
    except Exception as e:
        logger.error("Failed to load the recognition model: %s", e)
        messages.error(request, "Failed to load the model. Check logs for details.")
        return redirect("train")

    # --- Video Stream ---
    manager = get_webcam_manager()
    present = {name: False for name in class_names}
    spoof_detected = False
    headless = _is_headless_environment()
    max_frames = int(getattr(settings, "RECOGNITION_HEADLESS_ATTENDANCE_FRAMES", 100))
    frame_pause = float(getattr(settings, "RECOGNITION_HEADLESS_FRAME_SLEEP", 0.01))

    frame_count = 0
    try:
        with manager.frame_consumer() as consumer:
            while True:
                frame = consumer.read(timeout=1.0)
                if frame is None:
                    continue
                frame = imutils.resize(frame, width=800)
                frame_count += 1

                try:
                    # --- Face Recognition ---
                    embeddings = DeepFace.represent(
                        img_path=frame,
                        model_name=model_name,
                        detector_backend=detector_backend,
                        enforce_detection=enforce_detection,
                    )

                    # Normalize and safely extract the first embedding + facial_area
                    embedding_vector = None
                    facial_area = None
                    if (
                        isinstance(embeddings, np.ndarray)
                        and embeddings.ndim == 2
                        and len(embeddings) > 0
                    ):
                        embedding_vector = list(embeddings[0])
                    elif isinstance(embeddings, list) and len(embeddings) > 0:
                        first = embeddings[0]
                        if isinstance(first, dict):
                            embedding_vector = first.get("embedding")
                            facial_area = first.get("facial_area")  # noqa: F841
                        elif isinstance(first, (list, tuple, np.ndarray)):
                            embedding_vector = list(first)
                        else:
                            try:
                                embedding_vector = list(first)
                            except Exception:
                                embedding_vector = None

                    if embedding_vector is not None:
                        predicted_name, spoofed, normalized_region = (
                            _predict_identity_from_embedding(
                                frame,
                                embedding_vector,
                                facial_area if isinstance(facial_area, dict) else None,
                                model,
                                class_names,
                                attendance_type,
                            )
                        )

                        if spoofed:
                            spoof_detected = True
                            if normalized_region:
                                x = int(normalized_region["x"])
                                y = int(normalized_region["y"])
                                w = int(normalized_region["w"])
                                h = int(normalized_region["h"])
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                cv2.putText(
                                    frame,
                                    "Spoof detected",
                                    (x, max(y - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2,
                                )
                            continue

                        if predicted_name:
                            present[predicted_name] = True

                            if normalized_region:
                                x = int(normalized_region["x"])
                                y = int(normalized_region["y"])
                                w = int(normalized_region["w"])
                                h = int(normalized_region["h"])
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(
                                    frame,
                                    predicted_name,
                                    (x, max(y - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                )

                except Exception as e:
                    logger.warning("Could not process frame for recognition: %s", e)

                if not headless:
                    cv2.imshow(f"Marking Attendance ({attendance_type})", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    if frame_pause:
                        time.sleep(frame_pause)

                if frame_count >= max_frames:
                    break
    finally:
        if not headless:
            cv2.destroyAllWindows()

    if spoof_detected:
        messages.error(request, LIVENESS_FAILURE_MESSAGE)

    # --- Update Database ---
    if attendance_type == "in":
        update_attendance_in_db_in(present)
        messages.success(request, "Checked-in users have been marked present.")
    elif attendance_type == "out":
        update_attendance_in_db_out(present)
        messages.success(request, "Checked-out users have been marked.")

    return redirect("home")
