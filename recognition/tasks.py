"""Background jobs for face recognition and attendance processing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import sentry_sdk
from asgiref.sync import sync_to_async
from celery import shared_task
from deepface import DeepFace

from .pipeline import extract_embedding, find_closest_dataset_match, is_within_distance_threshold
from .utils import (
    get_face_detection_backend,
    get_face_recognition_model,
    get_deepface_distance_metric,
    load_dataset_embeddings_for_matching,
    update_attendance_in_db_in,
    update_attendance_in_db_out,
)

logger = logging.getLogger(__name__)


@shared_task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 3, "countdown": 5})
def recognize_face(self, image_bytes: list[int], direction: str) -> dict[str, Any]:
    """
    Celery task to perform face recognition on an image.

    Args:
        image_bytes: A list of integers representing the image bytes.
        direction: The attendance direction ('in' or 'out').

    Returns:
        A dictionary with the recognition result.
    """
    with sentry_sdk.start_transaction(op="task", name="recognize_face"):
        sentry_sdk.set_context("celery_task", {"task_id": self.request.id})
        try:
            image_np = np.frombuffer(bytes(image_bytes), np.uint8)
            frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            model_name = get_face_recognition_model()
            detector_backend = get_face_detection_backend()
            distance_metric = get_deepface_distance_metric()

            sentry_sdk.set_context(
                "recognition_context",
                {"model_name": model_name, "detector_backend": detector_backend},
            )

            logger.info("Starting face recognition for task %s", self.request.id)

            representations = DeepFace.represent(
                img_path=frame,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
            )
            embedding, _ = extract_embedding(representations)

            if embedding is None:
                logger.warning("No face detected in image for task %s", self.request.id)
                return {"status": "error", "message": "No face detected."}

            dataset = load_dataset_embeddings_for_matching(
                model_name, detector_backend, enforce_detection=False
            )
            match = find_closest_dataset_match(embedding, dataset, distance_metric)

            if match and is_within_distance_threshold(match[1], 0.4):
                username = match[0]
                logger.info(
                    "Face recognized for task %s: %s", self.request.id, username
                )

                # Update attendance in the database
                if direction == "in":
                    update_attendance_in_db_in({username: True})
                elif direction == "out":
                    update_attendance_in_db_out({username: True})

                return {"status": "success", "username": username}
            else:
                logger.warning("No match found for face in task %s", self.request.id)
                return {"status": "error", "message": "Face not recognized."}

        except Exception as e:
            logger.error(
                "Error in face recognition task %s: %s", self.request.id, e, exc_info=True
            )
            sentry_sdk.capture_exception(e)
            raise


async def process_single_attendance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Process a single attendance record asynchronously."""
    direction = str(record.get("direction", "in")).lower()
    payload = record.get("present") or {}
    attempt_ids = record.get("attempt_ids") or {}

    if direction not in {"in", "out"}:
        return {"status": "error", "error": "direction must be either 'in' or 'out'"}
    if not isinstance(payload, Mapping):
        return {"status": "error", "error": "present payload must be a mapping"}

    update_fn = update_attendance_in_db_in if direction == "in" else update_attendance_in_db_out
    update_async = sync_to_async(update_fn, thread_sensitive=True)

    try:
        await update_async(payload, attempt_ids=attempt_ids)
        return {"status": "success", "processed": len(payload)}
    except Exception as exc:
        logger.exception("Failed to process %s attendance payload: %s", direction, exc)
        return {"status": "error", "error": str(exc)}


@shared_task(bind=True)
def process_attendance_batch(
    self, records: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """Process a batch of attendance records using asyncio within a Celery task."""
    with sentry_sdk.start_transaction(op="task", name="process_attendance_batch"):
        sentry_sdk.set_context("celery_task", {"task_id": self.request.id})

        async def main():
            if not records:
                return []
            tasks = [process_single_attendance(record) for record in records]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(main())

        processed_count = sum(r.get("processed", 0) for r in results if isinstance(r, dict))
        error_count = sum(1 for r in results if isinstance(r, dict) and r["status"] == "error")

        logger.info(
            "Attendance batch %s processed with %d successes and %d errors.",
            self.request.id,
            processed_count,
            error_count,
        )
        return {"total": len(records or []), "success": processed_count, "error": error_count}
