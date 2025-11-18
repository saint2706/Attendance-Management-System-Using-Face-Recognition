"""Lightweight liveness utilities for the recognition pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

ArrayLike = np.ndarray


@dataclass
class LivenessBuffer:
    """Fixed-size buffer that stores the most recent frames for liveness checks."""

    maxlen: int = 5
    _frames: deque[ArrayLike] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.maxlen = max(2, int(self.maxlen))
        self._frames = deque(maxlen=self.maxlen)

    def append(self, frame: Optional[ArrayLike]) -> None:
        """Append a frame to the buffer, copying it to avoid downstream mutation."""

        if frame is None:
            return
        if not isinstance(frame, np.ndarray):
            return
        if frame.size == 0:
            return
        self._frames.append(frame.copy())

    def extend(self, frames: Iterable[ArrayLike]) -> None:
        """Extend the buffer with additional frames."""

        for frame in frames:
            self.append(frame)

    def snapshot(self) -> list[ArrayLike]:
        """Return a copy of the current buffer contents."""

        return list(self._frames)

    def clear(self) -> None:
        """Remove all cached frames."""

        self._frames.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._frames)


def _crop_to_region(frame: ArrayLike, face_region: Optional[dict[str, int]]) -> ArrayLike:
    if not isinstance(face_region, dict):
        return frame

    height, width = frame.shape[:2]
    x = max(int(face_region.get("x", 0) or 0), 0)
    y = max(int(face_region.get("y", 0) or 0), 0)
    w = max(int(face_region.get("w", 0) or 0), 0)
    h = max(int(face_region.get("h", 0) or 0), 0)

    if w <= 0 or h <= 0:
        return frame

    x2 = min(x + w, width)
    y2 = min(y + h, height)
    if x >= x2 or y >= y2:
        return frame

    return frame[y:y2, x:x2]


def _prepare_gray_frame(
    frame: Optional[ArrayLike],
    face_region: Optional[dict[str, int]],
    *,
    target_size: int = 128,
) -> Optional[ArrayLike]:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None

    working = _crop_to_region(frame, face_region)
    if working.ndim == 3:
        converted = None
        if hasattr(cv2, "cvtColor"):
            try:
                converted = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
            except Exception:  # pragma: no cover - cv2 may be stubbed in tests
                converted = None
        if isinstance(converted, np.ndarray):
            working = converted
        else:
            working = working.mean(axis=2).astype(np.uint8)
    elif working.ndim != 2:
        return None

    if target_size > 0 and hasattr(cv2, "resize"):
        try:
            resized = cv2.resize(working, (target_size, target_size))
            if isinstance(resized, np.ndarray):
                working = resized
        except Exception:  # pragma: no cover - cv2 may be stubbed
            working = np.copy(working)
    else:
        working = np.copy(working)

    if hasattr(cv2, "GaussianBlur"):
        try:
            blurred = cv2.GaussianBlur(working, (5, 5), 0)
            if isinstance(blurred, np.ndarray):
                working = blurred
        except Exception:  # pragma: no cover - cv2 may be stubbed
            pass

    return working


def _compute_motion_score(frames: Sequence[ArrayLike]) -> Optional[float]:
    magnitudes: list[float] = []
    for prev, curr in zip(frames, frames[1:]):
        try:
            prev_float = prev.astype(np.float32) / 255.0
            curr_float = curr.astype(np.float32) / 255.0
            if hasattr(cv2, "calcOpticalFlowFarneback") and hasattr(cv2, "cartToPolar"):
                flow = cv2.calcOpticalFlowFarneback(
                    prev_float,
                    curr_float,
                    None,
                    0.5,
                    1,
                    11,
                    2,
                    5,
                    1.1,
                    0,
                )
                if isinstance(flow, np.ndarray):
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    if isinstance(magnitude, np.ndarray):
                        magnitudes.append(float(np.mean(magnitude)))
                        continue
        except Exception:  # pragma: no cover - fall back to absolute differences
            pass

        diff = np.abs(curr.astype(np.float32) - prev.astype(np.float32))
        magnitudes.append(float(np.mean(diff) / 255.0))

    if not magnitudes:
        return None

    return float(np.median(magnitudes))


def is_live_face(
    frames: Sequence[ArrayLike],
    *,
    face_region: Optional[dict[str, int]] = None,
    min_frames: int = 3,
    motion_threshold: Optional[float] = None,
    return_score: bool = False,
) -> Optional[float | bool]:
    """Return liveness score or decision based on inter-frame motion."""

    required_frames = max(2, int(min_frames))
    prepared: list[ArrayLike] = []
    for frame in frames:
        prepared_frame = _prepare_gray_frame(frame, face_region)
        if prepared_frame is not None:
            prepared.append(prepared_frame)
        if len(prepared) >= required_frames:
            break

    if len(prepared) < required_frames:
        return None

    score = _compute_motion_score(prepared)
    if score is None:
        return None

    if return_score or motion_threshold is None:
        return score

    return score >= motion_threshold


__all__ = ["LivenessBuffer", "is_live_face"]
