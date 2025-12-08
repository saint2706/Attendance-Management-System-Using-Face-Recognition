"""Multi-frame consistency verification for replay attack detection.

Analyzes temporal patterns across frames to detect:
- Static replay (same frame repeated)
- Video replay (predictable video patterns)
- Screen flicker/artifacts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

ArrayLike = np.ndarray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameConsistencyResult:
    """Result of multi-frame consistency verification."""

    passed: bool
    confidence: float
    is_static_replay: bool
    periodicity_score: float
    unique_frame_ratio: float
    frames_analyzed: int


def _to_grayscale(frame: ArrayLike) -> ArrayLike:
    """Convert frame to grayscale if needed."""
    if frame is None or frame.size == 0:
        return np.array([])

    if frame.ndim == 3:
        if hasattr(cv2, "cvtColor"):
            try:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception:
                pass
        return frame.mean(axis=2).astype(np.uint8)

    return frame


def _resize_for_hash(frame: ArrayLike, size: int = 32) -> ArrayLike:
    """Resize frame for perceptual hashing."""
    if frame is None or frame.size == 0:
        return np.array([])

    if hasattr(cv2, "resize"):
        try:
            return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        except Exception:
            pass

    # Fallback: simple subsampling
    h, w = frame.shape[:2]
    step_h = max(1, h // size)
    step_w = max(1, w // size)
    return frame[::step_h, ::step_w][:size, :size]


def compute_frame_hash(frame: ArrayLike, hash_size: int = 8) -> str:
    """Compute a perceptual hash for a frame using average hash algorithm.

    Similar frames will have similar hashes, allowing detection of
    duplicate or near-duplicate frames in replay attacks.

    Args:
        frame: Input frame (color or grayscale).
        hash_size: Size of the hash (hash_size x hash_size bits).

    Returns:
        Hexadecimal string representation of the hash.
    """
    if frame is None or frame.size == 0:
        return ""

    gray = _to_grayscale(frame)
    if gray.size == 0:
        return ""

    # Resize to small square
    resized = _resize_for_hash(gray, hash_size)
    if resized.size == 0:
        return ""

    # Compute mean and create binary hash
    mean_val = np.mean(resized)
    binary = resized > mean_val

    # Convert to hexadecimal string
    hash_bits = binary.flatten()
    hash_int = 0
    for bit in hash_bits:
        hash_int = (hash_int << 1) | int(bit)

    return format(hash_int, f"0{hash_size * hash_size // 4}x")


def _hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hashes."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 64  # Maximum distance

    distance = 0
    try:
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        xor = int1 ^ int2
        distance = bin(xor).count("1")
    except ValueError:
        return 64

    return distance


def detect_static_replay(
    frames: Sequence[ArrayLike],
    similarity_threshold: int = 5,
) -> tuple[bool, float]:
    """Detect if frames are static (same frame repeated).

    Static replay attacks show identical or nearly identical frames,
    while live faces show natural micro-movements.

    Args:
        frames: Sequence of frames to analyze.
        similarity_threshold: Maximum Hamming distance for "same" frame.

    Returns:
        Tuple of (is_static, unique_frame_ratio).
    """
    if not frames or len(frames) < 2:
        return False, 1.0

    hashes: list[str] = []
    for frame in frames:
        h = compute_frame_hash(frame)
        if h:
            hashes.append(h)

    if len(hashes) < 2:
        return False, 1.0

    # Count unique frames using clustering
    unique_count = 1
    reference_hash = hashes[0]
    unique_hashes = [reference_hash]

    for h in hashes[1:]:
        is_unique = True
        for uh in unique_hashes:
            if _hamming_distance(h, uh) <= similarity_threshold:
                is_unique = False
                break
        if is_unique:
            unique_hashes.append(h)
            unique_count += 1

    unique_ratio = unique_count / len(hashes)

    # If very few unique frames, likely static replay
    is_static = unique_ratio < 0.3  # Less than 30% unique frames

    return is_static, unique_ratio


def _compute_frame_differences(frames: Sequence[ArrayLike]) -> list[float]:
    """Compute inter-frame differences for periodicity detection."""
    differences: list[float] = []

    for i in range(len(frames) - 1):
        if frames[i] is None or frames[i + 1] is None:
            continue

        gray1 = _to_grayscale(frames[i])
        gray2 = _to_grayscale(frames[i + 1])

        if gray1.size == 0 or gray2.size == 0:
            continue

        # Resize to same size for comparison
        target_size = 64
        resized1 = _resize_for_hash(gray1, target_size)
        resized2 = _resize_for_hash(gray2, target_size)

        if resized1.shape != resized2.shape:
            continue

        diff = np.mean(np.abs(resized1.astype(float) - resized2.astype(float)))
        differences.append(float(diff))

    return differences


def detect_periodic_patterns(
    frames: Sequence[ArrayLike],
    periodicity_threshold: float = 0.7,
) -> float:
    """Detect periodic patterns that indicate video replay.

    Video replays often show periodic patterns due to the video
    source's framerate or looping behavior.

    Args:
        frames: Sequence of frames to analyze.
        periodicity_threshold: Threshold for periodic pattern detection.

    Returns:
        Periodicity score (0.0 = random/natural, 1.0 = highly periodic).
    """
    if not frames or len(frames) < 4:
        return 0.0

    differences = _compute_frame_differences(frames)
    if len(differences) < 3:
        return 0.0

    diff_array = np.array(differences)

    # Compute autocorrelation to detect periodicity
    mean_diff = np.mean(diff_array)
    diff_centered = diff_array - mean_diff

    variance = np.var(diff_centered)
    if variance < 1e-10:
        # Very low variance in differences suggests static or very periodic
        return 0.9

    # Simple periodicity check: look for repeating patterns
    # Using FFT for frequency analysis
    try:
        fft = np.fft.fft(diff_centered)
        power_spectrum = np.abs(fft) ** 2

        # Exclude DC component and look for dominant frequencies
        power_spectrum[0] = 0
        if len(power_spectrum) > 2:
            power_spectrum[1] = 0  # Also exclude very low frequencies

        max_power = np.max(power_spectrum)
        total_power = np.sum(power_spectrum)

        if total_power > 0:
            # High ratio of max to total indicates periodic pattern
            periodicity = max_power / total_power
        else:
            periodicity = 0.0
    except Exception:
        periodicity = 0.0

    return float(min(1.0, periodicity))


def check_frame_consistency(
    frames: Sequence[ArrayLike],
    *,
    min_frames: int = 5,
    static_threshold: int = 5,
    periodicity_threshold: float = 0.7,
) -> FrameConsistencyResult:
    """Perform full multi-frame consistency verification.

    Combines static replay detection and periodicity analysis to
    identify replay attacks while allowing natural face movements.

    Args:
        frames: Sequence of frames to analyze.
        min_frames: Minimum frames required for analysis.
        static_threshold: Hamming distance threshold for static detection.
        periodicity_threshold: Threshold for periodicity detection.

    Returns:
        FrameConsistencyResult with pass/fail and details.
    """
    if not frames or len(frames) < min_frames:
        return FrameConsistencyResult(
            passed=False,
            confidence=0.0,
            is_static_replay=False,
            periodicity_score=0.0,
            unique_frame_ratio=0.0,
            frames_analyzed=len(frames) if frames else 0,
        )

    # Check for static replay
    is_static, unique_ratio = detect_static_replay(frames, static_threshold)

    # Check for periodic patterns
    periodicity_score = detect_periodic_patterns(frames, periodicity_threshold)

    # Decision logic:
    # - Not static (frames are changing) = good
    # - Low periodicity (natural movement) = good
    # - High unique frame ratio = good
    is_replay = is_static or periodicity_score > periodicity_threshold

    passed = not is_replay

    # Confidence calculation
    if passed:
        # Higher unique ratio and lower periodicity = higher confidence
        confidence = unique_ratio * 0.6 + (1.0 - periodicity_score) * 0.4
    else:
        # Failed check - low confidence
        confidence = 0.2

    return FrameConsistencyResult(
        passed=passed,
        confidence=confidence,
        is_static_replay=is_static,
        periodicity_score=periodicity_score,
        unique_frame_ratio=unique_ratio,
        frames_analyzed=len(frames),
    )


__all__ = [
    "FrameConsistencyResult",
    "compute_frame_hash",
    "detect_static_replay",
    "detect_periodic_patterns",
    "check_frame_consistency",
]
