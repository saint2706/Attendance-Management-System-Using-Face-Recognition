"""Lightweight liveness utilities for the recognition pipeline."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

ArrayLike = np.ndarray

logger = logging.getLogger(__name__)


class ChallengeType(str, Enum):
    """Types of liveness challenges."""

    MOTION = "motion"
    BLINK = "blink"
    HEAD_TURN = "head_turn"
    ANTI_SPOOF = "anti_spoof"
    # Enhanced liveness challenge types
    DEPTH_CUE = "depth_cue"
    CNN_ANTI_SPOOF = "cnn_anti_spoof"
    FRAME_CONSISTENCY = "frame_consistency"


@dataclass(frozen=True)
class LivenessCheckResult:
    """Result of a liveness check with confidence scoring."""

    passed: bool
    confidence: float
    challenge_type: ChallengeType
    motion_score: Optional[float] = None
    frames_analyzed: int = 0
    threshold_used: Optional[float] = None
    details: dict[str, Any] = field(default_factory=dict)


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

    # ⚡ Performance: Convert all frames to float32 once to avoid redundant conversions in loop
    float_frames = [f.astype(np.float32) / 255.0 for f in frames]
    use_optical_flow = hasattr(cv2, "calcOpticalFlowFarneback") and hasattr(cv2, "cartToPolar")

    for prev, curr in zip(float_frames, float_frames[1:]):
        success = False
        if use_optical_flow:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev,
                    curr,
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
                        success = True
            except Exception:  # pragma: no cover - fall back to absolute differences
                pass

        if not success:
            diff = np.abs(curr - prev)
            magnitudes.append(float(np.mean(diff)))

    if not magnitudes:
        return None

    return float(np.median(magnitudes))


def _compute_horizontal_motion(frames: Sequence[ArrayLike]) -> tuple[float, float]:
    """Compute left/right motion components from optical flow.

    Returns:
        Tuple of (left_motion, right_motion) scores.
    """
    left_scores: list[float] = []
    right_scores: list[float] = []

    # ⚡ Performance: Convert all frames to float32 once
    float_frames = [f.astype(np.float32) / 255.0 for f in frames]
    use_optical_flow = hasattr(cv2, "calcOpticalFlowFarneback")

    for prev, curr in zip(float_frames, float_frames[1:]):
        if use_optical_flow:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev,
                    curr,
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
                    # Horizontal flow (x component)
                    horizontal = flow[..., 0]
                    left_motion = float(np.mean(np.maximum(-horizontal, 0)))
                    right_motion = float(np.mean(np.maximum(horizontal, 0)))
                    left_scores.append(left_motion)
                    right_scores.append(right_motion)
            except Exception:  # pragma: no cover
                pass

    if not left_scores:
        return 0.0, 0.0

    return float(np.mean(left_scores)), float(np.mean(right_scores))


def _detect_blink_pattern(frames: Sequence[ArrayLike]) -> tuple[int, float]:
    """Detect blink patterns using eye aspect ratio changes.

    Returns:
        Tuple of (blink_count, blink_confidence).
    """
    # This is a simplified blink detection using intensity variance
    # A full implementation would use facial landmark detection
    intensities: list[float] = []

    for frame in frames:
        if frame is None or frame.size == 0:
            continue
        # Focus on upper portion of face where eyes are
        height = frame.shape[0]
        eye_region = frame[int(height * 0.2) : int(height * 0.5), :]
        if eye_region.size > 0:
            intensities.append(float(np.mean(eye_region)))

    if len(intensities) < 3:
        return 0, 0.0

    # Detect intensity dips (potential blinks)
    intensities_arr = np.array(intensities)
    mean_intensity = np.mean(intensities_arr)
    threshold = mean_intensity * 0.85  # 15% drop indicates potential blink

    # Count transitions below threshold
    below_threshold = intensities_arr < threshold
    blink_count = 0
    in_blink = False

    for is_below in below_threshold:
        if is_below and not in_blink:
            blink_count += 1
            in_blink = True
        elif not is_below:
            in_blink = False

    # Confidence based on intensity variance
    variance = float(np.var(intensities_arr))
    confidence = min(1.0, variance / 100.0)  # Normalize variance to confidence

    return blink_count, confidence


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


def check_liveness_with_challenge(
    frames: Sequence[ArrayLike],
    *,
    face_region: Optional[dict[str, int]] = None,
    challenge_type: ChallengeType = ChallengeType.MOTION,
    motion_threshold: float = 1.1,
    blink_required: int = 1,
    head_turn_threshold: float = 0.02,
) -> LivenessCheckResult:
    """Perform a liveness check with the specified challenge type.

    Args:
        frames: Sequence of frames to analyze.
        face_region: Optional face bounding box for cropping.
        challenge_type: Type of liveness challenge to perform.
        motion_threshold: Threshold for motion-based liveness.
        blink_required: Number of blinks required for blink challenge.
        head_turn_threshold: Threshold for head turn detection.

    Returns:
        LivenessCheckResult with pass/fail status and confidence.
    """
    min_frames = 3
    prepared: list[ArrayLike] = []

    for frame in frames:
        prepared_frame = _prepare_gray_frame(frame, face_region)
        if prepared_frame is not None:
            prepared.append(prepared_frame)

    frames_analyzed = len(prepared)

    if frames_analyzed < min_frames:
        return LivenessCheckResult(
            passed=False,
            confidence=0.0,
            challenge_type=challenge_type,
            motion_score=None,
            frames_analyzed=frames_analyzed,
            threshold_used=motion_threshold,
            details={"error": "insufficient_frames", "required": min_frames},
        )

    if challenge_type == ChallengeType.MOTION:
        motion_score = _compute_motion_score(prepared)
        if motion_score is None:
            return LivenessCheckResult(
                passed=False,
                confidence=0.0,
                challenge_type=challenge_type,
                motion_score=None,
                frames_analyzed=frames_analyzed,
                threshold_used=motion_threshold,
                details={"error": "motion_computation_failed"},
            )

        passed = motion_score >= motion_threshold
        # Convert motion score to confidence (0-1 range)
        confidence = min(1.0, motion_score / (motion_threshold * 2))

        return LivenessCheckResult(
            passed=passed,
            confidence=confidence,
            challenge_type=challenge_type,
            motion_score=motion_score,
            frames_analyzed=frames_analyzed,
            threshold_used=motion_threshold,
            details={"raw_motion_score": motion_score},
        )

    elif challenge_type == ChallengeType.BLINK:
        blink_count, blink_confidence = _detect_blink_pattern(prepared)
        passed = blink_count >= blink_required

        return LivenessCheckResult(
            passed=passed,
            confidence=blink_confidence,
            challenge_type=challenge_type,
            motion_score=None,
            frames_analyzed=frames_analyzed,
            threshold_used=float(blink_required),
            details={"blink_count": blink_count, "required": blink_required},
        )

    elif challenge_type == ChallengeType.HEAD_TURN:
        left_motion, right_motion = _compute_horizontal_motion(prepared)
        total_horizontal = left_motion + right_motion
        passed = total_horizontal >= head_turn_threshold
        confidence = min(1.0, total_horizontal / (head_turn_threshold * 2))

        return LivenessCheckResult(
            passed=passed,
            confidence=confidence,
            challenge_type=challenge_type,
            motion_score=total_horizontal,
            frames_analyzed=frames_analyzed,
            threshold_used=head_turn_threshold,
            details={
                "left_motion": left_motion,
                "right_motion": right_motion,
                "total_horizontal": total_horizontal,
            },
        )

    # Default to motion-based check for unknown challenge types
    motion_score = _compute_motion_score(prepared)
    passed = motion_score is not None and motion_score >= motion_threshold
    confidence = min(1.0, (motion_score or 0) / (motion_threshold * 2))

    return LivenessCheckResult(
        passed=passed,
        confidence=confidence,
        challenge_type=ChallengeType.MOTION,
        motion_score=motion_score,
        frames_analyzed=frames_analyzed,
        threshold_used=motion_threshold,
        details={"fallback": True},
    )


def run_multi_challenge_liveness(
    frames: Sequence[ArrayLike],
    *,
    face_region: Optional[dict[str, int]] = None,
    challenges: Sequence[ChallengeType] | None = None,
    require_all: bool = False,
    motion_threshold: float = 1.1,
) -> tuple[bool, float, list[LivenessCheckResult]]:
    """Run multiple liveness challenges and aggregate results.

    Args:
        frames: Sequence of frames to analyze.
        face_region: Optional face bounding box for cropping.
        challenges: List of challenge types to run. Defaults to [MOTION].
        require_all: If True, all challenges must pass. If False, any can pass.
        motion_threshold: Threshold for motion-based challenges.

    Returns:
        Tuple of (overall_passed, aggregate_confidence, individual_results).
    """
    if challenges is None:
        challenges = [ChallengeType.MOTION]

    results: list[LivenessCheckResult] = []

    for challenge in challenges:
        result = check_liveness_with_challenge(
            frames,
            face_region=face_region,
            challenge_type=challenge,
            motion_threshold=motion_threshold,
        )
        results.append(result)

    if not results:
        return False, 0.0, []

    # passed_count can be used for logging/debugging if needed
    _ = sum(1 for r in results if r.passed)
    total_confidence = sum(r.confidence for r in results) / len(results)

    if require_all:
        overall_passed = all(r.passed for r in results)
    else:
        overall_passed = any(r.passed for r in results)

    return overall_passed, total_confidence, results


@dataclass(frozen=True)
class EnhancedLivenessResult:
    """Result of enhanced multi-signal liveness verification."""

    passed: bool
    confidence: float
    motion_passed: bool
    depth_passed: bool
    cnn_passed: bool
    consistency_passed: bool
    failure_reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def run_enhanced_liveness_verification(
    frames: Sequence[ArrayLike],
    *,
    face_region: Optional[dict[str, int]] = None,
    motion_threshold: float = 1.1,
    depth_variance_threshold: float = 0.1,
    cnn_threshold: float = 0.75,
    require_all: bool = True,
) -> EnhancedLivenessResult:
    """Run comprehensive enhanced liveness verification.

    Combines motion detection, depth cue analysis, CNN anti-spoofing,
    and multi-frame consistency verification for robust liveness assurance.

    Args:
        frames: Sequence of frames to analyze.
        face_region: Optional face bounding box for cropping.
        motion_threshold: Threshold for motion-based liveness.
        depth_variance_threshold: Threshold for depth variance.
        cnn_threshold: Confidence threshold for CNN anti-spoof.
        require_all: If True, all checks must pass.

    Returns:
        EnhancedLivenessResult with comprehensive pass/fail status.
    """
    failure_reasons: list[str] = []
    details: dict[str, Any] = {}

    # Default all checks to pass (will be set to actual results)
    motion_passed = True
    depth_passed = True
    cnn_passed = True
    consistency_passed = True

    min_frames = 5
    if len(frames) < min_frames:
        return EnhancedLivenessResult(
            passed=False,
            confidence=0.0,
            motion_passed=False,
            depth_passed=False,
            cnn_passed=False,
            consistency_passed=False,
            failure_reasons=[f"Insufficient frames: {len(frames)} < {min_frames}"],
            details={"frames_provided": len(frames), "min_required": min_frames},
        )

    # 1. Motion-based liveness (existing functionality)
    motion_result = check_liveness_with_challenge(
        frames,
        face_region=face_region,
        challenge_type=ChallengeType.MOTION,
        motion_threshold=motion_threshold,
    )
    motion_passed = motion_result.passed
    details["motion"] = {
        "passed": motion_result.passed,
        "score": motion_result.motion_score,
        "threshold": motion_threshold,
    }
    if not motion_passed:
        failure_reasons.append("Motion check failed: insufficient movement detected")

    # 2. Depth cue analysis
    try:
        from recognition.depth_estimator import analyze_depth_cues

        depth_result = analyze_depth_cues(
            frames,
            face_region=face_region,
            variance_threshold=depth_variance_threshold,
        )
        depth_passed = depth_result.passed
        details["depth"] = {
            "passed": depth_result.passed,
            "variance": depth_result.depth_variance,
            "flatness_ratio": depth_result.flatness_ratio,
            "threshold": depth_variance_threshold,
        }
        if not depth_passed:
            failure_reasons.append("Depth check failed: surface appears flat")
    except ImportError:
        logger.debug("Depth estimator not available, skipping depth check")
        details["depth"] = {"passed": True, "skipped": True}
    except Exception as exc:
        logger.warning("Depth analysis failed: %s", exc)
        details["depth"] = {"passed": True, "error": str(exc)}

    # 3. CNN anti-spoof check
    try:
        from recognition.anti_spoof_cnn import run_cnn_antispoof

        cnn_result = run_cnn_antispoof(frames, threshold=cnn_threshold)
        cnn_passed = cnn_result.is_real
        details["cnn_antispoof"] = {
            "passed": cnn_result.is_real,
            "confidence": cnn_result.confidence,
            "spoof_probability": cnn_result.spoof_probability,
            "model_available": cnn_result.model_available,
        }
        if not cnn_passed:
            failure_reasons.append("CNN anti-spoof check failed: spoof detected")
    except ImportError:
        logger.debug("CNN anti-spoof not available, skipping")
        details["cnn_antispoof"] = {"passed": True, "skipped": True}
    except Exception as exc:
        logger.warning("CNN anti-spoof failed: %s", exc)
        details["cnn_antispoof"] = {"passed": True, "error": str(exc)}

    # 4. Frame consistency verification
    try:
        from recognition.frame_consistency import check_frame_consistency

        consistency_result = check_frame_consistency(frames)
        consistency_passed = consistency_result.passed
        details["consistency"] = {
            "passed": consistency_result.passed,
            "is_static_replay": consistency_result.is_static_replay,
            "periodicity_score": consistency_result.periodicity_score,
            "unique_frame_ratio": consistency_result.unique_frame_ratio,
        }
        if not consistency_passed:
            if consistency_result.is_static_replay:
                failure_reasons.append("Frame consistency failed: static replay detected")
            else:
                failure_reasons.append("Frame consistency failed: periodic pattern detected")
    except ImportError:
        logger.debug("Frame consistency checker not available, skipping")
        details["consistency"] = {"passed": True, "skipped": True}
    except Exception as exc:
        logger.warning("Frame consistency check failed: %s", exc)
        details["consistency"] = {"passed": True, "error": str(exc)}

    # Aggregate results
    all_checks = [motion_passed, depth_passed, cnn_passed, consistency_passed]
    if require_all:
        overall_passed = all(all_checks)
    else:
        # At least motion + one other check must pass
        overall_passed = motion_passed and sum(all_checks) >= 3

    # Compute aggregate confidence
    confidences = [motion_result.confidence]
    if depth_passed:
        confidences.append(details.get("depth", {}).get("passed", 1.0) and 0.8 or 0.2)
    if cnn_passed and details.get("cnn_antispoof", {}).get("confidence"):
        confidences.append(details["cnn_antispoof"]["confidence"])
    if consistency_passed:
        confidences.append(0.9)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return EnhancedLivenessResult(
        passed=overall_passed,
        confidence=avg_confidence,
        motion_passed=motion_passed,
        depth_passed=depth_passed,
        cnn_passed=cnn_passed,
        consistency_passed=consistency_passed,
        failure_reasons=failure_reasons,
        details=details,
    )


__all__ = [
    "LivenessBuffer",
    "is_live_face",
    "ChallengeType",
    "LivenessCheckResult",
    "EnhancedLivenessResult",
    "check_liveness_with_challenge",
    "run_multi_challenge_liveness",
    "run_enhanced_liveness_verification",
]
