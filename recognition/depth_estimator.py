"""Monocular depth cue estimation for liveness detection.

Provides lightweight depth analysis without requiring specialized
depth sensors. Uses gradient-based pseudo-depth estimation to detect
flat surfaces (prints, screens) vs. 3D faces.
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
class DepthAnalysisResult:
    """Result of monocular depth cue analysis."""

    passed: bool
    confidence: float
    depth_variance: float
    gradient_score: float
    flatness_ratio: float
    frames_analyzed: int


def _crop_to_region(frame: ArrayLike, face_region: Optional[dict[str, int]]) -> ArrayLike:
    """Crop frame to face region if provided."""
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


def estimate_pseudo_depth(frame: ArrayLike) -> ArrayLike:
    """Estimate pseudo-depth map using Laplacian of Gaussian.

    Real 3D faces show varying depth patterns while flat surfaces
    (prints, screens) show uniform depth characteristics.

    Args:
        frame: Input grayscale frame.

    Returns:
        Pseudo-depth map as a float array.
    """
    if frame is None or frame.size == 0:
        return np.array([])

    gray = _to_grayscale(frame)
    if gray.size == 0:
        return np.array([])

    # Apply Gaussian blur to reduce noise
    if hasattr(cv2, "GaussianBlur"):
        try:
            blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
        except Exception:
            blurred = gray.astype(np.float32)
    else:
        blurred = gray.astype(np.float32)

    # Compute Laplacian for edge-based depth estimation
    if hasattr(cv2, "Laplacian"):
        try:
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            depth_map = np.abs(laplacian)
        except Exception:
            # Fallback to simple gradient
            depth_map = np.abs(np.gradient(blurred, axis=0)) + np.abs(np.gradient(blurred, axis=1))
    else:
        depth_map = np.abs(np.gradient(blurred, axis=0)) + np.abs(np.gradient(blurred, axis=1))

    return depth_map


def compute_depth_variance(
    depth_map: ArrayLike,
    face_region: Optional[dict[str, int]] = None,
) -> float:
    """Compute variance in the depth map.

    Higher variance indicates a 3D face with depth variation.
    Lower variance suggests a flat surface (print/screen).

    Args:
        depth_map: Pseudo-depth array from estimate_pseudo_depth.
        face_region: Optional face bounding box for cropping.

    Returns:
        Variance score (higher = more depth variation).
    """
    if depth_map is None or depth_map.size == 0:
        return 0.0

    cropped = _crop_to_region(depth_map, face_region)
    if cropped.size == 0:
        return 0.0

    variance = float(np.var(cropped))

    # Normalize by mean to get coefficient of variation
    mean_val = float(np.mean(cropped))
    if mean_val > 0:
        normalized_variance = variance / (mean_val**2)
    else:
        normalized_variance = variance

    return normalized_variance


def _compute_gradient_score(frame: ArrayLike) -> float:
    """Compute gradient magnitude score for texture analysis.

    Real faces have complex gradient patterns while prints may show
    uniform or periodic gradient patterns.
    """
    if frame is None or frame.size == 0:
        return 0.0

    gray = _to_grayscale(frame)
    if gray.size == 0:
        return 0.0

    if hasattr(cv2, "Sobel"):
        try:
            grad_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        except Exception:
            gradient_magnitude = np.abs(np.gradient(gray.astype(np.float32)))
    else:
        gradient_magnitude = np.abs(np.gradient(gray.astype(np.float32)))

    # Return normalized gradient energy
    return float(np.mean(gradient_magnitude))


def _compute_flatness_ratio(depth_map: ArrayLike) -> float:
    """Compute ratio of flat regions in the depth map.

    Screen/print attacks typically show high flatness ratios.
    """
    if depth_map is None or depth_map.size == 0:
        return 1.0

    # Threshold for considering a region "flat"
    threshold = np.mean(depth_map) * 0.1

    flat_pixels = np.sum(depth_map < threshold)
    total_pixels = depth_map.size

    if total_pixels == 0:
        return 1.0

    return float(flat_pixels / total_pixels)


def analyze_depth_cues(
    frames: Sequence[ArrayLike],
    face_region: Optional[dict[str, int]] = None,
    *,
    variance_threshold: float = 0.1,
    min_frames: int = 3,
) -> DepthAnalysisResult:
    """Analyze depth cues across multiple frames.

    Combines depth variance, gradient analysis, and flatness detection
    to determine if the face appears to be a live 3D face or a flat
    presentation attack.

    Args:
        frames: Sequence of frames to analyze.
        face_region: Optional face bounding box for cropping.
        variance_threshold: Minimum variance to consider live.
        min_frames: Minimum frames required for analysis.

    Returns:
        DepthAnalysisResult with pass/fail and confidence.
    """
    if not frames or len(frames) < min_frames:
        return DepthAnalysisResult(
            passed=False,
            confidence=0.0,
            depth_variance=0.0,
            gradient_score=0.0,
            flatness_ratio=1.0,
            frames_analyzed=len(frames) if frames else 0,
        )

    variances: list[float] = []
    gradient_scores: list[float] = []
    flatness_ratios: list[float] = []

    for frame in frames:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            continue

        cropped = _crop_to_region(frame, face_region)
        depth_map = estimate_pseudo_depth(cropped)

        if depth_map.size > 0:
            variances.append(compute_depth_variance(depth_map))
            gradient_scores.append(_compute_gradient_score(cropped))
            flatness_ratios.append(_compute_flatness_ratio(depth_map))

    if not variances:
        return DepthAnalysisResult(
            passed=False,
            confidence=0.0,
            depth_variance=0.0,
            gradient_score=0.0,
            flatness_ratio=1.0,
            frames_analyzed=0,
        )

    avg_variance = float(np.mean(variances))
    avg_gradient = float(np.mean(gradient_scores))
    avg_flatness = float(np.mean(flatness_ratios))

    # Decision logic:
    # - High variance = more depth = likely real
    # - High gradient = more texture = likely real
    # - Low flatness = less flat regions = likely real
    passed = avg_variance >= variance_threshold and avg_flatness < 0.8

    # Confidence based on how far above/below thresholds
    variance_factor = min(1.0, avg_variance / variance_threshold) if variance_threshold > 0 else 0.5
    flatness_factor = 1.0 - avg_flatness
    confidence = (variance_factor * 0.6 + flatness_factor * 0.4)

    return DepthAnalysisResult(
        passed=passed,
        confidence=confidence,
        depth_variance=avg_variance,
        gradient_score=avg_gradient,
        flatness_ratio=avg_flatness,
        frames_analyzed=len(variances),
    )


__all__ = [
    "DepthAnalysisResult",
    "estimate_pseudo_depth",
    "compute_depth_variance",
    "analyze_depth_cues",
]
