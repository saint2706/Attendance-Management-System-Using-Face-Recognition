"""Domain adaptation utilities for handling different camera sources.

This module provides tools to analyze camera characteristics, normalize images
from different sources, and assess domain gaps between training and deployment
cameras to ensure consistent face recognition performance across devices.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraProfile:
    """Stores camera-specific characteristics for domain adaptation.

    Attributes:
        name: Human-readable camera identifier.
        mean_brightness: Average brightness of reference images (0-255).
        std_brightness: Standard deviation of brightness.
        mean_contrast: Average contrast ratio.
        color_temperature: Estimated color temperature (warm/neutral/cool).
        resolution: Typical resolution as (width, height).
        source_type: Camera type (webcam, kiosk, mobile, etc.).
    """

    name: str
    mean_brightness: float = 128.0
    std_brightness: float = 40.0
    mean_contrast: float = 1.0
    color_temperature: str = "neutral"
    resolution: Tuple[int, int] = (640, 480)
    source_type: str = "webcam"
    calibration_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "name": self.name,
            "mean_brightness": self.mean_brightness,
            "std_brightness": self.std_brightness,
            "mean_contrast": self.mean_contrast,
            "color_temperature": self.color_temperature,
            "resolution": list(self.resolution),
            "source_type": self.source_type,
            "calibration_samples": self.calibration_samples,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraProfile":
        """Deserialize from dictionary."""
        resolution = data.get("resolution", [640, 480])
        return cls(
            name=data["name"],
            mean_brightness=data.get("mean_brightness", 128.0),
            std_brightness=data.get("std_brightness", 40.0),
            mean_contrast=data.get("mean_contrast", 1.0),
            color_temperature=data.get("color_temperature", "neutral"),
            resolution=tuple(resolution) if isinstance(resolution, list) else resolution,
            source_type=data.get("source_type", "webcam"),
            calibration_samples=data.get("calibration_samples", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DomainGapResult:
    """Results from domain gap assessment between cameras.

    Attributes:
        source_camera: Name of the source (training) camera.
        target_camera: Name of the target (deployment) camera.
        brightness_shift: Difference in mean brightness.
        contrast_ratio: Ratio of contrast values.
        overall_gap_score: Composite gap score (0=identical, 1=very different).
        recommendations: Suggested actions to address the gap.
    """

    source_camera: str
    target_camera: str
    brightness_shift: float
    contrast_ratio: float
    overall_gap_score: float
    recommendations: List[str] = field(default_factory=list)


def estimate_camera_characteristics(
    images: List[np.ndarray],
    camera_name: str = "unknown",
    source_type: str = "webcam",
) -> CameraProfile:
    """Analyze a set of reference images to estimate camera characteristics.

    Args:
        images: List of BGR/RGB images as NumPy arrays.
        camera_name: Human-readable identifier for the camera.
        source_type: Type of camera (webcam, kiosk, mobile, etc.).

    Returns:
        CameraProfile with estimated characteristics.
    """
    if not images:
        logger.warning("No images provided for camera characterization")
        return CameraProfile(name=camera_name, source_type=source_type)

    brightness_values = []
    contrast_values = []
    resolutions = []

    for img in images:
        if img is None or not hasattr(img, "shape"):
            continue

        # Convert to grayscale for brightness analysis
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img

        brightness_values.append(np.mean(gray))
        contrast_values.append(np.std(gray))
        resolutions.append((img.shape[1], img.shape[0]))  # (width, height)

    if not brightness_values:
        return CameraProfile(name=camera_name, source_type=source_type)

    mean_brightness = float(np.mean(brightness_values))
    std_brightness = float(np.std(brightness_values))
    mean_contrast = float(np.mean(contrast_values))

    # Estimate color temperature from first color image
    color_temp = "neutral"
    for img in images:
        if img is not None and len(img.shape) == 3 and img.shape[2] >= 3:
            # Simple heuristic: compare red vs blue channel means
            red_mean = np.mean(img[:, :, 0] if img.shape[2] == 3 else img[:, :, 2])
            blue_mean = np.mean(img[:, :, 2] if img.shape[2] == 3 else img[:, :, 0])
            if red_mean > blue_mean * 1.1:
                color_temp = "warm"
            elif blue_mean > red_mean * 1.1:
                color_temp = "cool"
            break

    # Use most common resolution
    if resolutions:
        from collections import Counter

        resolution = Counter(resolutions).most_common(1)[0][0]
    else:
        resolution = (640, 480)

    return CameraProfile(
        name=camera_name,
        mean_brightness=mean_brightness,
        std_brightness=std_brightness,
        mean_contrast=mean_contrast,
        color_temperature=color_temp,
        resolution=resolution,
        source_type=source_type,
        calibration_samples=len(images),
    )


def normalize_for_camera(
    image: np.ndarray,
    source_profile: CameraProfile,
    target_profile: Optional[CameraProfile] = None,
) -> np.ndarray:
    """Apply camera-specific normalization to an image.

    Adjusts brightness and contrast to match the target camera profile,
    reducing domain shift effects on recognition accuracy.

    Args:
        image: Input image as NumPy array.
        source_profile: Profile of the camera that captured the image.
        target_profile: Profile to normalize towards. If None, uses neutral defaults.

    Returns:
        Normalized image as NumPy array.
    """
    if image is None or not hasattr(image, "shape"):
        return image

    # Default target profile (neutral reference)
    if target_profile is None:
        target_profile = CameraProfile(
            name="reference",
            mean_brightness=128.0,
            std_brightness=40.0,
            mean_contrast=50.0,
        )

    # Convert to float for processing
    img_float = image.astype(np.float32)

    # Brightness adjustment
    brightness_shift = target_profile.mean_brightness - source_profile.mean_brightness
    img_float = img_float + brightness_shift

    # Contrast adjustment (if source has valid contrast)
    if source_profile.mean_contrast > 0:
        contrast_scale = target_profile.mean_contrast / source_profile.mean_contrast
        # Limit contrast adjustment to prevent extreme changes
        contrast_scale = np.clip(contrast_scale, 0.5, 2.0)
        img_mean = np.mean(img_float)
        img_float = (img_float - img_mean) * contrast_scale + img_mean

    # Clip to valid range and convert back
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)


def assess_domain_gap(
    source_profile: CameraProfile,
    target_profile: CameraProfile,
) -> DomainGapResult:
    """Measure the domain gap between two camera profiles.

    Higher gap scores indicate greater potential for recognition accuracy
    degradation and may warrant additional calibration or threshold adjustment.

    Args:
        source_profile: Profile of training/enrollment camera.
        target_profile: Profile of deployment/recognition camera.

    Returns:
        DomainGapResult with gap metrics and recommendations.
    """
    # Calculate component differences
    brightness_shift = abs(target_profile.mean_brightness - source_profile.mean_brightness)
    brightness_gap = brightness_shift / 255.0  # Normalize to 0-1

    # Contrast ratio (avoid division by zero)
    if source_profile.mean_contrast > 0:
        contrast_ratio = target_profile.mean_contrast / source_profile.mean_contrast
    else:
        contrast_ratio = 1.0
    contrast_gap = abs(1.0 - contrast_ratio)  # 0 = same contrast

    # Resolution difference
    src_pixels = source_profile.resolution[0] * source_profile.resolution[1]
    tgt_pixels = target_profile.resolution[0] * target_profile.resolution[1]
    resolution_ratio = min(src_pixels, tgt_pixels) / max(src_pixels, tgt_pixels) if max(src_pixels, tgt_pixels) > 0 else 1.0
    resolution_gap = 1.0 - resolution_ratio

    # Color temperature mismatch
    color_gap = 0.0 if source_profile.color_temperature == target_profile.color_temperature else 0.2

    # Compute overall gap score (weighted average)
    overall_gap = (
        brightness_gap * 0.4 +
        contrast_gap * 0.3 +
        resolution_gap * 0.2 +
        color_gap * 0.1
    )
    overall_gap = min(overall_gap, 1.0)

    # Generate recommendations
    recommendations = []
    if brightness_gap > 0.2:
        recommendations.append(
            f"Significant brightness difference ({brightness_shift:.0f}). "
            "Consider adjusting lighting at deployment location or applying normalization."
        )
    if contrast_gap > 0.3:
        recommendations.append(
            f"Contrast ratio ({contrast_ratio:.2f}x) may affect recognition. "
            "Review camera settings or apply contrast normalization."
        )
    if resolution_gap > 0.3:
        recommendations.append(
            "Significant resolution difference. "
            "Consider capturing enrollment images at deployment resolution."
        )
    if color_gap > 0:
        recommendations.append(
            f"Color temperature mismatch ({source_profile.color_temperature} vs {target_profile.color_temperature}). "
            "May affect recognition in some lighting conditions."
        )
    if overall_gap > 0.4:
        recommendations.append(
            "High domain gap detected. Consider re-enrolling users with the deployment camera "
            "or adjusting recognition threshold for this camera."
        )

    return DomainGapResult(
        source_camera=source_profile.name,
        target_camera=target_profile.name,
        brightness_shift=brightness_shift,
        contrast_ratio=contrast_ratio,
        overall_gap_score=overall_gap,
        recommendations=recommendations,
    )


def save_camera_profile(profile: CameraProfile, path: Path) -> None:
    """Save a camera profile to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)
    logger.info("Saved camera profile '%s' to %s", profile.name, path)


def load_camera_profile(path: Path) -> CameraProfile:
    """Load a camera profile from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CameraProfile.from_dict(data)


def get_camera_profiles_dir() -> Path:
    """Return the directory for storing camera profiles."""
    from django.conf import settings

    base_dir = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    return base_dir / "configs" / "camera_profiles"


def list_camera_profiles() -> List[CameraProfile]:
    """List all saved camera profiles."""
    profiles_dir = get_camera_profiles_dir()
    if not profiles_dir.exists():
        return []

    profiles = []
    for profile_path in profiles_dir.glob("*.json"):
        try:
            profiles.append(load_camera_profile(profile_path))
        except Exception as exc:
            logger.warning("Failed to load camera profile %s: %s", profile_path, exc)

    return profiles


__all__ = [
    "CameraProfile",
    "DomainGapResult",
    "assess_domain_gap",
    "estimate_camera_characteristics",
    "get_camera_profiles_dir",
    "list_camera_profiles",
    "load_camera_profile",
    "normalize_for_camera",
    "save_camera_profile",
]
