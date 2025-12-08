"""Sample quality analysis utilities for training data assessment.

This module provides tools to analyze the quality and diversity of face
training samples, helping to identify gaps in dataset coverage and
suggest additional captures for fair recognition performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Thresholds for quality assessment
BRIGHTNESS_LOW_THRESHOLD = 60
BRIGHTNESS_HIGH_THRESHOLD = 200
SHARPNESS_MIN_THRESHOLD = 50.0
FACE_SIZE_MIN_RATIO = 0.15  # Face should be at least 15% of image


@dataclass
class SampleQualityMetrics:
    """Quality metrics for a single face sample.

    Attributes:
        image_path: Path to the evaluated image.
        brightness_score: Mean brightness (0-255), higher is brighter.
        lighting_bucket: Categorized lighting condition.
        sharpness_score: Laplacian variance, higher is sharper.
        is_sharp: Whether image meets sharpness threshold.
        face_size_ratio: Estimated face size as ratio of image.
        is_face_large_enough: Whether face meets size threshold.
        overall_quality: Combined quality score (0-1).
        issues: List of detected quality issues.
    """

    image_path: Optional[Path] = None
    brightness_score: float = 0.0
    lighting_bucket: str = "unknown"
    sharpness_score: float = 0.0
    is_sharp: bool = True
    face_size_ratio: float = 0.0
    is_face_large_enough: bool = True
    overall_quality: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class DatasetDiversityMetrics:
    """Diversity metrics for a collection of samples.

    Attributes:
        total_samples: Number of samples analyzed.
        lighting_distribution: Count per lighting bucket.
        quality_distribution: Count per quality tier.
        avg_quality_score: Mean quality score across samples.
        diversity_score: Overall diversity score (0-1).
        coverage_gaps: Identified gaps in coverage.
        recommendations: Suggested captures to improve diversity.
    """

    total_samples: int = 0
    lighting_distribution: Dict[str, int] = field(default_factory=dict)
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    avg_quality_score: float = 0.0
    diversity_score: float = 0.0
    coverage_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def _estimate_lighting_bucket(brightness: float) -> str:
    """Categorize brightness into lighting buckets."""
    if brightness < BRIGHTNESS_LOW_THRESHOLD:
        return "low_light"
    elif brightness < BRIGHTNESS_HIGH_THRESHOLD:
        return "moderate_light"
    else:
        return "bright_light"


def _calculate_sharpness(gray_image: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance."""
    try:
        import cv2

        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return float(laplacian.var())
    except ImportError:
        # Fallback: simple gradient-based sharpness estimate
        gx = np.diff(gray_image.astype(float), axis=1)
        gy = np.diff(gray_image.astype(float), axis=0)
        return float(np.mean(gx**2) + np.mean(gy**2))


def analyze_sample_quality(
    image: np.ndarray,
    image_path: Optional[Path] = None,
    face_region: Optional[Dict[str, int]] = None,
) -> SampleQualityMetrics:
    """Evaluate the quality of a single face sample.

    Args:
        image: Image as NumPy array (BGR or grayscale).
        image_path: Optional path for reference.
        face_region: Optional face bounding box with x, y, w, h keys.

    Returns:
        SampleQualityMetrics with quality assessment.
    """
    if image is None or not hasattr(image, "shape"):
        return SampleQualityMetrics(
            image_path=image_path,
            issues=["Invalid or empty image"],
            overall_quality=0.0,
        )

    metrics = SampleQualityMetrics(image_path=image_path)
    issues = []

    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image

    # Brightness analysis
    metrics.brightness_score = float(np.mean(gray))
    metrics.lighting_bucket = _estimate_lighting_bucket(metrics.brightness_score)

    if metrics.brightness_score < BRIGHTNESS_LOW_THRESHOLD:
        issues.append("Image is too dark - consider better lighting")
    elif metrics.brightness_score > BRIGHTNESS_HIGH_THRESHOLD:
        issues.append("Image may be overexposed - reduce lighting intensity")

    # Sharpness analysis
    metrics.sharpness_score = _calculate_sharpness(gray)
    metrics.is_sharp = metrics.sharpness_score >= SHARPNESS_MIN_THRESHOLD

    if not metrics.is_sharp:
        issues.append(f"Image is blurry (sharpness: {metrics.sharpness_score:.1f})")

    # Face size analysis
    if face_region:
        face_pixels = face_region.get("w", 0) * face_region.get("h", 0)
        image_pixels = image.shape[0] * image.shape[1]
        if image_pixels > 0:
            metrics.face_size_ratio = face_pixels / image_pixels
            metrics.is_face_large_enough = metrics.face_size_ratio >= FACE_SIZE_MIN_RATIO

            if not metrics.is_face_large_enough:
                issues.append("Face is too small in frame - move closer to camera")
    else:
        # Estimate face presence without explicit region
        center_region = gray[
            gray.shape[0] // 4 : 3 * gray.shape[0] // 4,
            gray.shape[1] // 4 : 3 * gray.shape[1] // 4,
        ]
        center_variance = np.var(center_region)
        if center_variance < 100:
            issues.append("Face may not be properly positioned in frame")

    metrics.issues = issues

    # Calculate overall quality score
    quality_components = [
        1.0 if BRIGHTNESS_LOW_THRESHOLD <= metrics.brightness_score <= BRIGHTNESS_HIGH_THRESHOLD else 0.5,
        1.0 if metrics.is_sharp else 0.3,
        1.0 if metrics.is_face_large_enough else 0.5,
    ]
    metrics.overall_quality = sum(quality_components) / len(quality_components)

    return metrics


def assess_dataset_diversity(
    samples: List[SampleQualityMetrics],
    target_samples_per_lighting: int = 3,
) -> DatasetDiversityMetrics:
    """Analyze the diversity of a collection of face samples.

    Args:
        samples: List of analyzed sample metrics.
        target_samples_per_lighting: Minimum samples desired per lighting condition.

    Returns:
        DatasetDiversityMetrics with diversity assessment.
    """
    if not samples:
        return DatasetDiversityMetrics(
            coverage_gaps=["No samples available"],
            recommendations=["Capture initial face samples"],
        )

    metrics = DatasetDiversityMetrics(total_samples=len(samples))

    # Analyze lighting distribution
    lighting_counts: Dict[str, int] = {"low_light": 0, "moderate_light": 0, "bright_light": 0}
    quality_tiers: Dict[str, int] = {"excellent": 0, "good": 0, "poor": 0}
    quality_scores = []

    for sample in samples:
        # Lighting
        bucket = sample.lighting_bucket
        if bucket in lighting_counts:
            lighting_counts[bucket] += 1

        # Quality tier
        if sample.overall_quality >= 0.8:
            quality_tiers["excellent"] += 1
        elif sample.overall_quality >= 0.5:
            quality_tiers["good"] += 1
        else:
            quality_tiers["poor"] += 1

        quality_scores.append(sample.overall_quality)

    metrics.lighting_distribution = lighting_counts
    metrics.quality_distribution = quality_tiers
    metrics.avg_quality_score = float(np.mean(quality_scores)) if quality_scores else 0.0

    # Identify coverage gaps
    gaps = []
    recommendations = []

    for bucket, count in lighting_counts.items():
        if count < target_samples_per_lighting:
            bucket_display = bucket.replace("_", " ")
            gaps.append(f"Insufficient {bucket_display} samples ({count}/{target_samples_per_lighting})")
            recommendations.append(f"Capture {target_samples_per_lighting - count} more images in {bucket_display} conditions")

    # Check for quality issues
    if quality_tiers["poor"] > len(samples) * 0.3:
        gaps.append("Too many low-quality samples")
        recommendations.append("Re-capture images with better focus and lighting")

    metrics.coverage_gaps = gaps
    metrics.recommendations = recommendations

    # Calculate diversity score
    total_lighting = sum(lighting_counts.values())
    if total_lighting > 0:
        # Entropy-based diversity (higher when more evenly distributed)
        proportions = [c / total_lighting for c in lighting_counts.values() if c > 0]
        entropy = -sum(p * np.log(p + 1e-10) for p in proportions)
        max_entropy = np.log(len(lighting_counts))
        lighting_diversity = entropy / max_entropy if max_entropy > 0 else 0

        # Coverage-based diversity
        covered_buckets = sum(1 for c in lighting_counts.values() if c >= target_samples_per_lighting)
        coverage_score = covered_buckets / len(lighting_counts)

        metrics.diversity_score = (lighting_diversity + coverage_score) / 2
    else:
        metrics.diversity_score = 0.0

    return metrics


def get_collection_recommendations(
    current_samples: List[SampleQualityMetrics],
    target_total: int = 10,
    target_per_lighting: int = 3,
) -> List[str]:
    """Generate actionable recommendations for improving dataset diversity.

    Args:
        current_samples: List of currently captured sample metrics.
        target_total: Target total number of samples per user.
        target_per_lighting: Target samples per lighting condition.

    Returns:
        List of specific recommendations.
    """
    recommendations = []

    if not current_samples:
        recommendations.append("Start by capturing face images under bright lighting")
        recommendations.append("Ensure face is centered and clearly visible")
        return recommendations

    diversity = assess_dataset_diversity(current_samples, target_per_lighting)

    # Add diversity recommendations
    recommendations.extend(diversity.recommendations)

    # Check total count
    if len(current_samples) < target_total:
        needed = target_total - len(current_samples)
        recommendations.append(f"Capture {needed} more images to reach recommended total of {target_total}")

    # Suggest pose variations if we have enough samples
    if len(current_samples) >= 4 and not diversity.coverage_gaps:
        recommendations.append("Consider adding slight head turns (left/right) for better profile coverage")

    if not recommendations:
        recommendations.append("Dataset meets diversity recommendations - no additional captures needed")

    return recommendations


def load_and_analyze_sample(image_path: Path) -> SampleQualityMetrics:
    """Load an image and analyze its quality.

    Args:
        image_path: Path to the image file.

    Returns:
        SampleQualityMetrics for the image.
    """
    try:
        import cv2

        image = cv2.imread(str(image_path))
    except ImportError:
        try:
            from PIL import Image

            pil_img = Image.open(image_path)
            image = np.array(pil_img)
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", image_path, exc)
            return SampleQualityMetrics(
                image_path=image_path,
                issues=[f"Failed to load image: {exc}"],
                overall_quality=0.0,
            )

    if image is None:
        return SampleQualityMetrics(
            image_path=image_path,
            issues=["Failed to load image"],
            overall_quality=0.0,
        )

    return analyze_sample_quality(image, image_path)


__all__ = [
    "DatasetDiversityMetrics",
    "SampleQualityMetrics",
    "analyze_sample_quality",
    "assess_dataset_diversity",
    "get_collection_recommendations",
    "load_and_analyze_sample",
]
