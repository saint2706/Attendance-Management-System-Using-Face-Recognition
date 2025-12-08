"""
Configuration utilities for recognition views.

This module contains configuration getter functions for various recognition settings.
"""

from __future__ import annotations

from django.conf import settings


def get_face_recognition_model() -> str:
    """Return the configured DeepFace recognition model name."""
    return getattr(settings, "RECOGNITION_MODEL", "Facenet")


def get_face_detection_backend() -> str:
    """Return the configured face detection backend."""
    return getattr(settings, "RECOGNITION_DETECTOR_BACKEND", "opencv")


def should_enforce_detection() -> bool:
    """Return whether face detection should be strictly enforced."""
    return getattr(settings, "RECOGNITION_ENFORCE_DETECTION", True)


def get_deepface_distance_metric() -> str:
    """Return the distance metric for face recognition."""
    return getattr(settings, "RECOGNITION_DISTANCE_METRIC", "cosine")


def is_liveness_enabled() -> bool:
    """Return whether DeepFace liveness detection is enabled."""
    return getattr(settings, "RECOGNITION_LIVENESS_ENABLED", False)


def is_lightweight_liveness_enabled() -> bool:
    """Return whether lightweight motion-based liveness is enabled."""
    return getattr(settings, "RECOGNITION_LIGHTWEIGHT_LIVENESS_ENABLED", True)


def get_lightweight_liveness_min_frames() -> int:
    """Return the minimum frames required for lightweight liveness check."""
    return getattr(settings, "RECOGNITION_LIGHTWEIGHT_LIVENESS_MIN_FRAMES", 3)


def get_lightweight_liveness_threshold() -> float:
    """Return the threshold for lightweight liveness score."""
    return getattr(settings, "RECOGNITION_LIGHTWEIGHT_LIVENESS_THRESHOLD", 0.3)


def get_lightweight_liveness_window() -> int:
    """Return the frame window size for liveness buffer."""
    return getattr(settings, "RECOGNITION_LIGHTWEIGHT_LIVENESS_WINDOW", 10)


def is_enhanced_liveness_enabled() -> bool:
    """Return whether enhanced liveness (CNN + depth + consistency) is enabled."""
    return getattr(settings, "RECOGNITION_ENHANCED_LIVENESS_ENABLED", False)


def get_cnn_antispoof_threshold() -> float:
    """Return confidence threshold for CNN anti-spoof classification."""
    return getattr(settings, "RECOGNITION_CNN_ANTISPOOF_THRESHOLD", 0.75)


def get_depth_variance_threshold() -> float:
    """Return threshold for depth variance check."""
    return getattr(settings, "RECOGNITION_DEPTH_VARIANCE_THRESHOLD", 0.1)


def get_frame_consistency_min_frames() -> int:
    """Return minimum frames for consistency verification."""
    return getattr(settings, "RECOGNITION_FRAME_CONSISTENCY_MIN_FRAMES", 5)

