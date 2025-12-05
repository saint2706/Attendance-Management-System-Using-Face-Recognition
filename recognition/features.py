"""
Feature flags for optional advanced functionality.

This module provides a centralized configuration system for enabling/disabling
advanced features to reduce complexity and maintenance burden for typical deployments.
"""

import os
from enum import Enum
from typing import Dict


class FeatureProfile(str, Enum):
    """Predefined feature profiles for common deployment scenarios."""

    BASIC = "basic"  # Minimal features for simple deployments
    STANDARD = "standard"  # Recommended for most production use
    ADVANCED = "advanced"  # Full feature set for enterprise deployments


class FeatureFlags:
    """
    Centralized feature flag management.

    Features can be controlled via:
    1. FEATURE_PROFILE environment variable (basic, standard, advanced)
    2. Individual ENABLE_* environment variables (override profile defaults)
    """

    # Feature profile from environment (defaults to advanced for backward compatibility)
    _profile: FeatureProfile = FeatureProfile.ADVANCED

    # Individual feature flags
    _liveness_detection: bool = True
    _deepface_antispoofing: bool = True
    _scheduled_evaluations: bool = True
    _fairness_audits: bool = True
    _liveness_evaluations: bool = True
    _performance_profiling: bool = True
    _encryption: bool = True
    _sentry: bool = True

    @classmethod
    def _initialize(cls) -> None:
        """Initialize feature flags from environment variables."""
        # Get profile from environment
        profile_name = os.environ.get("FEATURE_PROFILE", "").lower().strip()
        if profile_name:
            try:
                cls._profile = FeatureProfile(profile_name)
            except ValueError:
                # Invalid profile name, default to advanced
                cls._profile = FeatureProfile.ADVANCED

        # Apply profile defaults
        if cls._profile == FeatureProfile.BASIC:
            cls._liveness_detection = False
            cls._deepface_antispoofing = False
            cls._scheduled_evaluations = False
            cls._fairness_audits = False
            cls._liveness_evaluations = False
            cls._performance_profiling = False
            cls._encryption = False  # Will warn in production
            cls._sentry = False
        elif cls._profile == FeatureProfile.STANDARD:
            cls._liveness_detection = True
            cls._deepface_antispoofing = False  # Motion-based liveness is sufficient
            cls._scheduled_evaluations = False  # Manual evaluations as needed
            cls._fairness_audits = False  # Manual audits
            cls._liveness_evaluations = False
            cls._performance_profiling = False
            cls._encryption = True
            cls._sentry = True
        else:  # ADVANCED (default)
            cls._liveness_detection = True
            cls._deepface_antispoofing = True
            cls._scheduled_evaluations = True
            cls._fairness_audits = True
            cls._liveness_evaluations = True
            cls._performance_profiling = True
            cls._encryption = True
            cls._sentry = True

        # Override with individual environment variables
        cls._liveness_detection = cls._get_bool_env(
            "ENABLE_LIVENESS_DETECTION", cls._liveness_detection
        )
        cls._deepface_antispoofing = cls._get_bool_env(
            "ENABLE_DEEPFACE_ANTISPOOFING", cls._deepface_antispoofing
        )
        cls._scheduled_evaluations = cls._get_bool_env(
            "ENABLE_SCHEDULED_EVALUATIONS", cls._scheduled_evaluations
        )
        cls._fairness_audits = cls._get_bool_env(
            "ENABLE_FAIRNESS_AUDITS", cls._fairness_audits
        )
        cls._liveness_evaluations = cls._get_bool_env(
            "ENABLE_LIVENESS_EVALUATIONS", cls._liveness_evaluations
        )
        cls._performance_profiling = cls._get_bool_env(
            "ENABLE_PERFORMANCE_PROFILING", cls._performance_profiling
        )
        cls._encryption = cls._get_bool_env("ENABLE_ENCRYPTION", cls._encryption)
        cls._sentry = cls._get_bool_env("ENABLE_SENTRY", cls._sentry)

    @staticmethod
    def _get_bool_env(var_name: str, default: bool) -> bool:
        """Parse a boolean from an environment variable."""
        raw_value = os.environ.get(var_name)
        if raw_value is None:
            return default
        return raw_value.lower() in {"1", "true", "yes", "on"}

    @classmethod
    def get_profile(cls) -> FeatureProfile:
        """Return the active feature profile."""
        return cls._profile

    @classmethod
    def is_liveness_detection_enabled(cls) -> bool:
        """Check if motion-based liveness detection is enabled."""
        return cls._liveness_detection

    @classmethod
    def is_deepface_antispoofing_enabled(cls) -> bool:
        """Check if DeepFace's built-in anti-spoofing is enabled."""
        return cls._deepface_antispoofing

    @classmethod
    def is_scheduled_evaluations_enabled(cls) -> bool:
        """Check if automated nightly model evaluations are enabled."""
        return cls._scheduled_evaluations

    @classmethod
    def is_fairness_audits_enabled(cls) -> bool:
        """Check if scheduled fairness audits are enabled."""
        return cls._fairness_audits

    @classmethod
    def is_liveness_evaluations_enabled(cls) -> bool:
        """Check if liveness detection evaluations are enabled."""
        return cls._liveness_evaluations

    @classmethod
    def is_performance_profiling_enabled(cls) -> bool:
        """Check if Silk performance profiling middleware is enabled."""
        return cls._performance_profiling

    @classmethod
    def is_encryption_enabled(cls) -> bool:
        """Check if face data encryption at rest is enabled."""
        return cls._encryption

    @classmethod
    def is_sentry_enabled(cls) -> bool:
        """Check if Sentry error tracking is enabled."""
        return cls._sentry

    @classmethod
    def get_all_flags(cls) -> Dict[str, bool]:
        """Return a dictionary of all feature flags and their status."""
        return {
            "liveness_detection": cls._liveness_detection,
            "deepface_antispoofing": cls._deepface_antispoofing,
            "scheduled_evaluations": cls._scheduled_evaluations,
            "fairness_audits": cls._fairness_audits,
            "liveness_evaluations": cls._liveness_evaluations,
            "performance_profiling": cls._performance_profiling,
            "encryption": cls._encryption,
            "sentry": cls._sentry,
        }


# Initialize feature flags on module import
FeatureFlags._initialize()
