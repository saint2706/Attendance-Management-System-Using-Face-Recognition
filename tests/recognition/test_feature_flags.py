"""
Tests for the feature flags system.
"""

import os
from unittest import mock

from recognition.features import FeatureFlags, FeatureProfile


class TestFeatureFlags:
    """Test the FeatureFlags system."""

    def test_default_profile_is_advanced(self):
        """By default (no env vars), profile should be advanced for backward compatibility."""
        with mock.patch.dict(os.environ, {}, clear=True):
            FeatureFlags._initialize()
            assert FeatureFlags.get_profile() == FeatureProfile.ADVANCED
            assert FeatureFlags.is_liveness_detection_enabled() is True
            assert FeatureFlags.is_deepface_antispoofing_enabled() is True
            assert FeatureFlags.is_scheduled_evaluations_enabled() is True

    def test_basic_profile_disables_advanced_features(self):
        """Basic profile should disable all advanced features."""
        with mock.patch.dict(os.environ, {"FEATURE_PROFILE": "basic"}, clear=True):
            FeatureFlags._initialize()
            assert FeatureFlags.get_profile() == FeatureProfile.BASIC
            assert FeatureFlags.is_liveness_detection_enabled() is False
            assert FeatureFlags.is_deepface_antispoofing_enabled() is False
            assert FeatureFlags.is_scheduled_evaluations_enabled() is False
            assert FeatureFlags.is_fairness_audits_enabled() is False
            assert FeatureFlags.is_performance_profiling_enabled() is False

    def test_standard_profile_enables_core_security(self):
        """Standard profile should enable core security features but not scheduled tasks."""
        with mock.patch.dict(os.environ, {"FEATURE_PROFILE": "standard"}, clear=True):
            FeatureFlags._initialize()
            assert FeatureFlags.get_profile() == FeatureProfile.STANDARD
            assert FeatureFlags.is_liveness_detection_enabled() is True
            assert FeatureFlags.is_encryption_enabled() is True
            assert FeatureFlags.is_deepface_antispoofing_enabled() is False
            assert FeatureFlags.is_scheduled_evaluations_enabled() is False
            assert FeatureFlags.is_fairness_audits_enabled() is False

    def test_advanced_profile_enables_all_features(self):
        """Advanced profile should enable all features."""
        with mock.patch.dict(os.environ, {"FEATURE_PROFILE": "advanced"}, clear=True):
            FeatureFlags._initialize()
            assert FeatureFlags.get_profile() == FeatureProfile.ADVANCED
            assert FeatureFlags.is_liveness_detection_enabled() is True
            assert FeatureFlags.is_deepface_antispoofing_enabled() is True
            assert FeatureFlags.is_scheduled_evaluations_enabled() is True
            assert FeatureFlags.is_fairness_audits_enabled() is True
            assert FeatureFlags.is_liveness_evaluations_enabled() is True
            assert FeatureFlags.is_performance_profiling_enabled() is True
            assert FeatureFlags.is_encryption_enabled() is True

    def test_individual_flag_overrides_profile(self):
        """Individual feature flags should override profile defaults."""
        with mock.patch.dict(
            os.environ,
            {"FEATURE_PROFILE": "basic", "ENABLE_LIVENESS_DETECTION": "true"},
            clear=True,
        ):
            FeatureFlags._initialize()
            assert FeatureFlags.get_profile() == FeatureProfile.BASIC
            assert FeatureFlags.is_liveness_detection_enabled() is True  # Overridden
            assert FeatureFlags.is_deepface_antispoofing_enabled() is False  # Still from profile

    def test_bool_env_parsing(self):
        """Test various boolean environment variable formats."""
        test_cases = [
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("invalid", False),
        ]

        for value, expected in test_cases:
            with mock.patch.dict(
                os.environ, {"FEATURE_PROFILE": "basic", "ENABLE_ENCRYPTION": value}, clear=True
            ):
                FeatureFlags._initialize()
                assert FeatureFlags.is_encryption_enabled() == expected

    def test_invalid_profile_defaults_to_advanced(self):
        """Invalid profile names should fall back to advanced."""
        with mock.patch.dict(os.environ, {"FEATURE_PROFILE": "invalid_profile"}, clear=True):
            FeatureFlags._initialize()
            assert FeatureFlags.get_profile() == FeatureProfile.ADVANCED

    def test_get_all_flags_returns_dict(self):
        """get_all_flags should return a dictionary of all flags."""
        with mock.patch.dict(os.environ, {"FEATURE_PROFILE": "standard"}, clear=True):
            FeatureFlags._initialize()
            flags = FeatureFlags.get_all_flags()
            assert isinstance(flags, dict)
            assert "liveness_detection" in flags
            assert "deepface_antispoofing" in flags
            assert "scheduled_evaluations" in flags
            assert flags["liveness_detection"] is True
            assert flags["scheduled_evaluations"] is False

    def test_multiple_overrides(self):
        """Test multiple individual flag overrides."""
        with mock.patch.dict(
            os.environ,
            {
                "FEATURE_PROFILE": "basic",
                "ENABLE_LIVENESS_DETECTION": "true",
                "ENABLE_ENCRYPTION": "true",
                "ENABLE_SENTRY": "true",
            },
            clear=True,
        ):
            FeatureFlags._initialize()
            assert FeatureFlags.is_liveness_detection_enabled() is True
            assert FeatureFlags.is_encryption_enabled() is True
            assert FeatureFlags.is_sentry_enabled() is True
            # These should still be false from basic profile
            assert FeatureFlags.is_deepface_antispoofing_enabled() is False
            assert FeatureFlags.is_scheduled_evaluations_enabled() is False

    def test_case_insensitive_profile_name(self):
        """Profile names should be case-insensitive."""
        test_cases = ["BASIC", "Basic", "bAsIc", "basic"]
        for profile_value in test_cases:
            with mock.patch.dict(os.environ, {"FEATURE_PROFILE": profile_value}, clear=True):
                FeatureFlags._initialize()
                assert FeatureFlags.get_profile() == FeatureProfile.BASIC
