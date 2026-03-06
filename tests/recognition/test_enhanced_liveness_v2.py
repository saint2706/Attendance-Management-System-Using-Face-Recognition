"""Tests for enhanced liveness detection modules.

Tests cover:
- Depth cue estimation and analysis
- Frame consistency and replay detection
- CNN anti-spoof (with mocked model)
- Enhanced liveness verification orchestrator
"""

import numpy as np
import pytest


class TestDepthEstimator:
    """Tests for the depth_estimator module."""

    def test_estimate_pseudo_depth_returns_array(self):
        """Test that pseudo depth estimation returns an array."""
        from recognition.depth_estimator import estimate_pseudo_depth

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = estimate_pseudo_depth(frame)

        assert isinstance(result, np.ndarray)

    def test_estimate_pseudo_depth_handles_empty_frame(self):
        """Test that pseudo depth handles empty frame gracefully."""
        from recognition.depth_estimator import estimate_pseudo_depth

        result = estimate_pseudo_depth(np.array([]))

        assert result.size == 0

    def test_compute_depth_variance_uniform_image(self):
        """Test that uniform images have low depth variance."""
        from recognition.depth_estimator import compute_depth_variance

        # Uniform depth map should have zero variance
        uniform_map = np.ones((64, 64), dtype=np.float32) * 100
        variance = compute_depth_variance(uniform_map)

        assert variance == 0.0

    def test_compute_depth_variance_varied_image(self):
        """Test that varied images have higher depth variance."""
        from recognition.depth_estimator import compute_depth_variance

        # Create depth map with variation
        varied_map = np.random.rand(64, 64).astype(np.float32) * 100
        variance = compute_depth_variance(varied_map)

        assert variance > 0.0

    def test_analyze_depth_cues_insufficient_frames(self):
        """Test that insufficient frames return failed result."""
        from recognition.depth_estimator import analyze_depth_cues

        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]  # Only 1 frame

        result = analyze_depth_cues(frames, min_frames=3)

        assert result.passed is False
        assert result.confidence == 0.0
        assert result.frames_analyzed <= 1

    def test_analyze_depth_cues_returns_result(self):
        """Test that depth analysis returns a proper result."""
        from recognition.depth_estimator import DepthAnalysisResult, analyze_depth_cues

        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]

        result = analyze_depth_cues(frames)

        assert isinstance(result, DepthAnalysisResult)
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert result.frames_analyzed >= 0


class TestFrameConsistency:
    """Tests for the frame_consistency module."""

    def test_compute_frame_hash_returns_string(self):
        """Test that frame hash returns a hex string."""
        from recognition.frame_consistency import compute_frame_hash

        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        hash_result = compute_frame_hash(frame)

        assert isinstance(hash_result, str)
        assert len(hash_result) > 0

    def test_compute_frame_hash_empty_frame(self):
        """Test that empty frame returns empty string."""
        from recognition.frame_consistency import compute_frame_hash

        result = compute_frame_hash(np.array([]))

        assert result == ""

    def test_detect_static_replay_identical_frames(self):
        """Test detection of identical (static) frames."""
        from recognition.frame_consistency import detect_static_replay

        # Create identical frames
        base_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        base_frame[20:40, 20:40] = 255
        frames = [base_frame.copy() for _ in range(10)]

        is_static, unique_ratio = detect_static_replay(frames)

        assert is_static is True
        assert unique_ratio < 0.3

    def test_detect_static_replay_varied_frames(self):
        """Test that varied frames are not detected as static."""
        from recognition.frame_consistency import detect_static_replay

        # Create distinct frames
        frames = []
        for i in range(10):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[i * 5 : i * 5 + 20, 20:40] = 255  # Moving rectangle
            frames.append(frame)

        is_static, unique_ratio = detect_static_replay(frames)

        assert is_static is False
        assert unique_ratio > 0.3

    def test_detect_periodic_patterns_uniform_diffs(self):
        """Test periodic pattern detection with uniform differences."""
        from recognition.frame_consistency import detect_periodic_patterns

        # Create frames with nearly identical differences
        frames = [np.ones((64, 64, 3), dtype=np.uint8) * (i * 5 % 256) for i in range(10)]

        periodicity = detect_periodic_patterns(frames)

        assert isinstance(periodicity, float)
        assert 0.0 <= periodicity <= 1.0

    def test_check_frame_consistency_insufficient_frames(self):
        """Test that insufficient frames return failed result."""
        from recognition.frame_consistency import check_frame_consistency

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]

        result = check_frame_consistency(frames, min_frames=5)

        assert result.passed is False
        assert result.frames_analyzed < 5

    def test_check_frame_consistency_returns_result(self):
        """Test that consistency check returns proper result."""
        from recognition.frame_consistency import FrameConsistencyResult, check_frame_consistency

        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]

        result = check_frame_consistency(frames)

        assert isinstance(result, FrameConsistencyResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.is_static_replay, bool)
        assert 0.0 <= result.periodicity_score <= 1.0


class TestAntiSpoofCNN:
    """Tests for the anti_spoof_cnn module (with mocked TensorFlow)."""

    def test_antispoof_result_dataclass(self):
        """Test AntiSpoofResult dataclass creation."""
        from recognition.anti_spoof_cnn import AntiSpoofResult

        result = AntiSpoofResult(
            is_real=True,
            confidence=0.85,
            spoof_probability=0.15,
            model_available=True,
        )

        assert result.is_real is True
        assert result.confidence == 0.85
        assert result.spoof_probability == 0.15
        assert result.model_available is True


class TestEnhancedLivenessVerification:
    """Tests for the enhanced liveness verification in liveness.py."""

    def test_challenge_type_new_values(self):
        """Test that new challenge types are available."""
        from recognition.liveness import ChallengeType

        assert hasattr(ChallengeType, "DEPTH_CUE")
        assert hasattr(ChallengeType, "CNN_ANTI_SPOOF")
        assert hasattr(ChallengeType, "FRAME_CONSISTENCY")

        assert ChallengeType.DEPTH_CUE.value == "depth_cue"
        assert ChallengeType.CNN_ANTI_SPOOF.value == "cnn_anti_spoof"
        assert ChallengeType.FRAME_CONSISTENCY.value == "frame_consistency"

    def test_enhanced_liveness_result_dataclass(self):
        """Test EnhancedLivenessResult dataclass creation."""
        from recognition.liveness import EnhancedLivenessResult

        result = EnhancedLivenessResult(
            passed=True,
            confidence=0.9,
            motion_passed=True,
            depth_passed=True,
            cnn_passed=True,
            consistency_passed=True,
        )

        assert result.passed is True
        assert result.confidence == 0.9
        assert result.failure_reasons == []

    def test_run_enhanced_liveness_insufficient_frames(self):
        """Test that insufficient frames return failed result."""
        from recognition.liveness import run_enhanced_liveness_verification

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]

        result = run_enhanced_liveness_verification(frames)

        assert result.passed is False
        assert "Insufficient frames" in result.failure_reasons[0]

    def test_run_enhanced_liveness_with_valid_frames(self):
        """Test enhanced liveness with valid frame count."""
        from recognition.liveness import EnhancedLivenessResult, run_enhanced_liveness_verification

        # Create frames with some variation
        frames = []
        for i in range(6):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[i * 5 : i * 5 + 20, 20:40] = 255
            frames.append(frame)

        result = run_enhanced_liveness_verification(frames)

        assert isinstance(result, EnhancedLivenessResult)
        assert isinstance(result.passed, bool)
        assert "motion" in result.details
        assert "depth" in result.details
        assert "consistency" in result.details


# Django-dependent tests - require pytest-django plugin
@pytest.mark.django_db
class TestFeatureFlags:
    """Tests for enhanced liveness feature flag.

    These tests require Django to be set up via pytest-django.
    """

    def test_enhanced_liveness_flag_exists(self):
        """Test that enhanced liveness flag is accessible."""
        from recognition.features import FeatureFlags

        assert hasattr(FeatureFlags, "is_enhanced_liveness_enabled")
        result = FeatureFlags.is_enhanced_liveness_enabled()
        assert isinstance(result, bool)

    def test_get_all_flags_includes_enhanced(self):
        """Test that get_all_flags includes enhanced_liveness."""
        from recognition.features import FeatureFlags

        flags = FeatureFlags.get_all_flags()
        assert "enhanced_liveness" in flags


@pytest.mark.django_db
class TestConfigGetters:
    """Tests for config getter functions.

    These tests require Django to be set up via pytest-django.
    """

    def test_is_enhanced_liveness_enabled_getter(self):
        """Test enhanced liveness config getter."""
        from recognition.views.config import is_enhanced_liveness_enabled

        result = is_enhanced_liveness_enabled()
        assert isinstance(result, bool)

    def test_get_cnn_antispoof_threshold(self):
        """Test CNN threshold config getter."""
        from recognition.views.config import get_cnn_antispoof_threshold

        result = get_cnn_antispoof_threshold()
        assert isinstance(result, float)
        assert result == 0.75  # Default value

    def test_get_depth_variance_threshold(self):
        """Test depth variance threshold config getter."""
        from recognition.views.config import get_depth_variance_threshold

        result = get_depth_variance_threshold()
        assert isinstance(result, float)
        assert result == 0.1  # Default value

    def test_get_frame_consistency_min_frames(self):
        """Test frame consistency min frames config getter."""
        from recognition.views.config import get_frame_consistency_min_frames

        result = get_frame_consistency_min_frames()
        assert isinstance(result, int)
        assert result == 5  # Default value
