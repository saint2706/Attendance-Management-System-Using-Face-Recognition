"""Tests for enhanced liveness detection."""

import os
import sys
from unittest.mock import MagicMock

# Mock cv2 before importing liveness
sys.modules.setdefault("cv2", MagicMock(name="cv2"))

import numpy as np
import pytest

from recognition.liveness import (
    ChallengeType,
    LivenessBuffer,
    LivenessCheckResult,
    check_liveness_with_challenge,
    is_live_face,
    run_multi_challenge_liveness,
)


class TestLivenessCheckResult:
    """Tests for the LivenessCheckResult named tuple."""

    def test_create_result(self):
        """Test creating a liveness check result."""
        result = LivenessCheckResult(
            passed=True,
            confidence=0.85,
            challenge_type=ChallengeType.MOTION,
            motion_score=1.5,
            frames_analyzed=5,
            threshold_used=1.1,
        )
        
        assert result.passed is True
        assert result.confidence == 0.85
        assert result.challenge_type == ChallengeType.MOTION
        assert result.motion_score == 1.5
        assert result.frames_analyzed == 5

    def test_result_with_defaults(self):
        """Test result with default values."""
        result = LivenessCheckResult(
            passed=False,
            confidence=0.3,
            challenge_type=ChallengeType.BLINK,
        )
        
        assert result.passed is False
        assert result.motion_score is None
        assert result.frames_analyzed == 0


class TestChallengeType:
    """Tests for ChallengeType enum."""

    def test_challenge_types(self):
        """Test all challenge types exist."""
        assert ChallengeType.MOTION.value == "motion"
        assert ChallengeType.BLINK.value == "blink"
        assert ChallengeType.HEAD_TURN.value == "head_turn"
        assert ChallengeType.ANTI_SPOOF.value == "anti_spoof"


class TestCheckLivenessWithChallenge:
    """Tests for the check_liveness_with_challenge function."""

    def test_insufficient_frames(self):
        """Test that insufficient frames returns a failed result."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]  # Only 1 frame
        
        result = check_liveness_with_challenge(
            frames,
            challenge_type=ChallengeType.MOTION,
        )
        
        assert result.passed is False
        assert result.confidence == 0.0
        assert "insufficient_frames" in result.details.get("error", "")

    def test_motion_challenge_with_movement(self):
        """Test motion challenge with moving frames."""
        frames = []
        for i in range(5):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            # Add varying intensity to simulate motion
            frame[:, :, 0] = i * 20  # Varying blue channel
            frames.append(frame)
        
        result = check_liveness_with_challenge(
            frames,
            challenge_type=ChallengeType.MOTION,
            motion_threshold=0.01,  # Low threshold for test
        )
        
        # Result depends on motion detection, but should not error
        assert isinstance(result.passed, bool)
        assert result.challenge_type == ChallengeType.MOTION
        assert result.frames_analyzed > 0

    def test_blink_challenge(self):
        """Test blink challenge returns a result."""
        frames = []
        for i in range(5):
            frame = np.ones((64, 64, 3), dtype=np.uint8) * 128
            # Simulate intensity variation for blink
            if i == 2:  # Middle frame has lower intensity (blink)
                frame[:32, :, :] = 64
            frames.append(frame)
        
        result = check_liveness_with_challenge(
            frames,
            challenge_type=ChallengeType.BLINK,
            blink_required=1,
        )
        
        assert isinstance(result.passed, bool)
        assert result.challenge_type == ChallengeType.BLINK
        assert "blink_count" in result.details

    def test_head_turn_challenge(self):
        """Test head turn challenge returns a result."""
        frames = []
        for i in range(5):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            # Create horizontal gradient shift
            for x in range(64):
                frame[:, x, 0] = min(255, (x + i * 5) % 256)
            frames.append(frame)
        
        result = check_liveness_with_challenge(
            frames,
            challenge_type=ChallengeType.HEAD_TURN,
            head_turn_threshold=0.001,
        )
        
        assert isinstance(result.passed, bool)
        assert result.challenge_type == ChallengeType.HEAD_TURN
        assert "left_motion" in result.details
        assert "right_motion" in result.details


class TestRunMultiChallengeLiveness:
    """Tests for the run_multi_challenge_liveness function."""

    def test_single_challenge(self):
        """Test running a single challenge."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
        
        passed, confidence, results = run_multi_challenge_liveness(
            frames,
            challenges=[ChallengeType.MOTION],
        )
        
        assert isinstance(passed, bool)
        assert 0 <= confidence <= 1
        assert len(results) == 1
        assert results[0].challenge_type == ChallengeType.MOTION

    def test_multiple_challenges(self):
        """Test running multiple challenges."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
        
        passed, confidence, results = run_multi_challenge_liveness(
            frames,
            challenges=[ChallengeType.MOTION, ChallengeType.BLINK],
        )
        
        assert isinstance(passed, bool)
        assert len(results) == 2

    def test_require_all_challenges(self):
        """Test requiring all challenges to pass."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
        
        passed, confidence, results = run_multi_challenge_liveness(
            frames,
            challenges=[ChallengeType.MOTION, ChallengeType.BLINK],
            require_all=True,
        )
        
        # With static frames, likely to fail
        assert isinstance(passed, bool)
        assert len(results) == 2

    def test_default_challenges(self):
        """Test using default challenges (MOTION only)."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
        
        passed, confidence, results = run_multi_challenge_liveness(frames)
        
        assert len(results) == 1
        assert results[0].challenge_type == ChallengeType.MOTION


class TestLivenessBuffer:
    """Tests for the LivenessBuffer class."""

    def test_buffer_max_length(self):
        """Test that buffer respects max length."""
        buffer = LivenessBuffer(maxlen=3)
        
        for i in range(5):
            buffer.append(np.zeros((10, 10, 3), dtype=np.uint8))
        
        assert len(buffer) == 3

    def test_buffer_snapshot(self):
        """Test getting a snapshot of buffer contents."""
        buffer = LivenessBuffer(maxlen=5)
        
        for i in range(3):
            frame = np.ones((10, 10, 3), dtype=np.uint8) * i
            buffer.append(frame)
        
        snapshot = buffer.snapshot()
        assert len(snapshot) == 3

    def test_buffer_clear(self):
        """Test clearing the buffer."""
        buffer = LivenessBuffer(maxlen=5)
        buffer.append(np.zeros((10, 10, 3), dtype=np.uint8))
        buffer.clear()
        
        assert len(buffer) == 0

    def test_buffer_rejects_invalid_frames(self):
        """Test that invalid frames are not added."""
        buffer = LivenessBuffer(maxlen=5)
        
        buffer.append(None)
        buffer.append("not an array")
        buffer.append(np.array([]))  # Empty array
        
        assert len(buffer) == 0
