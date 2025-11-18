import numpy as np

from recognition.liveness import LivenessBuffer, is_live_face


def _generate_frame(offset: int = 0) -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    start = 10 + offset
    end = start + 12
    frame[start:end, 20:32] = 255
    return frame


def test_liveness_buffer_truncates_old_frames() -> None:
    buffer = LivenessBuffer(maxlen=3)
    for idx in range(5):
        buffer.append(_generate_frame(idx))

    assert len(buffer.snapshot()) == 3


def test_is_live_face_detects_motion() -> None:
    frames = [_generate_frame(offset) for offset in range(4)]
    score = is_live_face(frames, min_frames=3, return_score=True)

    assert score is not None and score > 0.0

    decision = is_live_face(frames, min_frames=3, motion_threshold=float(score) * 0.5)
    assert decision is True


def test_is_live_face_rejects_static_sequence() -> None:
    static_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frames = [static_frame.copy() for _ in range(4)]

    score = is_live_face(frames, min_frames=3, return_score=True)
    assert score is not None and score <= 0.001

    decision = is_live_face(frames, min_frames=3, motion_threshold=0.05)
    assert decision is False


def test_is_live_face_requires_minimum_frames() -> None:
    frames = [_generate_frame(0)]
    assert is_live_face(frames, min_frames=3) is None
