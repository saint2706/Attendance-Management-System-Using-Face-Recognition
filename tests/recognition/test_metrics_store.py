from recognition.metrics_store import _normalize_confidence


def test_normalize_confidence_none_distance():
    assert _normalize_confidence(None, 0.5) is None


def test_normalize_confidence_none_threshold():
    assert _normalize_confidence(0.2, None) is None


def test_normalize_confidence_zero_threshold():
    assert _normalize_confidence(0.2, 0.0) is None


def test_normalize_confidence_valid():
    assert _normalize_confidence(0.2, 0.4) == 0.5
    assert _normalize_confidence(0.5, 0.4) == 0.0
    assert _normalize_confidence(0.0, 0.4) == 1.0
