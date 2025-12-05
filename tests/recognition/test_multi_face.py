"""Tests for multi-face detection functionality."""

from unittest.mock import patch

import numpy as np

from recognition.multi_face import (
    filter_faces_by_size,
    get_max_faces_limit,
    is_multi_face_enabled,
    process_face_recognition,
    process_multi_face_recognition,
    process_single_face_recognition,
)


class TestMultiFaceConfiguration:
    """Test configuration helpers."""

    @patch("recognition.multi_face.settings")
    def test_is_multi_face_enabled_default(self, mock_settings):
        """Test default multi-face mode is disabled."""
        mock_settings.RECOGNITION_MULTI_FACE_ENABLED = False
        assert is_multi_face_enabled() is False

    @patch("recognition.multi_face.settings")
    def test_is_multi_face_enabled_when_enabled(self, mock_settings):
        """Test multi-face mode when explicitly enabled."""
        mock_settings.RECOGNITION_MULTI_FACE_ENABLED = True
        assert is_multi_face_enabled() is True

    @patch("recognition.multi_face.settings")
    def test_get_max_faces_limit_default(self, mock_settings):
        """Test default max faces limit is 5."""
        mock_settings.RECOGNITION_MAX_FACES_PER_FRAME = 5
        assert get_max_faces_limit() == 5

    @patch("recognition.multi_face.settings")
    def test_get_max_faces_limit_custom(self, mock_settings):
        """Test custom max faces limit."""
        mock_settings.RECOGNITION_MAX_FACES_PER_FRAME = 3
        assert get_max_faces_limit() == 3


class TestFaceFiltering:
    """Test face size filtering."""

    @patch("recognition.multi_face.settings")
    def test_filter_faces_by_size_all_pass(self, mock_settings):
        """Test filtering when all faces meet minimum size."""
        mock_settings.RECOGNITION_MULTI_FACE_MIN_SIZE = 50

        faces = [
            (np.array([1, 2, 3]), {"x": 0, "y": 0, "w": 100, "h": 100}),
            (np.array([4, 5, 6]), {"x": 200, "y": 50, "w": 80, "h": 90}),
        ]

        filtered = filter_faces_by_size(faces)
        assert len(filtered) == 2

    @patch("recognition.multi_face.settings")
    def test_filter_faces_by_size_some_filtered(self, mock_settings):
        """Test filtering removes faces too small."""
        mock_settings.RECOGNITION_MULTI_FACE_MIN_SIZE = 50

        faces = [
            (np.array([1, 2, 3]), {"x": 0, "y": 0, "w": 100, "h": 100}),  # Pass
            (np.array([4, 5, 6]), {"x": 200, "y": 50, "w": 30, "h": 30}),  # Filtered
            (np.array([7, 8, 9]), {"x": 300, "y": 60, "w": 60, "h": 70}),  # Pass
        ]

        filtered = filter_faces_by_size(faces)
        assert len(filtered) == 2
        # Check that correct faces passed
        assert filtered[0][1]["w"] == 100
        assert filtered[1][1]["w"] == 60

    @patch("recognition.multi_face.settings")
    def test_filter_faces_by_size_no_area_info(self, mock_settings):
        """Test faces without area info are kept."""
        mock_settings.RECOGNITION_MULTI_FACE_MIN_SIZE = 50

        faces = [(np.array([1, 2, 3]), None)]

        filtered = filter_faces_by_size(faces)
        assert len(filtered) == 1


class TestSingleFaceRecognition:
    """Test single-face recognition mode."""

    def test_single_face_no_embedding(self):
        """Test single-face mode when no embedding extracted."""
        result = process_single_face_recognition(
            representations=[],  # Empty representations
            dataset_index=[],
            distance_metric="cosine",
            distance_threshold=0.4,
        )

        assert result["recognized"] is False
        assert result["mode"] == "single"
        assert "error" in result

    @patch("recognition.multi_face.extract_embedding")
    @patch("recognition.multi_face.find_closest_dataset_match")
    def test_single_face_with_match(self, mock_find_match, mock_extract):
        """Test single-face mode with successful match."""
        # Mock embedding extraction
        mock_extract.return_value = (
            np.array([0.1, 0.2, 0.3]),
            {"x": 100, "y": 50, "w": 150, "h": 150},
        )

        # Mock match finding
        mock_find_match.return_value = ("john_doe", 0.25, "/path/to/identity")

        dataset = [{"embedding": np.array([0.1, 0.2, 0.3]), "username": "john_doe"}]

        result = process_single_face_recognition(
            representations=[{"embedding": [0.1, 0.2, 0.3]}],
            dataset_index=dataset,
            distance_metric="cosine",
            distance_threshold=0.4,
        )

        assert result["recognized"] is True
        assert result["username"] == "john_doe"
        assert result["distance"] == 0.25
        assert result["mode"] == "single"


class TestMultiFaceRecognition:
    """Test multi-face recognition mode."""

    @patch("recognition.multi_face.extract_all_embeddings")
    def test_multi_face_no_faces_detected(self, mock_extract_all):
        """Test multi-face mode when no faces detected."""
        mock_extract_all.return_value = []

        result = process_multi_face_recognition(
            representations=[],
            dataset_index=[],
            distance_metric="cosine",
            distance_threshold=0.4,
        )

        assert result["count"] == 0
        assert result["mode"] == "multi"
        assert "error" in result

    @patch("recognition.multi_face.get_max_faces_limit")
    @patch("recognition.multi_face.filter_faces_by_size")
    @patch("recognition.multi_face.extract_all_embeddings")
    @patch("recognition.multi_face.find_closest_dataset_match")
    def test_multi_face_multiple_matches(
        self, mock_find_match, mock_extract_all, mock_filter, mock_max_faces
    ):
        """Test multi-face mode with multiple successful matches."""
        # Mock settings
        mock_max_faces.return_value = 5

        # Mock embedding extraction (2 faces)
        face_embeddings = [
            (np.array([0.1, 0.2, 0.3]), {"x": 100, "y": 50, "w": 150, "h": 150}),
            (np.array([0.4, 0.5, 0.6]), {"x": 300, "y": 60, "w": 140, "h": 140}),
        ]
        mock_extract_all.return_value = face_embeddings
        mock_filter.return_value = face_embeddings  # All pass filtering

        # Mock match finding (both match)
        mock_find_match.side_effect = [
            ("john_doe", 0.25, "/path/to/john"),
            ("jane_smith", 0.30, "/path/to/jane"),
        ]

        dataset = []  # Not used due to mocking

        result = process_multi_face_recognition(
            representations=[{"embedding": [0.1, 0.2, 0.3]}],  # Mocked anyway
            dataset_index=dataset,
            distance_metric="cosine",
            distance_threshold=0.4,
        )

        assert result["count"] == 2
        assert result["mode"] == "multi"
        assert len(result["faces"]) == 2

        # Check first face
        assert result["faces"][0]["recognized"] is True
        assert result["faces"][0]["match"]["username"] == "john_doe"
        assert result["faces"][0]["match"]["distance"] == 0.25

        # Check second face
        assert result["faces"][1]["recognized"] is True
        assert result["faces"][1]["match"]["username"] == "jane_smith"
        assert result["faces"][1]["match"]["distance"] == 0.30

    @patch("recognition.multi_face.get_max_faces_limit")
    @patch("recognition.multi_face.filter_faces_by_size")
    @patch("recognition.multi_face.extract_all_embeddings")
    def test_multi_face_respects_max_limit(self, mock_extract_all, mock_filter, mock_max_faces):
        """Test that max_faces limit is enforced."""
        mock_max_faces.return_value = 2  # Limit to 2 faces

        # Mock 3 faces extracted
        face_embeddings = [
            (np.array([0.1, 0.2, 0.3]), {"x": 100, "y": 50, "w": 150, "h": 150}),
            (np.array([0.4, 0.5, 0.6]), {"x": 300, "y": 60, "w": 140, "h": 140}),
            (np.array([0.7, 0.8, 0.9]), {"x": 500, "y": 70, "w": 130, "h": 130}),
        ]
        mock_extract_all.return_value = face_embeddings
        mock_filter.return_value = face_embeddings

        result = process_multi_face_recognition(
            representations=[],
            dataset_index=[],
            distance_metric="cosine",
            distance_threshold=0.4,
        )

        # Should only process first 2 faces
        assert result["count"] == 2


class TestAutomaticModeSelection:
    """Test automatic single vs multi mode selection."""

    @patch("recognition.multi_face.is_multi_face_enabled")
    @patch("recognition.multi_face.process_single_face_recognition")
    def test_auto_mode_selects_single(self, mock_single, mock_enabled):
        """Test automatic mode selection chooses single-face mode."""
        mock_enabled.return_value = False
        mock_single.return_value = {"mode": "single", "recognized": True}

        result = process_face_recognition(
            representations=[],
            dataset_index=[],
            distance_metric="cosine",
            distance_threshold=0.4,
        )

        assert result["mode"] == "single"
        mock_single.assert_called_once()

    @patch("recognition.multi_face.is_multi_face_enabled")
    @patch("recognition.multi_face.process_multi_face_recognition")
    def test_auto_mode_selects_multi(self, mock_multi, mock_enabled):
        """Test automatic mode selection chooses multi-face mode."""
        mock_enabled.return_value = True
        mock_multi.return_value = {"mode": "multi", "count": 2}

        result = process_face_recognition(
            representations=[],
            dataset_index=[],
            distance_metric="cosine",
            distance_threshold=0.4,
        )

        assert result["mode"] == "multi"
        mock_multi.assert_called_once()
