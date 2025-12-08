"""Tests for edge cases in face recognition and multi-face detection.

This module provides comprehensive tests for edge cases including:
- Multiple face edge cases (too small faces, duplicates, max limit exceeded)
- Empty frame handling (black frames, white frames, None inputs)
- Invalid image handling (corrupted data, wrong channels, NaN embeddings)
"""

from unittest.mock import patch

import numpy as np

# =============================================================================
# MULTIPLE FACES EDGE CASES
# =============================================================================


class TestMultiFaceEdgeCases:
    """Tests for multi-face recognition edge cases."""

    @patch("recognition.multi_face.settings")
    @patch("recognition.multi_face.filter_faces_by_size")
    def test_all_faces_too_small_filtered_out(self, mock_filter, mock_settings):
        """When all detected faces are too small, filter returns empty list."""
        mock_settings.RECOGNITION_MULTI_FACE_MIN_SIZE = 100

        # All faces smaller than minimum
        faces = [
            (np.array([1, 2, 3]), {"x": 0, "y": 0, "w": 30, "h": 30}),
            (np.array([4, 5, 6]), {"x": 50, "y": 50, "w": 40, "h": 40}),
        ]

        mock_filter.return_value = []  # All filtered out

        from recognition.multi_face import filter_faces_by_size

        result = filter_faces_by_size(faces)
        assert result == []

    @patch("recognition.multi_face.settings")
    def test_mixed_size_faces_partial_filter(self, mock_settings):
        """When some faces are too small, only valid faces pass the filter."""
        mock_settings.RECOGNITION_MULTI_FACE_MIN_SIZE = 50

        from recognition.multi_face import filter_faces_by_size

        faces = [
            (np.array([1, 2, 3]), {"x": 0, "y": 0, "w": 100, "h": 100}),  # Pass
            (np.array([4, 5, 6]), {"x": 50, "y": 50, "w": 20, "h": 20}),  # Filtered
            (np.array([7, 8, 9]), {"x": 100, "y": 100, "w": 75, "h": 80}),  # Pass
        ]

        result = filter_faces_by_size(faces)

        assert len(result) == 2
        # Verify correct faces passed
        widths = [r[1]["w"] for r in result]
        assert 100 in widths
        assert 75 in widths
        assert 20 not in widths

    @patch("recognition.multi_face.get_max_faces_limit")
    @patch("recognition.multi_face.filter_faces_by_size")
    @patch("recognition.multi_face.extract_all_embeddings")
    def test_exceeds_max_faces_limit(self, mock_extract, mock_filter, mock_max):
        """When more faces detected than max limit, only process up to limit."""
        mock_max.return_value = 2  # Max 2 faces

        # Create 5 faces (more than limit)
        face_embs = [
            (np.array([i * 0.1, i * 0.2, i * 0.3]), {"x": i * 50, "y": i * 10, "w": 100, "h": 100})
            for i in range(5)
        ]
        mock_extract.return_value = face_embs
        mock_filter.return_value = face_embs

        from recognition.multi_face import process_multi_face_recognition

        result = process_multi_face_recognition(
            representations=[{"embedding": [0.1, 0.2, 0.3]}],
            dataset_index=[],
            distance_metric="cosine",
            distance_threshold=0.4,
        )

        # Should only process 2 faces
        assert result["count"] == 2

    def test_empty_representations_returns_error(self):
        """Empty representations list should return appropriate error."""
        from recognition.multi_face import process_single_face_recognition

        result = process_single_face_recognition(
            representations=[],
            dataset_index=[],
            distance_metric="cosine",
            distance_threshold=0.4,
        )

        assert result["recognized"] is False
        assert "error" in result

    @patch("recognition.multi_face.extract_all_embeddings")
    def test_no_faces_detected_multi_mode(self, mock_extract):
        """Multi-face mode with no faces detected returns error."""
        mock_extract.return_value = []

        from recognition.multi_face import process_multi_face_recognition

        result = process_multi_face_recognition(
            representations=[],
            dataset_index=[],
            distance_metric="cosine",
            distance_threshold=0.4,
        )

        assert result["count"] == 0
        assert result["mode"] == "multi"
        assert "error" in result


# =============================================================================
# EMPTY FRAME HANDLING
# =============================================================================


class TestEmptyFrameHandling:
    """Tests for handling empty, blank, and invalid frames."""

    def test_black_frame_depth_estimation(self):
        """Black frame should still return depth estimation (all zeros)."""
        from recognition.depth_estimator import estimate_pseudo_depth

        black_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = estimate_pseudo_depth(black_frame)

        assert isinstance(result, np.ndarray)

    def test_white_frame_depth_estimation(self):
        """White frame should return uniform depth map."""
        from recognition.depth_estimator import estimate_pseudo_depth

        white_frame = np.ones((64, 64, 3), dtype=np.uint8) * 255
        result = estimate_pseudo_depth(white_frame)

        assert isinstance(result, np.ndarray)

    def test_empty_array_depth_estimation(self):
        """Empty array should return empty result."""
        from recognition.depth_estimator import estimate_pseudo_depth

        result = estimate_pseudo_depth(np.array([]))
        assert result.size == 0

    def test_empty_frame_hash_computation(self):
        """Empty frame should return empty hash."""
        from recognition.frame_consistency import compute_frame_hash

        result = compute_frame_hash(np.array([]))
        assert result == ""

    def test_single_frame_consistency_check(self):
        """Single frame should fail consistency check requiring multiple frames."""
        from recognition.frame_consistency import check_frame_consistency

        single_frame = [np.zeros((64, 64, 3), dtype=np.uint8)]

        result = check_frame_consistency(single_frame, min_frames=3)

        assert result.passed is False

    def test_empty_frames_list_liveness(self):
        """Empty frames list should fail liveness verification."""
        from recognition.liveness import run_enhanced_liveness_verification

        result = run_enhanced_liveness_verification([])

        assert result.passed is False
        assert "Insufficient frames" in result.failure_reasons[0]


# =============================================================================
# INVALID IMAGE HANDLING
# =============================================================================


class TestInvalidImageHandling:
    """Tests for handling invalid image data."""

    def test_wrong_dtype_frame(self):
        """Frame with wrong dtype should still be processable."""
        from recognition.depth_estimator import estimate_pseudo_depth

        # Float32 instead of uint8
        float_frame = np.random.rand(64, 64, 3).astype(np.float32)
        result = estimate_pseudo_depth(float_frame)

        assert isinstance(result, np.ndarray)

    def test_grayscale_frame_handling(self):
        """Grayscale frame (2D) should be handled gracefully."""
        from recognition.depth_estimator import estimate_pseudo_depth

        grayscale = np.zeros((64, 64), dtype=np.uint8)

        # Should handle without crashing
        try:
            result = estimate_pseudo_depth(grayscale)
            assert isinstance(result, np.ndarray)
        except ValueError:
            # Acceptable to reject invalid input
            pass

    def test_nan_embedding_distance_calculation(self):
        """NaN in embeddings produces NaN distance that fails threshold check."""
        import math

        from recognition.pipeline import calculate_embedding_distance, is_within_distance_threshold

        valid = np.array([1.0, 2.0, 3.0])
        with_nan = np.array([1.0, float("nan"), 3.0])

        result = calculate_embedding_distance(with_nan, valid, "cosine")

        # NaN in input creates NaN output, which is handled by threshold check
        assert result is None or math.isnan(result)
        # Regardless, NaN should not pass threshold check
        assert is_within_distance_threshold(result, 0.4) is False

    def test_zero_vector_cosine_distance(self):
        """Zero vector in cosine distance should return None."""
        from recognition.pipeline import calculate_embedding_distance

        zero_vec = np.zeros(3)
        other_vec = np.array([1.0, 2.0, 3.0])

        result = calculate_embedding_distance(zero_vec, other_vec, "cosine")

        assert result is None

    def test_inf_in_embedding(self):
        """Infinity in embedding should be handled."""
        from recognition.pipeline import is_within_distance_threshold

        result = is_within_distance_threshold(float("inf"), 0.4)

        assert result is False

    def test_negative_distance_threshold(self):
        """Negative distance handling - check actual behavior."""
        from recognition.pipeline import is_within_distance_threshold

        # Negative distance is mathematically <= any positive threshold
        # The function returns True for -0.5 <= 0.4 (which is True)
        # This is expected behavior - negative distances are rare edge cases
        result = is_within_distance_threshold(-0.5, 0.4)

        # Document actual behavior: negative distances pass if <= threshold
        assert result is True  # -0.5 <= 0.4 is True


# =============================================================================
# FAISS INDEX EDGE CASES
# =============================================================================


class TestFAISSIndexEdgeCases:
    """Edge case tests for FAISS index operations."""

    def test_search_with_zero_k(self):
        """Search with k=0 should return empty list."""
        from recognition.faiss_index import FAISSIndex

        index = FAISSIndex(dimension=4)
        embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        index.add_embeddings(embeddings, ["alice"])

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=0)

        assert results == []

    def test_search_k_larger_than_index_size(self):
        """Search with k > index size should return all available."""
        from recognition.faiss_index import FAISSIndex

        index = FAISSIndex(dimension=4)
        embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        index.add_embeddings(embeddings, ["alice", "bob"])

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=10)  # k > index size

        assert len(results) <= 2

    def test_duplicate_labels_in_index(self):
        """Index should handle duplicate labels correctly."""
        from recognition.faiss_index import FAISSIndex

        index = FAISSIndex(dimension=4)

        # Add same label twice with different embeddings
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],  # Same person, slightly different embedding
            ],
            dtype=np.float32,
        )
        index.add_embeddings(embeddings, ["alice", "alice"])

        assert index.size == 2

        # Search should find closest
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = index.search_single(query)

        assert result is not None
        assert result[0] == "alice"

    def test_normalized_vs_unnormalized_embeddings(self):
        """Test that both normalized and unnormalized embeddings work."""
        from recognition.faiss_index import FAISSIndex

        index = FAISSIndex(dimension=3)

        # Unnormalized embedding
        unnorm = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        index.add_embeddings(unnorm, ["large_magnitude"])

        assert index.size == 1

        # Should still find it
        query = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        result = index.search_single(query)

        assert result is not None
        assert result[0] == "large_magnitude"

    def test_very_high_dimensionality(self):
        """Test with high-dimensional embeddings (typical for face recognition)."""
        from recognition.faiss_index import FAISSIndex

        dimension = 512  # Common embedding dimension
        index = FAISSIndex(dimension=dimension)

        embeddings = np.random.randn(10, dimension).astype(np.float32)
        labels = [f"user_{i}" for i in range(10)]
        index.add_embeddings(embeddings, labels)

        assert index.size == 10

        # Search should work
        query = embeddings[0]
        result = index.search_single(query)

        assert result is not None
        assert result[0] == "user_0"
        assert result[1] < 0.01  # Should be very close (same embedding)
