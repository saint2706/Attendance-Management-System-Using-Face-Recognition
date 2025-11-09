"""Tests for the shared webcam manager lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

import numpy as np

from recognition.webcam_manager import get_webcam_manager, reset_webcam_manager


class WebcamManagerLifecycleTests(SimpleTestCase):
    """Ensure the shared webcam manager releases resources cleanly."""

    def tearDown(self) -> None:
        reset_webcam_manager()
        return super().tearDown()

    @patch("recognition.webcam_manager.VideoStream")
    def test_shutdown_stops_underlying_stream(self, mock_videostream):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)

        stream = MagicMock()
        stream.start.return_value = stream
        stream.read.side_effect = lambda: frame
        mock_videostream.return_value = stream

        manager = get_webcam_manager()
        with manager.frame_consumer() as consumer:
            self.assertIsNotNone(consumer.read(timeout=0.1))

        manager.shutdown()
        stream.stop.assert_called_once()

    @patch("recognition.webcam_manager.VideoStream")
    def test_reset_disposes_existing_instance(self, mock_videostream):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)

        stream_one = MagicMock()
        stream_one.start.return_value = stream_one
        stream_one.read.side_effect = lambda: frame

        stream_two = MagicMock()
        stream_two.start.return_value = stream_two
        stream_two.read.side_effect = lambda: frame

        mock_videostream.side_effect = [stream_one, stream_two]

        manager_first = get_webcam_manager()
        with manager_first.frame_consumer() as consumer:
            self.assertIsNotNone(consumer.read(timeout=0.1))

        reset_webcam_manager()
        stream_one.stop.assert_called_once()

        manager_second = get_webcam_manager()
        self.assertIsNot(manager_first, manager_second)
        with manager_second.frame_consumer() as consumer:
            self.assertIsNotNone(consumer.read(timeout=0.1))
