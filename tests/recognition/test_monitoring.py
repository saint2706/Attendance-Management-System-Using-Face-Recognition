"""Tests for the monitoring instrumentation utilities."""

from __future__ import annotations

import time

from django.test import TestCase, override_settings

from recognition import monitoring


class MonitoringInstrumentationTests(TestCase):
    """Ensure monitoring helpers capture health signals as expected."""

    def setUp(self) -> None:
        monitoring.reset_for_tests()
        return super().setUp()

    def test_camera_start_and_stop_state(self) -> None:
        monitoring.record_camera_start(success=True, latency=0.25)
        snapshot = monitoring.get_health_snapshot()
        self.assertTrue(snapshot["camera"]["running"])
        self.assertEqual(snapshot["camera"]["last_start"]["status"], "success")
        self.assertEqual(snapshot["metrics"]["camera_start"]["success"], 1.0)

        monitoring.record_camera_stop(success=True, latency=0.15)
        snapshot = monitoring.get_health_snapshot()
        self.assertFalse(snapshot["camera"]["running"])
        self.assertEqual(snapshot["metrics"]["camera_stop"]["success"], 1.0)

    @override_settings(RECOGNITION_FRAME_DELAY_ALERT_SECONDS=0.01)
    def test_frame_delay_alert_when_threshold_exceeded(self) -> None:
        monitoring.reset_for_tests()
        monitoring.record_frame_delay(0.05, capture_time=time.time())
        snapshot = monitoring.get_health_snapshot()
        alert_types = {alert["type"] for alert in snapshot["alerts"]}
        self.assertIn("frame_delay", alert_types)

    @override_settings(RECOGNITION_WARMUP_ALERT_SECONDS=0.01)
    def test_stage_duration_alert_when_warmup_slow(self) -> None:
        monitoring.reset_for_tests()
        monitoring.observe_stage_duration(
            "deepface_warmup",
            0.05,
            threshold_key="warmup",
        )
        snapshot = monitoring.get_health_snapshot()
        stages = [alert["data"].get("stage") for alert in snapshot["alerts"]]
        self.assertIn("deepface_warmup", stages)
