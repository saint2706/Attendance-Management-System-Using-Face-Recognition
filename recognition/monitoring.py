"""Monitoring utilities for webcam and recognition health."""

from __future__ import annotations

import datetime as _dt
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from django.conf import settings

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)


@dataclass
class _HealthState:
    """Mutable snapshot of the latest monitoring information."""

    running: bool = False
    last_start: Optional[Dict[str, Any]] = None
    last_stop: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    last_frame_timestamp: Optional[float] = None
    last_frame_delay: Optional[float] = None
    consumer_count: int = 0
    stage_durations: Dict[str, float] = field(default_factory=dict)


_STATE = _HealthState()
_STATE_LOCK = threading.Lock()
_ALERTS: deque[Dict[str, Any]] = deque()

_THRESHOLD_SETTING_NAMES: Dict[str, str] = {
    "camera_start": "RECOGNITION_CAMERA_START_ALERT_SECONDS",
    "frame_delay": "RECOGNITION_FRAME_DELAY_ALERT_SECONDS",
    "model_load": "RECOGNITION_MODEL_LOAD_ALERT_SECONDS",
    "warmup": "RECOGNITION_WARMUP_ALERT_SECONDS",
    "recognition_iteration": "RECOGNITION_LOOP_ALERT_SECONDS",
}

_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "camera_start": 3.0,
    "frame_delay": 0.75,
    "model_load": 4.0,
    "warmup": 3.0,
    "recognition_iteration": 1.5,
}


def _max_alert_history() -> int:
    value = getattr(settings, "RECOGNITION_HEALTH_ALERT_HISTORY", 50)
    try:
        numeric = int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        numeric = 50
    return max(1, numeric)


def _now_timestamp() -> float:
    return time.time()


def _format_timestamp(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()


def _create_event(status: str, latency: Optional[float], error: Optional[str]) -> Dict[str, Any]:
    return {
        "timestamp": _now_timestamp(),
        "status": status,
        "latency": latency,
        "error": error,
    }


def _append_alert(
    event_type: str, severity: str, message: str, data: Optional[Dict[str, Any]] = None
) -> None:
    payload = {
        "timestamp": _format_timestamp(_now_timestamp()),
        "type": event_type,
        "severity": severity,
        "message": message,
        "data": data or {},
    }
    with _STATE_LOCK:
        _ALERTS.append(payload)
        max_alerts = _max_alert_history()
        while len(_ALERTS) > max_alerts:
            _ALERTS.popleft()


def _build_metrics() -> None:
    global REGISTRY
    global CAMERA_START_COUNTER
    global CAMERA_START_LATENCY
    global CAMERA_STOP_COUNTER
    global CAMERA_STOP_LATENCY
    global CAMERA_RUNNING_GAUGE
    global ACTIVE_CONSUMERS_GAUGE
    global FRAME_DELAY_HISTOGRAM
    global FRAME_DROP_COUNTER
    global LAST_FRAME_TIMESTAMP_GAUGE
    global STAGE_DURATION_HISTOGRAM
    global WARMUP_ALERT_COUNTER

    REGISTRY = CollectorRegistry(auto_describe=True)

    CAMERA_START_COUNTER = Counter(
        "webcam_manager_start",
        "Total camera start attempts",
        labelnames=("status",),
        registry=REGISTRY,
    )
    CAMERA_START_LATENCY = Histogram(
        "webcam_manager_start_latency_seconds",
        "Camera start latency in seconds",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
        registry=REGISTRY,
    )
    CAMERA_STOP_COUNTER = Counter(
        "webcam_manager_stop",
        "Total camera shutdown attempts",
        labelnames=("status",),
        registry=REGISTRY,
    )
    CAMERA_STOP_LATENCY = Histogram(
        "webcam_manager_stop_latency_seconds",
        "Camera shutdown latency in seconds",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        registry=REGISTRY,
    )
    CAMERA_RUNNING_GAUGE = Gauge(
        "webcam_manager_running",
        "1 when the webcam manager is actively capturing",
        registry=REGISTRY,
    )
    ACTIVE_CONSUMERS_GAUGE = Gauge(
        "webcam_active_consumers",
        "Number of active frame consumers",
        registry=REGISTRY,
    )
    FRAME_DELAY_HISTOGRAM = Histogram(
        "webcam_frame_delay_seconds",
        "Time between successive frames",
        buckets=(0.0, 0.016, 0.033, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        registry=REGISTRY,
    )
    FRAME_DROP_COUNTER = Counter(
        "webcam_frame_drop",
        "Count of times the webcam returned no frame",
        registry=REGISTRY,
    )
    LAST_FRAME_TIMESTAMP_GAUGE = Gauge(
        "webcam_last_frame_timestamp_seconds",
        "Unix timestamp of the most recent captured frame",
        registry=REGISTRY,
    )
    STAGE_DURATION_HISTOGRAM = Histogram(
        "recognition_stage_duration_seconds",
        "Duration of critical recognition stages",
        labelnames=("stage",),
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        registry=REGISTRY,
    )
    WARMUP_ALERT_COUNTER = Counter(
        "recognition_warmup_alerts",
        "Count of recognition warm-up stages exceeding thresholds",
        labelnames=("stage",),
        registry=REGISTRY,
    )


_build_metrics()


def reset_for_tests() -> None:
    """Reset in-memory state and metrics (intended for test suites)."""

    with _STATE_LOCK:
        global _STATE
        _STATE = _HealthState()
        _ALERTS.clear()
    _build_metrics()


def get_threshold(key: str) -> float:
    """Fetch the configured alert threshold for the supplied key."""

    if key not in _THRESHOLD_SETTING_NAMES:
        raise KeyError(f"Unknown threshold key: {key}")
    setting = _THRESHOLD_SETTING_NAMES[key]
    default = _DEFAULT_THRESHOLDS[key]
    value = getattr(settings, setting, default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        numeric = default
    return numeric


def get_alert_thresholds() -> Dict[str, float]:
    """Return the effective thresholds for all monitored stages."""

    return {key: get_threshold(key) for key in _THRESHOLD_SETTING_NAMES}


def _metric_value(name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
    labels = labels or {}
    # The Prometheus client automatically appends suffixes like _total for counters.
    sample = REGISTRY.get_sample_value(name, labels)
    if sample is None and name.endswith("_total"):
        # Some Prometheus helpers omit the suffix in get_sample_value lookups.
        sample = REGISTRY.get_sample_value(name.replace("_total", ""), labels)
    return sample


def record_camera_start(
    success: bool, latency: Optional[float], error: Optional[str] = None
) -> None:
    """Record metrics and internal state for a camera start attempt."""

    status = "success" if success else "failure"
    CAMERA_START_COUNTER.labels(status=status).inc()
    if latency is not None:
        CAMERA_START_LATENCY.observe(latency)
    with _STATE_LOCK:
        if success:
            _STATE.running = True
            _STATE.last_error = None
            CAMERA_RUNNING_GAUGE.set(1)
        else:
            CAMERA_RUNNING_GAUGE.set(0)
        _STATE.last_start = _create_event(status, latency, error)
    log_extra = {
        "event": "webcam_start",
        "status": status,
        "latency_seconds": latency,
    }
    if success:
        logger.info("Webcam manager started", extra=log_extra)
        threshold = get_threshold("camera_start")
        if latency is not None and latency > threshold:
            message = f"Webcam start latency {latency:.3f}s exceeded threshold {threshold:.3f}s"
            logger.warning(
                message, extra={**log_extra, "severity": "warning", "threshold": threshold}
            )
            _append_alert(
                "camera_start_latency",
                "warning",
                message,
                {"latency": latency, "threshold": threshold},
            )
    else:
        message = "Failed to start webcam manager"
        logger.error(message, extra={**log_extra, "error": error})
        _append_alert(
            "camera_start_failure",
            "error",
            message,
            {"error": error or "unknown", "latency": latency},
        )


def record_camera_stop(
    success: bool,
    latency: Optional[float],
    *,
    error: Optional[str] = None,
    timed_out: bool = False,
) -> None:
    """Record metrics for camera shutdown attempts and derive alerts."""

    status = "timeout" if timed_out and success else ("success" if success else "failure")
    CAMERA_STOP_COUNTER.labels(status=status).inc()
    if latency is not None:
        CAMERA_STOP_LATENCY.observe(latency)
    with _STATE_LOCK:
        _STATE.running = False
        CAMERA_RUNNING_GAUGE.set(0)
        if not success:
            _STATE.last_error = error
        _STATE.last_stop = _create_event(status, latency, error)
    log_extra = {
        "event": "webcam_stop",
        "status": status,
        "latency_seconds": latency,
    }
    if success and not timed_out:
        logger.info("Webcam manager stopped", extra=log_extra)
    elif success and timed_out:
        message = "Webcam manager shutdown timed out"
        logger.warning(message, extra=log_extra)
        _append_alert(
            "camera_stop_timeout",
            "warning",
            message,
            {"latency": latency},
        )
    else:
        message = "Webcam manager failed to stop cleanly"
        logger.error(message, extra={**log_extra, "error": error})
        _append_alert(
            "camera_stop_failure",
            "error",
            message,
            {"error": error or "unknown", "latency": latency},
        )


def update_consumer_count(count: int) -> None:
    """Persist the active consumer count in metrics and in-memory state."""

    ACTIVE_CONSUMERS_GAUGE.set(count)
    with _STATE_LOCK:
        _STATE.consumer_count = count


def record_frame_drop() -> None:
    """Increment drop counters when no frame is received."""

    FRAME_DROP_COUNTER.inc()
    logger.debug("Webcam returned no frame", extra={"event": "frame_drop"})


def record_frame_delay(delay: float, capture_time: Optional[float]) -> None:
    """Track inter-frame delay and log alerts when thresholds are exceeded."""

    FRAME_DELAY_HISTOGRAM.observe(max(0.0, delay))
    timestamp = capture_time or _now_timestamp()
    LAST_FRAME_TIMESTAMP_GAUGE.set(timestamp)
    with _STATE_LOCK:
        _STATE.last_frame_delay = delay
        _STATE.last_frame_timestamp = timestamp
    threshold = get_threshold("frame_delay")
    if delay > threshold:
        message = f"Frame delay {delay:.3f}s exceeded threshold {threshold:.3f}s"
        logger.warning(
            message,
            extra={
                "event": "frame_delay",
                "severity": "warning",
                "delay_seconds": delay,
                "threshold": threshold,
            },
        )
        _append_alert(
            "frame_delay",
            "warning",
            message,
            {"delay": delay, "threshold": threshold},
        )


def observe_stage_duration(
    stage: str, duration: float, *, threshold_key: Optional[str] = None
) -> None:
    """Record stage durations and emit alerts for slow executions."""

    STAGE_DURATION_HISTOGRAM.labels(stage=stage).observe(max(0.0, duration))
    with _STATE_LOCK:
        _STATE.stage_durations[stage] = duration
    if threshold_key:
        threshold = get_threshold(threshold_key)
        if duration > threshold:
            logger.warning(
                "Recognition stage '%s' exceeded threshold",
                stage,
                extra={
                    "event": "stage_duration",
                    "stage": stage,
                    "duration_seconds": duration,
                    "threshold": threshold,
                },
            )
            if threshold_key == "warmup":
                WARMUP_ALERT_COUNTER.labels(stage=stage).inc()
            _append_alert(
                "stage_duration",
                "warning",
                f"Stage '{stage}' duration {duration:.3f}s exceeded {threshold:.3f}s",
                {"stage": stage, "duration": duration, "threshold": threshold},
            )


def get_health_snapshot() -> Dict[str, Any]:
    """Return a serialisable snapshot of webcam health and alert history."""

    with _STATE_LOCK:
        last_start = _STATE.last_start.copy() if _STATE.last_start else None
        if last_start and "timestamp" in last_start:
            last_start["timestamp"] = _format_timestamp(last_start["timestamp"])
        last_stop = _STATE.last_stop.copy() if _STATE.last_stop else None
        if last_stop and "timestamp" in last_stop:
            last_stop["timestamp"] = _format_timestamp(last_stop["timestamp"])

        camera = {
            "running": _STATE.running,
            "consumers": _STATE.consumer_count,
            "last_start": last_start,
            "last_stop": last_stop,
            "last_error": _STATE.last_error,
        }
        frames = {
            "last_frame_timestamp": _format_timestamp(_STATE.last_frame_timestamp),
            "last_frame_delay": _STATE.last_frame_delay,
        }
        stages = {
            stage: {"last_duration": duration} for stage, duration in _STATE.stage_durations.items()
        }
        alerts = list(_ALERTS)
    metrics = {
        "camera_start": {
            "success": _metric_value("webcam_manager_start_total", {"status": "success"}) or 0,
            "failure": _metric_value("webcam_manager_start_total", {"status": "failure"}) or 0,
        },
        "camera_stop": {
            "success": _metric_value("webcam_manager_stop_total", {"status": "success"}) or 0,
            "failure": _metric_value("webcam_manager_stop_total", {"status": "failure"}) or 0,
            "timeout": _metric_value("webcam_manager_stop_total", {"status": "timeout"}) or 0,
        },
        "frame_drop_total": _metric_value("webcam_frame_drop_total") or 0,
    }
    return {
        "camera": camera,
        "frames": frames,
        "stages": stages,
        "alerts": alerts,
        "metrics": metrics,
        "thresholds": get_alert_thresholds(),
    }


def export_metrics() -> bytes:
    """Serialise the Prometheus metrics registry."""

    return generate_latest(REGISTRY)


def prometheus_content_type() -> str:
    """Expose the correct ``Content-Type`` for Prometheus responses."""

    return CONTENT_TYPE_LATEST


__all__ = [
    "export_metrics",
    "get_alert_thresholds",
    "get_health_snapshot",
    "get_threshold",
    "observe_stage_duration",
    "prometheus_content_type",
    "record_camera_start",
    "record_camera_stop",
    "record_frame_delay",
    "record_frame_drop",
    "reset_for_tests",
    "update_consumer_count",
]
