"""Thread-safe shared webcam manager used across attendance routines."""

from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np
from imutils.video import VideoStream

from . import monitoring

logger = logging.getLogger(__name__)


class _FrameConsumer:
    """Context manager handed out by :class:`WebcamManager` to read frames."""

    def __init__(self, manager: "WebcamManager") -> None:
        self._manager = manager
        self._last_frame_id = -1
        self._active = False

    def __enter__(self) -> "_FrameConsumer":  # pragma: no cover - trivial
        self._manager._register_consumer()
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.close()

    def close(self) -> None:
        if self._active:
            self._manager._release_consumer()
            self._active = False

    def read(self, timeout: Optional[float] = 1.0) -> Optional[np.ndarray]:
        """Return the next frame or ``None`` if no frame is available."""

        if not self._active:
            return None

        frame, frame_id = self._manager._wait_for_frame(self._last_frame_id, timeout)
        if frame is not None:
            self._last_frame_id = frame_id
        return frame


class WebcamManager:
    """Shared webcam manager to avoid reinitialising the camera per request."""

    def __init__(self, src: int = 0, warmup_time: float = 2.0) -> None:
        self._src = src
        self._warmup_time = max(0.0, warmup_time)
        self._stream: Optional[VideoStream] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_lock = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = 0
        self._consumer_lock = threading.Lock()
        self._consumer_count = 0

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._running:
            return

        start_time = time.perf_counter()
        try:
            self._stream = VideoStream(src=self._src).start()
            if self._warmup_time:
                time.sleep(self._warmup_time)

            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
        except Exception as exc:
            latency = time.perf_counter() - start_time
            monitoring.record_camera_start(False, latency, error=str(exc))
            logger.exception(
                "Webcam manager start failed", extra={"event": "webcam_start", "status": "failure"}
            )
            raise
        else:
            latency = time.perf_counter() - start_time
            monitoring.record_camera_start(True, latency)

    def shutdown(self) -> None:
        if not self._running:
            return

        self._running = False
        stop_start = time.perf_counter()
        with self._frame_lock:
            self._frame_lock.notify_all()

        join_timed_out = False
        shutdown_success = True
        shutdown_error: Optional[str] = None
        shutdown_exception: Optional[Exception] = None
        if self._thread:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                join_timed_out = True
            self._thread = None

        if self._stream:
            try:
                self._stream.stop()
            except Exception as exc:
                shutdown_success = False
                shutdown_error = str(exc)
                shutdown_exception = exc
                logger.exception(
                    "Webcam manager stop failed",
                    extra={"event": "webcam_stop", "status": "failure"},
                )
            finally:
                self._stream = None

        self._latest_frame = None
        self._latest_frame_id = 0

        latency = time.perf_counter() - stop_start
        monitoring.record_camera_stop(
            shutdown_success,
            latency,
            error=shutdown_error,
            timed_out=join_timed_out,
        )
        if shutdown_exception is not None:
            raise shutdown_exception

    # -- consumer helpers ---------------------------------------------

    def frame_consumer(self) -> _FrameConsumer:
        self.start()
        return _FrameConsumer(self)

    def _register_consumer(self) -> None:
        with self._consumer_lock:
            self._consumer_count += 1
            monitoring.update_consumer_count(self._consumer_count)

    def _release_consumer(self) -> None:
        with self._consumer_lock:
            self._consumer_count = max(0, self._consumer_count - 1)
            monitoring.update_consumer_count(self._consumer_count)

    # -- frame handling ------------------------------------------------

    def _capture_loop(self) -> None:
        assert self._stream is not None  # pragma: no cover - defensive

        last_capture_monotonic = None
        while self._running and self._stream is not None:
            frame = self._stream.read()
            if frame is None:
                monitoring.record_frame_drop()
                time.sleep(0.01)
                continue

            capture_monotonic = time.monotonic()
            if last_capture_monotonic is None:
                frame_delay = 0.0
            else:
                frame_delay = capture_monotonic - last_capture_monotonic
            last_capture_monotonic = capture_monotonic
            monitoring.record_frame_delay(frame_delay, capture_time=time.time())

            with self._frame_lock:
                self._latest_frame = frame.copy()
                self._latest_frame_id += 1
                self._frame_lock.notify_all()

    def _wait_for_frame(
        self, after_frame_id: int, timeout: Optional[float]
    ) -> Tuple[Optional[np.ndarray], int]:
        end_time = None if timeout is None else time.time() + max(timeout, 0.0)

        with self._frame_lock:
            while self._running and self._latest_frame_id <= after_frame_id:
                if timeout is None:
                    self._frame_lock.wait()
                    continue

                remaining = end_time - time.time()
                if remaining <= 0:
                    break
                self._frame_lock.wait(timeout=remaining)

            if self._latest_frame is None or self._latest_frame_id <= after_frame_id:
                return None, after_frame_id

            frame_copy = self._latest_frame.copy()
            return frame_copy, self._latest_frame_id


_manager_lock = threading.Lock()
_manager_instance: Optional[WebcamManager] = None


def get_webcam_manager() -> WebcamManager:
    """Return the shared :class:`WebcamManager` instance."""

    global _manager_instance
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = WebcamManager()
    return _manager_instance


def reset_webcam_manager() -> None:
    """Shutdown the shared manager and remove the singleton reference."""

    global _manager_instance
    if _manager_instance is not None:
        _manager_instance.shutdown()
    _manager_instance = None


atexit.register(reset_webcam_manager)
