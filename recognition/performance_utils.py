"""Performance utilities for hardware-aware face recognition optimization.

This module provides utilities for detecting available hardware accelerators (NPU, GPU, CPU),
recommending optimal models and backends, and profiling recognition performance.

Hardware Detection Priority:
1. NPU (Neural Processing Unit) - Intel OpenVINO or Windows DirectML
2. GPU - NVIDIA/AMD via TensorFlow/CUDA
3. CPU - Optimized with lightweight models

Example usage:
    >>> from recognition.performance_utils import detect_hardware, get_recommended_config
    >>> 
    >>> hardware = detect_hardware()
    >>> print(hardware)
    >>> # {'cpu': True, 'gpu': {...}, 'npu': {...}}
    >>> 
    >>> config = get_recommended_config(hardware)
    >>> print(config['backend'])
    >>> # 'openvino' or 'tensorflow' or 'cpu'
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class HardwareInfo:
    """Information about available hardware accelerators."""

    cpu_available: bool = True
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None
    npu_available: bool = False
    npu_type: Optional[str] = None  # 'intel' or 'directml'
    npu_device: Optional[str] = None
    npu_backend: Optional[str] = None  # 'openvino' or 'directml'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for logging and API responses."""
        return {
            "cpu": self.cpu_available,
            "gpu": {
                "available": self.gpu_available,
                "name": self.gpu_name,
                "memory_mb": self.gpu_memory_mb,
            },
            "npu": {
                "available": self.npu_available,
                "type": self.npu_type,
                "device": self.npu_device,
                "backend": self.npu_backend,
            },
        }


def detect_npu_availability() -> Dict[str, Any]:
    """Detect if NPU (Neural Processing Unit) is available.

    Checks for:
    1. Intel NPU via OpenVINO Python API
    2. Windows NPU via DirectML (Windows 11 with Intel Core Ultra)

    Returns:
        Dictionary with NPU availability info:
        {
            'available': bool,
            'type': 'intel' | 'directml' | None,
            'device': str | None,
            'backend': 'openvino' | 'directml' | None
        }
    """
    npu_info: Dict[str, Any] = {
        "available": False,
        "type": None,
        "device": None,
        "backend": None,
    }

    # Try OpenVINO NPU detection
    try:
        from openvino.runtime import Core

        core = Core()
        available_devices = core.available_devices

        # Check for NPU devices (Intel NPU typically shows as 'NPU' or 'NPU.0')
        npu_devices = [d for d in available_devices if d.startswith("NPU")]

        if npu_devices:
            npu_info["available"] = True
            npu_info["type"] = "intel"
            npu_info["device"] = npu_devices[0]
            npu_info["backend"] = "openvino"
            logger.info(f"Intel NPU detected via OpenVINO: {npu_devices[0]}")
            return npu_info

    except ImportError:
        logger.debug("OpenVINO not installed, skipping Intel NPU detection")
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.debug(f"OpenVINO NPU detection failed: {exc}")

    # Try DirectML NPU detection (Windows only)
    try:
        import platform

        if platform.system() == "Windows":
            # DirectML detection would require tensorflow-directml or ONNX Runtime with DirectML EP
            # For now, we'll check if it's available in the environment
            try:
                import tensorflow as tf

                # Check if DirectML plugin is available
                # Note: This is a placeholder - actual DirectML NPU detection
                # would require specific DirectML APIs
                available_devices = tf.config.list_physical_devices()
                directml_devices = [d for d in available_devices if "DirectML" in str(d)]

                if directml_devices:
                    npu_info["available"] = True
                    npu_info["type"] = "directml"
                    npu_info["device"] = str(directml_devices[0])
                    npu_info["backend"] = "directml"
                    logger.info(f"DirectML NPU detected: {directml_devices[0]}")
                    return npu_info

            except (ImportError, AttributeError):
                pass

    except Exception as exc:  # pragma: no cover - defensive programming
        logger.debug(f"DirectML NPU detection failed: {exc}")

    logger.debug("No NPU detected")
    return npu_info


def detect_gpu_availability() -> Dict[str, Any]:
    """Detect if GPU is available for TensorFlow acceleration.

    Returns:
        Dictionary with GPU availability info:
        {
            'available': bool,
            'name': str | None,
            'memory_mb': int | None
        }
    """
    gpu_info: Dict[str, Any] = {
        "available": False,
        "name": None,
        "memory_mb": None,
    }

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            gpu_info["available"] = True
            # Get name and memory of first GPU
            try:
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                gpu_info["name"] = gpu_details.get("device_name", "Unknown GPU")
            except Exception:  # pragma: no cover - may not be available on all TF versions
                gpu_info["name"] = str(gpus[0].name)

            # Try to get memory info
            try:
                memory_info = tf.config.experimental.get_memory_info(gpus[0])
                gpu_info["memory_mb"] = memory_info.get("current", 0) // (1024 * 1024)
            except Exception:  # pragma: no cover - may not be available
                pass

            logger.info(f"GPU detected: {gpu_info['name']}")
            return gpu_info

    except ImportError:
        logger.debug("TensorFlow not installed, skipping GPU detection")
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.debug(f"GPU detection failed: {exc}")

    logger.debug("No GPU detected")
    return gpu_info


def detect_openvino_devices() -> list[str]:
    """List all available OpenVINO compute devices (CPU, GPU, NPU).

    Returns:
        List of device names (e.g., ['CPU', 'GPU.0', 'NPU.0'])
    """
    try:
        from openvino.runtime import Core

        core = Core()
        devices = core.available_devices
        logger.debug(f"OpenVINO devices available: {devices}")
        return list(devices)

    except ImportError:
        logger.debug("OpenVINO not installed")
        return []
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.debug(f"Failed to enumerate OpenVINO devices: {exc}")
        return []


def detect_hardware() -> HardwareInfo:
    """Detect all available hardware accelerators with priority: NPU → GPU → CPU.

    Returns:
        HardwareInfo dataclass with availability details for each accelerator type.
    """
    logger.info("Detecting available hardware accelerators...")

    # NPU detection (highest priority)
    npu_info = detect_npu_availability()

    # GPU detection (second priority)
    gpu_info = detect_gpu_availability()

    # CPU is always available
    hardware = HardwareInfo(
        cpu_available=True,
        gpu_available=gpu_info["available"],
        gpu_name=gpu_info["name"],
        gpu_memory_mb=gpu_info["memory_mb"],
        npu_available=npu_info["available"],
        npu_type=npu_info["type"],
        npu_device=npu_info["device"],
        npu_backend=npu_info["backend"],
    )

    # Log hardware summary
    if hardware.npu_available:
        logger.info(
            f"Hardware detected: NPU ({hardware.npu_type} via {hardware.npu_backend})"
        )
    elif hardware.gpu_available:
        logger.info(f"Hardware detected: GPU ({hardware.gpu_name})")
    else:
        logger.info("Hardware detected: CPU only (no accelerators found)")

    return hardware


def get_recommended_backend(hardware: HardwareInfo) -> str:
    """Recommend the best inference backend based on available hardware.

    Priority: OpenVINO (NPU/GPU) → TensorFlow (GPU) → CPU

    Args:
        hardware: HardwareInfo object from detect_hardware()

    Returns:
        Recommended backend: 'openvino', 'tensorflow', or 'cpu'
    """
    if hardware.npu_available and hardware.npu_backend == "openvino":
        return "openvino"

    if hardware.gpu_available:
        # Prefer TensorFlow for GPU if available
        return "tensorflow"

    return "cpu"


def get_recommended_model(hardware: HardwareInfo, *, prefer_accuracy: bool = True) -> str:
    """Recommend the best model based on hardware and accuracy preference.

    Args:
        hardware: HardwareInfo object from detect_hardware()
        prefer_accuracy: If True, prefer accuracy over speed. If False, prefer speed.

    Returns:
        Recommended model name: 'Facenet', 'VGG-Face', 'OpenFace', etc.
    """
    # NPU: Use Facenet (can be optimized with OpenVINO)
    if hardware.npu_available:
        return "Facenet"  # Or 'Facenet512' for higher accuracy

    # GPU: Use Facenet for accuracy or VGG-Face for balanced performance
    if hardware.gpu_available:
        return "Facenet" if prefer_accuracy else "VGG-Face"

    # CPU: Use OpenFace for speed or Facenet if accuracy is critical
    if prefer_accuracy:
        return "Facenet"  # User willing to wait for accuracy
    else:
        return "OpenFace"  # Fast and lightweight for CPU


def get_recommended_config(
    hardware: Optional[HardwareInfo] = None, prefer_accuracy: bool = True
) -> Dict[str, Any]:
    """Get complete recommended configuration for face recognition.

    Args:
        hardware: Optional HardwareInfo. If None, will detect automatically.
        prefer_accuracy: If True, prefer accuracy over speed.

    Returns:
        Configuration dictionary with recommended backend, model, and settings.
    """
    if hardware is None:
        hardware = detect_hardware()

    backend = get_recommended_backend(hardware)
    model = get_recommended_model(hardware, prefer_accuracy=prefer_accuracy)

    config = {
        "backend": backend,
        "model": model,
        "hardware": hardware.to_dict(),
        "detector_backend": "ssd",  # Fast and reliable
        "enforce_detection": False,  # More permissive for attendance use case
    }

    # Add backend-specific settings
    if backend == "openvino":
        config["openvino_device"] = hardware.npu_device or "NPU"
    elif backend == "tensorflow" and hardware.gpu_available:
        config["use_gpu"] = True

    return config


def log_recognition_timing(func: F) -> F:
    """Decorator to measure and log face recognition operation timing.

    Example:
        @log_recognition_timing
        def recognize_face(image):
            # ... recognition logic
            return result
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"Recognition operation '{func.__name__}' completed in {elapsed_ms:.2f}ms"
            )

            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                f"Recognition operation '{func.__name__}' failed after {elapsed_ms:.2f}ms: {exc}"
            )
            raise

    return wrapper  # type: ignore


def profile_model_performance(
    model_name: str,
    detector_backend: str = "ssd",
    num_iterations: int = 10,
) -> Dict[str, float]:
    """Benchmark a specific DeepFace model's performance.

    Args:
        model_name: DeepFace model name (e.g., 'Facenet', 'OpenFace')
        detector_backend: Face detector backend
        num_iterations: Number of test iterations for averaging

    Returns:
        Dictionary with performance metrics (mean, std, min, max inference time in ms)
    """
    try:
        from deepface import DeepFace
    except ImportError:
        logger.error("DeepFace not installed, cannot profile models")
        return {}

    # Create a synthetic test image (224x224 RGB)
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    timings = []

    logger.info(f"Profiling model '{model_name}' with {num_iterations} iterations...")

    for i in range(num_iterations):
        start_time = time.perf_counter()

        try:
            DeepFace.represent(
                img_path=test_image,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            timings.append(elapsed_ms)

        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Iteration {i + 1} failed: {exc}")
            continue

    if not timings:
        logger.error(f"All profiling iterations failed for model '{model_name}'")
        return {}

    metrics = {
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "iterations": len(timings),
    }

    logger.info(
        f"Model '{model_name}' performance: "
        f"mean={metrics['mean_ms']:.2f}ms, "
        f"std={metrics['std_ms']:.2f}ms"
    )

    return metrics


__all__ = [
    "HardwareInfo",
    "detect_hardware",
    "detect_npu_availability",
    "detect_gpu_availability",
    "detect_openvino_devices",
    "get_recommended_backend",
    "get_recommended_model",
    "get_recommended_config",
    "log_recognition_timing",
    "profile_model_performance",
]
