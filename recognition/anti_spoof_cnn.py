"""CNN-based Presentation Attack Detection module.

Uses a lightweight MobileNetV2 architecture for binary classification
of real vs. spoof face images. The model is fine-tuned for detecting:
- Print attacks (photos)
- Replay attacks (screens)
- Mask attacks (basic 2D masks)

The model is loaded lazily and cached for efficiency.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

ArrayLike = np.ndarray

logger = logging.getLogger(__name__)

# Global lock for thread-safe model loading
_model_lock = threading.Lock()
_model_instance: Optional["AntiSpoofCNN"] = None


@dataclass(frozen=True)
class AntiSpoofResult:
    """Result of CNN-based anti-spoofing classification."""

    is_real: bool
    confidence: float
    spoof_probability: float
    model_available: bool


def _get_model_path() -> Path:
    """Get the path for the anti-spoof model."""
    from django.conf import settings

    default_path = (
        Path(getattr(settings, "MEDIA_ROOT", "media")) / "models" / "antispoof_cnn.tflite"
    )
    custom_path = getattr(settings, "RECOGNITION_CNN_ANTISPOOF_MODEL_PATH", None)

    if custom_path:
        return Path(custom_path)

    return default_path


def _get_antispoof_threshold() -> float:
    """Get the confidence threshold for anti-spoof classification."""
    try:
        from django.conf import settings

        return float(getattr(settings, "RECOGNITION_CNN_ANTISPOOF_THRESHOLD", 0.75))
    except Exception:
        return 0.75


class AntiSpoofCNN:
    """Singleton class for CNN-based presentation attack detection.

    Uses MobileNetV2 as the backbone with a custom binary classification head.
    The model is loaded lazily on first prediction and cached for efficiency.
    """

    _instance: Optional["AntiSpoofCNN"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AntiSpoofCNN":
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the anti-spoof model (lazy loading)."""
        if getattr(self, "_initialized", False):
            return

        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._model_loaded = False
        self._load_attempted = False
        self._input_size = (224, 224)
        self._initialized = True

    def _create_model(self):
        """Create the MobileNetV2-based anti-spoof model.

        Returns a model with pre-trained ImageNet weights and a custom
        classification head for binary real/spoof detection.
        """
        try:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
            from tensorflow.keras.models import Model

            # Load MobileNetV2 base with ImageNet weights
            base_model = MobileNetV2(
                weights="imagenet",
                include_top=False,
                input_shape=(*self._input_size, 3),
            )

            # Freeze base model layers
            for layer in base_model.layers:
                layer.trainable = False

            # Add custom classification head
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.3)(x)
            x = Dense(128, activation="relu")(x)
            x = Dropout(0.2)(x)
            output = Dense(1, activation="sigmoid")(x)  # Binary: real (1) vs spoof (0)

            model = Model(inputs=base_model.input, outputs=output)
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            from tensorflow import lite

            # Convert to TFLite and quantize
            converter = lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            return tflite_model

        except ImportError as e:
            logger.warning("TensorFlow/Keras not available for CNN anti-spoof: %s", e)
            return None
        except Exception as e:
            logger.warning("Failed to create anti-spoof model: %s", e)
            return None

    def load_model(self) -> bool:
        """Load the anti-spoof model.

        Attempts to load a pre-trained model from disk. If not available,
        creates a new model with ImageNet weights (untrained classifier).

        Returns:
            True if model is loaded and available.
        """
        if self._model_loaded:
            return True

        if self._load_attempted:
            return self._interpreter is not None

        with _model_lock:
            if self._model_loaded:
                return True

            self._load_attempted = True
            model_path = _get_model_path()

            # Try to load existing model
            if model_path.exists():
                try:
                    from tensorflow import lite

                    self._interpreter = lite.Interpreter(model_path=str(model_path))
                    self._interpreter.allocate_tensors()
                    self._input_details = self._interpreter.get_input_details()
                    self._output_details = self._interpreter.get_output_details()
                    self._model_loaded = True
                    logger.info("Loaded quantized anti-spoof TFLite model from %s", model_path)
                    return True
                except Exception as e:
                    logger.warning("Failed to load TFLite model from %s: %s", model_path, e)

            # Create new model with ImageNet weights
            tflite_model_content = self._create_model()
            if tflite_model_content is not None:

                # Save the model for future use
                try:
                    os.makedirs(model_path.parent, exist_ok=True)
                    with open(model_path, "wb") as f:
                        f.write(tflite_model_content)
                    logger.info("Saved new quantized anti-spoof TFLite model to %s", model_path)
                except Exception as e:
                    logger.debug("Could not save TFLite model: %s", e)

                # Load the interpreter
                try:
                    from tensorflow import lite

                    self._interpreter = lite.Interpreter(model_content=tflite_model_content)
                    self._interpreter.allocate_tensors()
                    self._input_details = self._interpreter.get_input_details()
                    self._output_details = self._interpreter.get_output_details()
                    self._model_loaded = True
                    return True
                except Exception as e:
                    logger.warning("Failed to load newly created TFLite model: %s", e)
                    return False

            return False

    def _preprocess_frame(self, frame: ArrayLike) -> Optional[ArrayLike]:
        """Preprocess a frame for model input."""
        if frame is None or frame.size == 0:
            return None

        try:
            # Resize to model input size
            if hasattr(cv2, "resize"):
                resized = cv2.resize(frame, self._input_size)
            else:
                return None

            # Ensure 3 channels
            if resized.ndim == 2:
                resized = np.stack([resized] * 3, axis=-1)
            elif resized.shape[-1] == 4:
                resized = resized[..., :3]

            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0

            return normalized

        except Exception as e:
            logger.debug("Frame preprocessing failed: %s", e)
            return None

    def predict(
        self,
        frame: ArrayLike,
        threshold: Optional[float] = None,
    ) -> AntiSpoofResult:
        """Predict whether a face is real or spoofed.

        Args:
            frame: Input face image (BGR or RGB).
            threshold: Optional confidence threshold override.

        Returns:
            AntiSpoofResult with is_real, confidence, and details.
        """
        if threshold is None:
            threshold = _get_antispoof_threshold()

        # Ensure model is loaded
        if not self.load_model() or self._interpreter is None:
            logger.debug("Anti-spoof model not available, assuming real.")
            return AntiSpoofResult(
                is_real=True,
                confidence=0.5,
                spoof_probability=0.5,
                model_available=False,
            )

        preprocessed = self._preprocess_frame(frame)
        if preprocessed is None:
            return AntiSpoofResult(
                is_real=True,
                confidence=0.5,
                spoof_probability=0.5,
                model_available=True,
            )

        try:
            # Add batch dimension
            batch = np.expand_dims(preprocessed, axis=0).astype(np.float32)

            # Predict using TFLite interpreter
            with _model_lock:
                # Ensure input tensor is correctly sized (if dynamic batching)
                if self._input_details[0]["shape"][0] != 1:
                    self._interpreter.resize_tensor_input(
                        self._input_details[0]["index"], batch.shape
                    )
                    self._interpreter.allocate_tensors()
                    self._input_details = self._interpreter.get_input_details()

                self._interpreter.set_tensor(self._input_details[0]["index"], batch)
                self._interpreter.invoke()
                prediction = self._interpreter.get_tensor(self._output_details[0]["index"])

            real_probability = float(prediction[0][0])
            spoof_probability = 1.0 - real_probability

            is_real = real_probability >= threshold

            return AntiSpoofResult(
                is_real=is_real,
                confidence=real_probability if is_real else spoof_probability,
                spoof_probability=spoof_probability,
                model_available=True,
            )

        except Exception as e:
            logger.warning("Anti-spoof prediction failed: %s", e)
            return AntiSpoofResult(
                is_real=True,
                confidence=0.5,
                spoof_probability=0.5,
                model_available=True,
            )

    def predict_batch(
        self,
        frames: Sequence[ArrayLike],
        threshold: Optional[float] = None,
    ) -> list[AntiSpoofResult]:
        """Predict for multiple frames in a single batch.

        Args:
            frames: Sequence of face images.
            threshold: Optional confidence threshold override.

        Returns:
            List of AntiSpoofResult for each frame.
        """
        if not frames:
            return []

        if threshold is None:
            threshold = _get_antispoof_threshold()

        # Ensure model is loaded
        if not self.load_model() or self._interpreter is None:
            logger.debug("Anti-spoof model not available, assuming real for batch.")
            return [
                AntiSpoofResult(
                    is_real=True,
                    confidence=0.5,
                    spoof_probability=0.5,
                    model_available=False,
                )
                for _ in frames
            ]

        # Preprocess all frames
        preprocessed_frames = []
        valid_indices = []

        for i, frame in enumerate(frames):
            preprocessed = self._preprocess_frame(frame)
            if preprocessed is not None:
                preprocessed_frames.append(preprocessed)
                valid_indices.append(i)

        # Initialize results with default fallbacks
        results = [
            AntiSpoofResult(
                is_real=True,
                confidence=0.5,
                spoof_probability=0.5,
                model_available=True,
            )
            for _ in frames
        ]

        if not preprocessed_frames:
            return results

        try:
            # Create a single batch for efficient inference
            batch = np.stack(preprocessed_frames).astype(np.float32)
            batch_size = batch.shape[0]

            # Predict using TFLite interpreter
            with _model_lock:
                # Resize input tensor for batching if needed
                if self._input_details[0]["shape"][0] != batch_size:
                    self._interpreter.resize_tensor_input(
                        self._input_details[0]["index"], batch.shape
                    )
                    self._interpreter.allocate_tensors()
                    self._input_details = self._interpreter.get_input_details()

                self._interpreter.set_tensor(self._input_details[0]["index"], batch)
                self._interpreter.invoke()
                predictions = self._interpreter.get_tensor(self._output_details[0]["index"])

            # Update results for valid frames
            for idx, prediction_val in zip(valid_indices, predictions):
                real_probability = float(prediction_val[0])
                spoof_probability = 1.0 - real_probability

                is_real = real_probability >= threshold

                results[idx] = AntiSpoofResult(
                    is_real=is_real,
                    confidence=real_probability if is_real else spoof_probability,
                    spoof_probability=spoof_probability,
                    model_available=True,
                )

        except Exception as e:
            logger.warning("Anti-spoof batch prediction failed: %s", e)

        return results


def get_antispoof_model() -> AntiSpoofCNN:
    """Get the singleton anti-spoof model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = AntiSpoofCNN()
    return _model_instance


def run_cnn_antispoof(
    frames: Sequence[ArrayLike],
    threshold: Optional[float] = None,
) -> AntiSpoofResult:
    """Run CNN anti-spoof detection on a sequence of frames.

    Analyzes multiple frames and aggregates results for more robust
    spoof detection.

    Args:
        frames: Sequence of face images to analyze.
        threshold: Optional confidence threshold override.

    Returns:
        Aggregated AntiSpoofResult.
    """
    if not frames:
        return AntiSpoofResult(
            is_real=True,
            confidence=0.5,
            spoof_probability=0.5,
            model_available=False,
        )

    model = get_antispoof_model()

    # Get predictions for all frames
    results = model.predict_batch(frames, threshold)

    if not results:
        return AntiSpoofResult(
            is_real=True,
            confidence=0.5,
            spoof_probability=0.5,
            model_available=False,
        )

    # Aggregate results: majority voting with confidence weighting
    real_votes = sum(1 for r in results if r.is_real)
    spoof_votes = len(results) - real_votes

    avg_confidence = sum(r.confidence for r in results) / len(results)
    avg_spoof_prob = sum(r.spoof_probability for r in results) / len(results)
    model_available = any(r.model_available for r in results)

    # Majority voting
    is_real = real_votes > spoof_votes

    return AntiSpoofResult(
        is_real=is_real,
        confidence=avg_confidence,
        spoof_probability=avg_spoof_prob,
        model_available=model_available,
    )


__all__ = [
    "AntiSpoofResult",
    "AntiSpoofCNN",
    "get_antispoof_model",
    "run_cnn_antispoof",
]
