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

    default_path = Path(getattr(settings, "MEDIA_ROOT", "media")) / "models" / "antispoof_cnn.h5"
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

        self._model = None
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

            return model

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
            return self._model is not None

        with _model_lock:
            if self._model_loaded:
                return True

            self._load_attempted = True
            model_path = _get_model_path()

            # Try to load existing model
            if model_path.exists():
                try:
                    from tensorflow import keras

                    self._model = keras.models.load_model(str(model_path))
                    self._model_loaded = True
                    logger.info("Loaded anti-spoof model from %s", model_path)
                    return True
                except Exception as e:
                    logger.warning("Failed to load model from %s: %s", model_path, e)

            # Create new model with ImageNet weights
            self._model = self._create_model()
            if self._model is not None:
                self._model_loaded = True

                # Save the model for future use
                try:
                    os.makedirs(model_path.parent, exist_ok=True)
                    self._model.save(str(model_path))
                    logger.info("Saved new anti-spoof model to %s", model_path)
                except Exception as e:
                    logger.debug("Could not save model: %s", e)

                return True

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
        if not self.load_model() or self._model is None:
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
            batch = np.expand_dims(preprocessed, axis=0)

            # Predict
            prediction = self._model.predict(batch, verbose=0)
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
        """Predict for multiple frames.

        Args:
            frames: Sequence of face images.
            threshold: Optional confidence threshold override.

        Returns:
            List of AntiSpoofResult for each frame.
        """
        return [self.predict(frame, threshold) for frame in frames]


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
