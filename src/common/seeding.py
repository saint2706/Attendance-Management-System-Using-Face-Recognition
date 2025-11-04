"""
Global seeding utilities for reproducibility.

This module provides functions to set random seeds across different libraries
(Python, NumPy, TensorFlow) to ensure reproducible results in training and evaluation.
"""

import os
import random
from typing import Optional


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow (if available).

    This function ensures reproducibility by setting the random seed for all
    major libraries used in the project. Call this at the start of training,
    evaluation, or any other random-dependent operations.

    Args:
        seed: Integer seed value for random number generators. Default is 42.

    Example:
        >>> set_global_seed(42)
        >>> # Now all random operations will be reproducible
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # TensorFlow and tf.keras
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Set environment variables for TensorFlow determinism
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    except ImportError:
        pass

    # PyTorch (if used in the future)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_random_state(seed: Optional[int] = None) -> int:
    """
    Get or generate a random state for use in sklearn and other libraries.

    Args:
        seed: Optional seed value. If None, uses the default seed of 42.

    Returns:
        The seed value to use for random_state parameters.
    """
    return seed if seed is not None else 42
