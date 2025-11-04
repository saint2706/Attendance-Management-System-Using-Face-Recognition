"""
Evaluation utilities for face recognition metrics.
"""
from .metrics import (
    bootstrap_confidence_intervals,
    calculate_eer,
    calculate_verification_metrics,
    generate_metric_plots,
)

__all__ = [
    "calculate_verification_metrics",
    "calculate_eer",
    "bootstrap_confidence_intervals",
    "generate_metric_plots",
]
