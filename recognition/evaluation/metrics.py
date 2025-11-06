"""
Verification-style metrics with confidence intervals for face recognition.

This module implements task-appropriate metrics including:
- ROC, PR, DET curves
- Equal Error Rate (EER)
- FAR@TPR and TPR@FAR operating points
- F1 at optimal threshold
- Brier score for calibration
- Bootstrap confidence intervals
- Calibration (reliability) diagrams
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the Equal Error Rate (EER) and its threshold.

    EER is the point where False Accept Rate (FAR) equals False Reject Rate (FRR).

    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted scores or probabilities

    Returns:
        Tuple of (eer, eer_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr  # False Negative Rate (same as False Reject Rate)

    # Find the point where FAR = FRR
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]

    return float(eer), float(eer_threshold)


def calculate_operating_points(
    y_true: np.ndarray, y_scores: np.ndarray, target_metrics: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Calculate operating points for FAR@TPR and TPR@FAR.

    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        target_metrics: Dict with keys like 'FAR@TPR=0.95' or 'TPR@FAR=0.01'

    Returns:
        Dictionary mapping metric names to their values and thresholds
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    operating_points = {}

    for metric_name, target_value in target_metrics.items():
        if "FAR@TPR" in metric_name:
            # Find FAR at given TPR
            target_tpr = target_value
            idx = np.argmin(np.abs(tpr - target_tpr))
            operating_points[metric_name] = {
                "value": float(fpr[idx]),
                "threshold": float(thresholds[idx]),
                "tpr": float(tpr[idx]),
            }
        elif "TPR@FAR" in metric_name:
            # Find TPR at given FAR
            target_far = target_value
            idx = np.argmin(np.abs(fpr - target_far))
            operating_points[metric_name] = {
                "value": float(tpr[idx]),
                "threshold": float(thresholds[idx]),
                "far": float(fpr[idx]),
            }

    return operating_points


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Find the threshold that maximizes F1 score.

    Args:
        y_true: True binary labels
        y_scores: Predicted scores

    Returns:
        Tuple of (optimal_threshold, best_f1)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Calculate F1 for each threshold
    # Note: precision_recall_curve returns n+1 precision/recall but n thresholds
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return best_threshold, best_f1


def calculate_verification_metrics(
    y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5
) -> Dict:
    """
    Calculate comprehensive verification metrics for face recognition.

    Args:
        y_true: True binary labels (1 for genuine, 0 for impostor)
        y_scores: Predicted scores or probabilities
        threshold: Classification threshold

    Returns:
        Dictionary containing all verification metrics
    """
    # Binary predictions based on threshold
    y_pred = (y_scores >= threshold).astype(int)

    # ROC metrics
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    # Precision-Recall metrics
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    # EER
    eer, eer_threshold = calculate_eer(y_true, y_scores)

    # Optimal F1 threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_true, y_scores)

    # Operating points
    operating_points = calculate_operating_points(
        y_true,
        y_scores,
        {"FAR@TPR=0.95": 0.95, "FAR@TPR=0.90": 0.90, "TPR@FAR=0.01": 0.01, "TPR@FAR=0.05": 0.05},
    )

    # Brier score (calibration metric)
    brier = brier_score_loss(y_true, y_scores)

    # F1 at current threshold
    f1_at_threshold = f1_score(y_true, y_pred)

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "brier_score": float(brier),
        "optimal_threshold": float(optimal_threshold),
        "optimal_f1": float(optimal_f1),
        "f1_at_threshold": float(f1_at_threshold),
        "threshold_used": float(threshold),
        "operating_points": operating_points,
    }

    return metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict:
    """
    Calculate bootstrap confidence intervals for key metrics.

    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with metrics and their confidence intervals
    """
    np.random.seed(random_state)

    n_samples = len(y_true)
    auc_scores = []
    eer_scores = []
    optimal_f1_scores = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]

        # Skip if bootstrap sample doesn't contain both classes
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            # Calculate metrics
            auc_scores.append(roc_auc_score(y_true_boot, y_scores_boot))
            eer, _ = calculate_eer(y_true_boot, y_scores_boot)
            eer_scores.append(eer)
            _, f1 = find_optimal_threshold(y_true_boot, y_scores_boot)
            optimal_f1_scores.append(f1)
        except Exception:
            continue

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    def get_ci(scores):
        if len(scores) == 0:
            return (None, None)
        return (
            float(np.percentile(scores, lower_percentile)),
            float(np.percentile(scores, upper_percentile)),
        )

    ci_results = {
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "auc": {
            "mean": float(np.mean(auc_scores)) if auc_scores else None,
            "ci_lower": get_ci(auc_scores)[0],
            "ci_upper": get_ci(auc_scores)[1],
        },
        "eer": {
            "mean": float(np.mean(eer_scores)) if eer_scores else None,
            "ci_lower": get_ci(eer_scores)[0],
            "ci_upper": get_ci(eer_scores)[1],
        },
        "optimal_f1": {
            "mean": float(np.mean(optimal_f1_scores)) if optimal_f1_scores else None,
            "ci_lower": get_ci(optimal_f1_scores)[0],
            "ci_upper": get_ci(optimal_f1_scores)[1],
        },
    }

    return ci_results


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: Path) -> None:
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: Path) -> None:
    """Generate and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (TPR)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_det_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: Path) -> None:
    """Generate and save Detection Error Tradeoff (DET) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    plt.figure(figsize=(8, 6))
    plt.plot(fpr * 100, fnr * 100, color="red", lw=2, label="DET curve")
    plt.xlabel("False Accept Rate (%)")
    plt.ylabel("False Reject Rate (%)")
    plt.title("Detection Error Tradeoff (DET)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray, y_scores: np.ndarray, output_path: Path, n_bins: int = 10
) -> None:
    """Generate and save calibration (reliability) diagram."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_scores, n_bins=n_bins, strategy="uniform"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot (Reliability Diagram)")
    plt.legend()
    plt.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_metric_plots(y_true: np.ndarray, y_scores: np.ndarray, output_dir: Path) -> None:
    """
    Generate all metric plots.

    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curve(y_true, y_scores, output_dir / "roc.png")
    plot_pr_curve(y_true, y_scores, output_dir / "pr.png")
    plot_det_curve(y_true, y_scores, output_dir / "det.png")
    plot_calibration_curve(y_true, y_scores, output_dir / "calibration.png")


def save_metrics_json(metrics: Dict, ci_results: Dict, output_path: Path) -> None:
    """Save metrics and confidence intervals to JSON."""
    combined = {
        "metrics": metrics,
        "confidence_intervals": ci_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)


def save_metrics_markdown(metrics: Dict, ci_results: Dict, output_path: Path) -> None:
    """Save metrics and confidence intervals to a readable Markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Face Recognition Evaluation Metrics\n\n")

        f.write("## Primary Metrics\n\n")
        f.write(f"- **ROC AUC**: {metrics['roc_auc']:.4f}\n")
        f.write(f"- **PR AUC**: {metrics['pr_auc']:.4f}\n")
        f.write(f"- **Equal Error Rate (EER)**: {metrics['eer']:.4f}\n")
        f.write(f"- **EER Threshold**: {metrics['eer_threshold']:.4f}\n")
        f.write(f"- **Brier Score**: {metrics['brier_score']:.4f}\n\n")

        f.write("## Optimal Operating Point\n\n")
        f.write(f"- **Optimal Threshold**: {metrics['optimal_threshold']:.4f}\n")
        f.write(f"- **F1 at Optimal Threshold**: {metrics['optimal_f1']:.4f}\n")
        f.write(f"- **F1 at Current Threshold**: {metrics['f1_at_threshold']:.4f}\n\n")

        f.write("## Operating Points\n\n")
        for op_name, op_data in metrics["operating_points"].items():
            f.write(f"### {op_name}\n")
            for key, value in op_data.items():
                f.write(f"- {key}: {value:.4f}\n")
            f.write("\n")

        f.write("## Confidence Intervals (95%)\n\n")
        if ci_results:
            for metric_name, ci_data in ci_results.items():
                if metric_name in ["n_bootstrap", "confidence_level"]:
                    continue
                if ci_data["mean"] is not None:
                    f.write(
                        f"- **{metric_name.upper()}**: {ci_data['mean']:.4f} "
                        f"[{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}]\n"
                    )
