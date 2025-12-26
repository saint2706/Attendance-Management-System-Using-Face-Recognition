"""
Failure analysis utilities for face recognition.

This module provides functions to analyze and document failure cases,
including false accepts and false rejects, with metadata about potential
causes (lighting, pose, occlusion).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def detect_lighting_issues(image_path: Optional[Path]) -> str:
    """
    Heuristic to detect lighting issues in an image.

    Args:
        image_path: Path to the image

    Returns:
        String describing lighting condition ('normal', 'dark', 'bright', 'unknown')
    """
    if image_path is None:
        return "unknown"

    try:
        import cv2

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "unknown"

        mean_brightness = np.mean(img)
        if mean_brightness < 60:
            return "dark"
        elif mean_brightness > 200:
            return "bright"
        else:
            return "normal"
    except Exception:
        return "unknown"


def detect_pose_issues(facial_area: Optional[Dict]) -> str:
    """
    Heuristic to detect pose issues based on facial area geometry and landmarks.

    Analyzes the face bounding box aspect ratio and landmark positions
    to estimate head pose (yaw angle primarily).

    Args:
        facial_area: Dictionary with face bounding box coordinates.
            Expected keys: 'x', 'y', 'w', 'h' for bounding box.
            Optional keys: 'left_eye', 'right_eye' for landmark-based analysis.

    Returns:
        String describing pose:
        - 'frontal': Face is roughly frontal (within ~15° yaw)
        - 'profile': Face is turned significantly (>30° yaw)
        - 'tilted': Face has significant roll/pitch
        - 'unknown': Cannot determine pose
    """
    if facial_area is None:
        return "unknown"

    try:
        # Extract bounding box dimensions
        w = facial_area.get("w", 0)
        h = facial_area.get("h", 0)

        if w <= 0 or h <= 0:
            return "unknown"

        # Check for eye landmarks (more accurate pose estimation)
        left_eye = facial_area.get("left_eye")
        right_eye = facial_area.get("right_eye")

        if left_eye and right_eye:
            # Calculate eye-line angle for roll detection
            eye_dx = right_eye[0] - left_eye[0]
            eye_dy = right_eye[1] - left_eye[1]

            if eye_dx != 0:
                roll_angle = abs(np.arctan(eye_dy / eye_dx) * 180 / np.pi)
                if roll_angle > 20:
                    return "tilted"

            # Calculate eye distance ratio relative to face width
            eye_distance = np.sqrt(eye_dx**2 + eye_dy**2)
            eye_ratio = eye_distance / w

            # Frontal faces typically have eye distance ~35-45% of face width
            # Profile faces have compressed eye distance
            if eye_ratio < 0.25:
                return "profile"
            elif eye_ratio > 0.55:
                # Very wide eye spacing might indicate extreme frontal or distortion
                return "frontal"

        # Fall back to aspect ratio heuristic
        # Frontal faces typically have width/height ratio ~0.7-0.9
        aspect_ratio = w / h

        if aspect_ratio < 0.5:
            # Very narrow face suggests profile view
            return "profile"
        elif aspect_ratio > 1.1:
            # Very wide face suggests tilted or unusual angle
            return "tilted"
        else:
            return "frontal"

    except Exception:
        return "unknown"


def detect_occlusion(
    image_path: Optional[Path], facial_area: Optional[Dict] = None
) -> str:
    """
    Heuristic to detect occlusion (glasses, masks, etc.) using image analysis.

    Analyzes the lower face region (mouth/nose area) for color uniformity
    and landmark visibility to detect potential occlusions.

    Args:
        image_path: Path to the image
        facial_area: Optional dictionary with face landmarks.
            If provided with 'nose' and 'mouth_left'/'mouth_right' keys,
            can provide more accurate occlusion detection.

    Returns:
        String describing occlusion status:
        - 'none': No significant occlusion detected
        - 'partial': Glasses or minor occlusion detected
        - 'heavy': Mask or major occlusion detected (mouth/nose covered)
        - 'unknown': Cannot determine occlusion status
    """
    if image_path is None:
        return "unknown"

    try:
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return "unknown"

        # Convert to grayscale and HSV for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        height, width = gray.shape[:2]

        # If we have facial landmarks, use them for precise analysis
        if facial_area:
            face_x = facial_area.get("x", 0)
            face_y = facial_area.get("y", 0)
            face_w = facial_area.get("w", width)
            face_h = facial_area.get("h", height)

            # Define lower face region (bottom 40% of face - nose/mouth area)
            lower_y = face_y + int(face_h * 0.6)
            lower_h = int(face_h * 0.4)
            lower_region = gray[
                max(0, lower_y) : min(height, lower_y + lower_h),
                max(0, face_x) : min(width, face_x + face_w),
            ]

            # Define eye region (top 30-50% of face)
            eye_y = face_y + int(face_h * 0.25)
            eye_h = int(face_h * 0.25)
            eye_region = gray[
                max(0, eye_y) : min(height, eye_y + eye_h),
                max(0, face_x) : min(width, face_x + face_w),
            ]
        else:
            # Fall back to center regions
            lower_region = gray[int(height * 0.6) :, int(width * 0.25) : int(width * 0.75)]
            eye_region = gray[
                int(height * 0.25) : int(height * 0.45),
                int(width * 0.2) : int(width * 0.8),
            ]

        # Check for heavy occlusion (mask detection)
        if lower_region.size > 0:
            lower_variance = np.var(lower_region)
            # Masks typically have low texture variance (uniform color)
            # Natural skin/mouth has higher variance
            if lower_variance < 200:  # Low variance suggests uniform covering
                # Additional check: look for common mask colors (blue, white, black)
                if facial_area:
                    lower_hsv = hsv[
                        max(0, lower_y) : min(height, lower_y + lower_h),
                        max(0, face_x) : min(width, face_x + face_w),
                    ]
                else:
                    lower_hsv = hsv[
                        int(height * 0.6) :, int(width * 0.25) : int(width * 0.75)
                    ]

                if lower_hsv.size > 0:
                    # Check saturation - masks often have low saturation (white/black)
                    # or specific hue (blue surgical masks)
                    mean_saturation = np.mean(lower_hsv[:, :, 1])
                    if mean_saturation < 40:  # Low saturation = likely mask
                        return "heavy"

        # Check for partial occlusion (glasses detection)
        if eye_region.size > 0:
            # Glasses create distinct horizontal edges across eyes
            eye_edges = cv2.Canny(eye_region, 50, 150)
            edge_density = np.sum(eye_edges > 0) / eye_edges.size

            # High edge density in eye region may indicate glasses frames
            if edge_density > 0.15:
                return "partial"

        return "none"

    except Exception:
        return "unknown"


def analyze_failures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    image_paths: Optional[List[Path]] = None,
    labels: Optional[List[str]] = None,
    top_n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze top-N false accepts and false rejects.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_scores: Prediction scores
        image_paths: Optional list of image paths
        labels: Optional list of identity labels
        top_n: Number of top failures to return

    Returns:
        Tuple of (false_accepts_df, false_rejects_df)
    """
    # False Accepts: predicted 1 but true is 0
    fa_indices = np.where((y_pred == 1) & (y_true == 0))[0]

    # False Rejects: predicted 0 but true is 1
    fr_indices = np.where((y_pred == 0) & (y_true == 1))[0]

    def create_failure_df(indices, failure_type):
        if len(indices) == 0:
            return pd.DataFrame()

        # Sort by score (confidence)
        sorted_indices = indices[np.argsort(-y_scores[indices])][:top_n]

        rows = []
        for idx in sorted_indices:
            row = {
                "index": int(idx),
                "failure_type": failure_type,
                "true_label": int(y_true[idx]),
                "predicted_label": int(y_pred[idx]),
                "score": float(y_scores[idx]),
            }

            if image_paths is not None and idx < len(image_paths):
                img_path = image_paths[idx]
                row["image_path"] = str(img_path)
                row["lighting"] = detect_lighting_issues(img_path)
                row["pose"] = detect_pose_issues(None)
                row["occlusion"] = detect_occlusion(img_path)
            else:
                row["image_path"] = "N/A"
                row["lighting"] = "unknown"
                row["pose"] = "unknown"
                row["occlusion"] = "unknown"

            if labels is not None and idx < len(labels):
                row["identity"] = labels[idx]
            else:
                row["identity"] = "unknown"

            rows.append(row)

        return pd.DataFrame(rows)

    fa_df = create_failure_df(fa_indices, "false_accept")
    fr_df = create_failure_df(fr_indices, "false_reject")

    return fa_df, fr_df


def generate_failure_report(fa_df: pd.DataFrame, fr_df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate a narrative failure analysis report.

    Args:
        fa_df: DataFrame of false accepts
        fr_df: DataFrame of false rejects
        output_path: Path to save the Markdown report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Failure Analysis Report\n\n")

        f.write(
            "This report analyzes failure cases (false accepts and false rejects) to identify "
            "patterns and potential improvements.\n\n"
        )

        # False Accepts
        f.write("## False Accepts\n\n")
        if len(fa_df) > 0:
            f.write(f"**Total False Accepts Analyzed**: {len(fa_df)} (top cases by confidence)\n\n")

            # Lighting analysis
            lighting_counts = fa_df["lighting"].value_counts()
            f.write("### Lighting Conditions\n\n")
            for condition, count in lighting_counts.items():
                f.write(f"- {condition}: {count} cases\n")
            f.write("\n")

            # Pose analysis
            pose_counts = fa_df["pose"].value_counts()
            f.write("### Pose Distribution\n\n")
            for pose, count in pose_counts.items():
                f.write(f"- {pose}: {count} cases\n")
            f.write("\n")

            # Representative cases
            f.write("### Representative Cases\n\n")
            for i, row in fa_df.head(3).iterrows():
                f.write(f"**Case {i + 1}:**\n")
                f.write(f"- Predicted Score: {row['score']:.4f}\n")
                f.write(f"- Lighting: {row['lighting']}\n")
                f.write(f"- Pose: {row['pose']}\n")
                f.write(f"- Occlusion: {row['occlusion']}\n\n")
        else:
            f.write("No false accepts found.\n\n")

        # False Rejects
        f.write("## False Rejects\n\n")
        if len(fr_df) > 0:
            f.write(f"**Total False Rejects Analyzed**: {len(fr_df)} (lowest confidence cases)\n\n")

            # Lighting analysis
            lighting_counts = fr_df["lighting"].value_counts()
            f.write("### Lighting Conditions\n\n")
            for condition, count in lighting_counts.items():
                f.write(f"- {condition}: {count} cases\n")
            f.write("\n")

            # Pose analysis
            pose_counts = fr_df["pose"].value_counts()
            f.write("### Pose Distribution\n\n")
            for pose, count in pose_counts.items():
                f.write(f"- {pose}: {count} cases\n")
            f.write("\n")

            # Representative cases
            f.write("### Representative Cases\n\n")
            for i, row in fr_df.head(3).iterrows():
                f.write(f"**Case {i + 1}:**\n")
                f.write(f"- Predicted Score: {row['score']:.4f}\n")
                f.write(f"- Lighting: {row['lighting']}\n")
                f.write(f"- Pose: {row['pose']}\n")
                f.write(f"- Occlusion: {row['occlusion']}\n\n")
        else:
            f.write("No false rejects found.\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        f.write("### For False Accepts:\n")
        f.write(
            "1. **Increase threshold**: Raising the acceptance threshold will reduce false accepts at the cost of more false rejects.\n"
        )
        f.write(
            "2. **Improve enrollment quality**: Ensure enrollment photos are high-quality, well-lit, and frontal.\n"
        )
        f.write(
            "3. **Add liveness detection**: Implement anti-spoofing to prevent photo-based attacks.\n\n"
        )

        f.write("### For False Rejects:\n")
        f.write(
            "1. **Lower threshold**: Reducing the threshold will decrease false rejects but may increase false accepts.\n"
        )
        f.write(
            "2. **Collect more diverse samples**: Include images with various lighting conditions and poses during enrollment.\n"
        )
        f.write(
            "3. **Improve preprocessing**: Enhance face alignment and normalization to handle pose variations.\n"
        )
        f.write(
            "4. **Secondary authentication**: For low-confidence matches, prompt for PIN or OTP.\n\n"
        )


def analyze_subgroups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
    output_path: Optional[Path],
) -> pd.DataFrame:
    """
    Perform subgroup analysis to detect bias.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_scores: Prediction scores
        groups: Group labels for each sample (e.g., camera ID, time of day)
        output_path: Path to save subgroup metrics

    Returns:
        DataFrame with per-group metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    unique_groups = np.unique(groups)
    rows = []

    for group in unique_groups:
        group_mask = groups == group
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]

        if len(group_y_true) == 0:
            continue

        row = {
            "group": str(group),
            "n_samples": len(group_y_true),
            "accuracy": accuracy_score(group_y_true, group_y_pred),
            "precision": precision_score(
                group_y_true, group_y_pred, average="binary", zero_division=0
            ),
            "recall": recall_score(group_y_true, group_y_pred, average="binary", zero_division=0),
            "f1_score": f1_score(group_y_true, group_y_pred, average="binary", zero_division=0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
