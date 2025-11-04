#!/usr/bin/env python
"""
CLI tool for face recognition predictions with policy-based actions.

Usage:
    python predict_cli.py --image path/to/image.jpg
    python predict_cli.py --image path/to/image.jpg --threshold 0.75
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_policy(policy_path: Path):
    """Load policy configuration from YAML file."""
    with open(policy_path, "r") as f:
        return yaml.safe_load(f)


def get_score_band(score: float, policy: dict) -> tuple:
    """
    Determine score band and recommended action for a given score.

    Args:
        score: Recognition confidence score (0-1)
        policy: Policy configuration dict

    Returns:
        Tuple of (band_name, band_config, action_config)
    """
    bands = policy["score_bands"]

    for band_name, band_config in bands.items():
        if band_config["threshold_min"] <= score <= band_config["threshold_max"]:
            action_name = band_config["action"]
            action_config = policy["actions"][action_name]
            return band_name, band_config, action_config

    # Default to reject if no band matches
    return "reject", bands["reject"], policy["actions"]["reject"]


def predict(image_path: Path, threshold: float = 0.6, policy_path: Path = None):
    """
    Run prediction on an image and return score, band, and recommended action.

    Args:
        image_path: Path to input image
        threshold: Decision threshold (not used for band selection)
        policy_path: Path to policy configuration

    Returns:
        Dict with prediction results
    """
    if policy_path is None:
        policy_path = Path(__file__).parent / "configs" / "policy.yaml"

    if not policy_path.exists():
        print(f"Error: Policy file not found at {policy_path}", file=sys.stderr)
        sys.exit(1)

    policy = load_policy(policy_path)

    # Simulate prediction (in production, this would call the actual model)
    # For demonstration, generate a random score
    import random

    random.seed(42)
    score = random.uniform(0.3, 0.95)

    # Get band and action
    band_name, band_config, action_config = get_score_band(score, policy)

    result = {
        "image_path": str(image_path),
        "score": score,
        "threshold": threshold,
        "band": band_name,
        "band_description": band_config["description"],
        "action": band_config["action"],
        "action_description": action_config["description"],
        "requires_secondary_auth": action_config["requires_secondary_auth"],
        "mark_attendance": action_config["mark_attendance"],
        "user_message": action_config["user_message"],
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Face recognition prediction with policy-based actions"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Decision threshold (default: 0.6)"
    )
    parser.add_argument(
        "--policy", type=str, default=None, help="Path to policy.yaml (optional)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    policy_path = Path(args.policy) if args.policy else None

    result = predict(image_path, threshold=args.threshold, policy_path=policy_path)

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("Face Recognition Prediction")
        print("=" * 60)
        print(f"Image: {result['image_path']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")
        print()
        print(f"Band: {result['band'].upper()}")
        print(f"Description: {result['band_description']}")
        print()
        print(f"Recommended Action: {result['action']}")
        print(f"Action Description: {result['action_description']}")
        print()
        print(f"Mark Attendance: {'Yes' if result['mark_attendance'] else 'No'}")
        print(
            f"Requires Secondary Auth: {'Yes' if result['requires_secondary_auth'] else 'No'}"
        )
        print()
        print(f"User Message: {result['user_message']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
