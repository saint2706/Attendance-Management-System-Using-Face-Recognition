#!/usr/bin/env python
"""
CLI tool for face recognition predictions with policy-based actions.

Usage:
    python predict_cli.py --image path/to/image.jpg
    python predict_cli.py --image path/to/image.jpg --threshold 0.75
    python predict_cli.py --image path/to/image.jpg --profile strict_office
    python predict_cli.py --image path/to/image.jpg --site office1
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _setup_django():
    """Initialize Django for CLI usage."""
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE",
        "attendance_system_facial_recognition.settings",
    )
    try:
        import django
        if not django.apps.apps.ready:
            django.setup()
        return True
    except Exception:
        return False


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


def get_threshold_from_profile(profile_name: str = None, site_code: str = None) -> tuple:
    """
    Get threshold from a profile name or site code.
    
    Returns:
        Tuple of (threshold, profile_name) where profile_name may be None for system default.
    """
    if not _setup_django():
        return None, None
    
    try:
        from recognition.models import ThresholdProfile
        from django.conf import settings
        
        if profile_name:
            try:
                profile = ThresholdProfile.objects.get(name=profile_name)
                return profile.distance_threshold, profile.name
            except ThresholdProfile.DoesNotExist:
                logger.warning(f"Profile '{profile_name}' not found, using system default")
        
        if site_code:
            profile = ThresholdProfile.get_for_site(site_code)
            if profile:
                return profile.distance_threshold, profile.name
        
        # Return system default
        system_default = float(getattr(settings, "RECOGNITION_DISTANCE_THRESHOLD", 0.4))
        return system_default, None
    except Exception as e:
        logger.debug(f"Could not load profile: {e}")
        return None, None


def predict(
    image_path: Path,
    threshold: float = 0.6,
    policy_path: Path = None,
    profile_name: str = None,
    site_code: str = None,
):
    """
    Run prediction on an image and return score, band, and recommended action.

    Args:
        image_path: Path to input image
        threshold: Decision threshold (not used for band selection)
        policy_path: Path to policy configuration
        profile_name: Name of threshold profile to use
        site_code: Site code for profile selection

    Returns:
        Dict with prediction results
    """
    if policy_path is None:
        policy_path = Path(__file__).parent / "configs" / "policy.yaml"

    if not policy_path.exists():
        logger.error(
            "Policy file not found",
            extra={"event": "policy_missing", "policy_path": str(policy_path)},
        )
        sys.exit(1)

    policy = load_policy(policy_path)

    # Try to get threshold from profile
    used_profile = None
    if profile_name or site_code:
        profile_threshold, used_profile = get_threshold_from_profile(profile_name, site_code)
        if profile_threshold is not None:
            threshold = profile_threshold

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
        "profile_used": used_profile,
        "site_code": site_code,
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
    parser.add_argument("--policy", type=str, default=None, help="Path to policy.yaml (optional)")
    parser.add_argument(
        "--profile", type=str, default=None, help="Name of threshold profile to use"
    )
    parser.add_argument(
        "--site", type=str, default=None, help="Site code for automatic profile selection"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(
            "Image not found",
            extra={"event": "image_missing", "image_path": str(image_path)},
        )
        sys.exit(1)

    policy_path = Path(args.policy) if args.policy else None

    result = predict(
        image_path,
        threshold=args.threshold,
        policy_path=policy_path,
        profile_name=args.profile,
        site_code=args.site,
    )

    log_context = {
        "event": "prediction_result",
        "image_path": str(image_path),
        "threshold": args.threshold,
        "policy_path": str(policy_path) if policy_path else None,
        "profile": args.profile,
        "site": args.site,
        "format": "json" if args.json else "human",
        "result": result,
    }

    if args.json:
        logger.info(json.dumps(result, indent=2), extra=log_context)
    else:
        lines = [
            "=" * 60,
            "Face Recognition Prediction",
            "=" * 60,
            f"Image: {result['image_path']}",
            f"Score: {result['score']:.4f}",
            f"Threshold: {result['threshold']:.4f}",
        ]
        if result.get("profile_used"):
            lines.append(f"Profile: {result['profile_used']}")
        if result.get("site_code"):
            lines.append(f"Site: {result['site_code']}")
        lines.extend([
            "",
            f"Band: {result['band'].upper()}",
            f"Description: {result['band_description']}",
            "",
            f"Recommended Action: {result['action']}",
            f"Action Description: {result['action_description']}",
            "",
            f"Mark Attendance: {'Yes' if result['mark_attendance'] else 'No'}",
            f"Requires Secondary Auth: {'Yes' if result['requires_secondary_auth'] else 'No'}",
            "",
            f"User Message: {result['user_message']}",
            "=" * 60,
        ])
        logger.info("\n".join(lines), extra=log_context)


if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
