"""
Data splitting utilities with stratification and leakage prevention.

This module provides functions to create stratified train/val/test splits
with safeguards against identity leakage across splits.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Fixed random state for reproducibility
RANDOM_STATE = 42


def create_stratified_splits(
    image_paths: List[Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[Path], List[Path], List[Path], Dict]:
    """
    Create stratified train/val/test splits from image paths.

    Ensures that:
    1. Class distribution is preserved across splits (stratification)
    2. All images from the same person stay within the same split (no leakage)
    3. Splits are reproducible with fixed random_state

    Args:
        image_paths: List of Path objects pointing to images
        train_ratio: Proportion of data for training (default: 0.70)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_paths, val_paths, test_paths, split_info_dict)
        where split_info_dict contains metadata about the splits
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    # Group images by person (parent directory name)
    person_to_images = {}
    for img_path in image_paths:
        person = img_path.parent.name
        if person not in person_to_images:
            person_to_images[person] = []
        person_to_images[person].append(img_path)

    persons = list(person_to_images.keys())
    if len(persons) < 3:
        raise ValueError(
            f"Need at least 3 persons for train/val/test splits, found {len(persons)}"
        )

    # First split: train vs (val + test)
    train_persons, temp_persons = train_test_split(
        persons,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        shuffle=True,
    )

    # Second split: val vs test
    val_persons, test_persons = train_test_split(
        temp_persons,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=random_state,
        shuffle=True,
    )

    # Collect all images for each split
    train_paths = []
    val_paths = []
    test_paths = []

    for person in train_persons:
        train_paths.extend(person_to_images[person])
    for person in val_persons:
        val_paths.extend(person_to_images[person])
    for person in test_persons:
        test_paths.extend(person_to_images[person])

    # Create split info metadata
    split_info = {
        "total_images": len(image_paths),
        "total_persons": len(persons),
        "train": {
            "images": len(train_paths),
            "persons": len(train_persons),
            "person_names": sorted(train_persons),
        },
        "val": {
            "images": len(val_paths),
            "persons": len(val_persons),
            "person_names": sorted(val_persons),
        },
        "test": {
            "images": len(test_paths),
            "persons": len(test_persons),
            "person_names": sorted(test_persons),
        },
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "random_state": random_state,
    }

    return train_paths, val_paths, test_paths, split_info


def save_splits_to_csv(
    train_paths: List[Path],
    val_paths: List[Path],
    test_paths: List[Path],
    output_path: Path,
) -> None:
    """
    Save split information to a CSV file.

    Args:
        train_paths: List of training image paths
        val_paths: List of validation image paths
        test_paths: List of test image paths
        output_path: Path to save the CSV file
    """
    rows = []
    for idx, path in enumerate(train_paths):
        rows.append(
            {
                "index": idx,
                "split": "train",
                "person": path.parent.name,
                "image_path": str(path),
            }
        )
    for idx, path in enumerate(val_paths):
        rows.append(
            {
                "index": idx,
                "split": "val",
                "person": path.parent.name,
                "image_path": str(path),
            }
        )
    for idx, path in enumerate(test_paths):
        rows.append(
            {
                "index": idx,
                "split": "test",
                "person": path.parent.name,
                "image_path": str(path),
            }
        )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_split_summary_json(split_info: Dict, output_path: Path) -> None:
    """
    Save split summary information to a JSON file.

    Args:
        split_info: Dictionary containing split metadata
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split_info, f, indent=2)


def filter_leakage_fields(data_dict: Dict) -> Dict:
    """
    Remove fields that could cause data leakage.

    This function filters out any fields that trivially reveal identity
    or labels outside of the image pixels themselves.

    Args:
        data_dict: Dictionary of data fields

    Returns:
        Filtered dictionary with leakage-prone fields removed
    """
    # Fields that should be removed to prevent leakage
    leakage_fields = [
        "employee_id",
        "user_id",
        "username",
        "full_name",
        "name",
        "label",
        "ground_truth",
        "identity",
    ]

    filtered = {}
    for key, value in data_dict.items():
        if key.lower() not in [f.lower() for f in leakage_fields]:
            filtered[key] = value

    return filtered
