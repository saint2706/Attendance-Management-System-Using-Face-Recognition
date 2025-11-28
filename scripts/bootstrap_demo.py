"""Bootstrap a local demo environment with synthetic data and users."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import django
from django.contrib.auth import get_user_model


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_django() -> None:
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")
    django.setup()


def _create_superuser(username: str, email: str, password: str) -> Tuple[bool, str]:
    """Create a superuser if one does not already exist."""

    User = get_user_model()
    created = False
    user = User.objects.filter(username=username).first()
    if user is None:
        user = User.objects.create_superuser(username=username, email=email, password=password)
        created = True
    else:
        if user.email != email:
            user.email = email
        if not user.is_superuser:
            user.is_superuser = True
            user.is_staff = True
        if password:
            user.set_password(password)
        user.save(update_fields=["email", "is_superuser", "is_staff", "password"])
    return created, user.username


def _ensure_demo_users(usernames: Iterable[str], password: str) -> list[str]:
    """Ensure demo user accounts matching the dataset exist."""

    User = get_user_model()
    created: list[str] = []
    for username in usernames:
        user, was_created = User.objects.get_or_create(username=username)
        if was_created:
            user.set_password(password)
            user.is_staff = False
            user.is_superuser = False
            user.save()
            created.append(username)
        elif password:
            user.set_password(password)
            user.save(update_fields=["password"])
    return created


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the local demo flow.")
    parser.add_argument(
        "--admin-username",
        default="demo_admin",
        help="Username for the demo superuser.",
    )
    parser.add_argument(
        "--admin-email",
        default="demo_admin@example.com",
        help="Email for the demo superuser.",
    )
    parser.add_argument(
        "--admin-password",
        default="demo_admin_pass",
        help="Password for the demo superuser.",
    )
    parser.add_argument(
        "--user-password",
        default="demo_user_pass",
        help="Password assigned to synthetic dataset users.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    project_root = _project_root()
    sample_root = project_root / "sample_data" / "face_recognition_data" / "training_dataset"
    training_root = project_root / "face_recognition_data" / "training_dataset"

    _ensure_django()
    from src.common import SYNTHETIC_USERS, sync_demo_dataset

    sync_demo_dataset(sample_root, training_root)
    created_admin, admin_username = _create_superuser(
        args.admin_username, args.admin_email, args.admin_password
    )
    created_users = _ensure_demo_users(SYNTHETIC_USERS.keys(), args.user_password)

    print("=== Demo bootstrap complete ===")
    print(f"Dataset root: {training_root}")
    print(f"Demo superuser: {admin_username} / {args.admin_password}")
    if created_admin:
        print("(created new superuser)")
    if created_users:
        print("Created demo users: " + ", ".join(created_users))
    else:
        print("Demo users already present; passwords refreshed.")
    print("Synthetic dataset refreshed at both sample_data/ and face_recognition_data/ roots.")
    print("Launch the server with: python manage.py runserver")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
