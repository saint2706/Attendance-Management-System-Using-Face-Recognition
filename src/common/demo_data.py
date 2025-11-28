"""Helpers for generating synthetic demo datasets.

These utilities build a small, fully synthetic dataset that mirrors the
expected ``face_recognition_data/training_dataset`` layout. Images are encrypted
with the configured ``DATA_ENCRYPTION_KEY`` so the runtime pipeline can ingest
files immediately after generation.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Mapping

from PIL import Image, ImageDraw, ImageFont

from .crypto import encrypt_bytes

SYNTHETIC_USERS: Mapping[str, tuple[str, tuple[int, int, int]]] = {
    "user_001": ("Cyan", (0, 160, 200)),
    "user_002": ("Magenta", (180, 60, 160)),
    "user_003": ("Amber", (210, 150, 60)),
}


def _render_avatar_frame(username: str, frame_index: int, color: tuple[int, int, int]) -> bytes:
    """Return JPEG bytes for a synthetic avatar frame.

    Args:
        username: Identifier rendered onto the frame.
        frame_index: Zero-based frame index.
        color: RGB tuple used for the avatar fill colour.

    Returns:
        Raw JPEG bytes representing the synthetic frame.
    """

    base_color = (15, 15, 20)
    image = Image.new("RGB", (256, 256), base_color)
    draw = ImageDraw.Draw(image)

    draw.ellipse((68, 50, 188, 170), fill=color)
    torso = tuple(min(component + 30, 255) for component in color)
    draw.rectangle((70, 160, 190, 220), fill=torso)

    label = f"{username}\nFrame {frame_index + 1}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((40, 20), label, fill=(230, 230, 230), font=font)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def generate_encrypted_dataset(target_root: Path) -> List[Path]:
    """Generate an encrypted synthetic dataset under ``target_root``.

    The layout mirrors ``<root>/<username>/<username>_frame_XX.jpg``. Images are
    encrypted with :func:`encrypt_bytes` using the active ``DATA_ENCRYPTION_KEY``.

    Args:
        target_root: Destination root directory for the dataset.

    Returns:
        List of file paths that were created or overwritten.
    """

    created: List[Path] = []
    for username, (_, colour) in SYNTHETIC_USERS.items():
        user_dir = target_root / username
        user_dir.mkdir(parents=True, exist_ok=True)

        for frame_index in range(3):
            frame_bytes = _render_avatar_frame(username, frame_index, colour)
            encrypted = encrypt_bytes(frame_bytes)
            destination = user_dir / f"{username}_frame_{frame_index + 1:02d}.jpg"
            destination.write_bytes(encrypted)
            created.append(destination)

    return created


def sync_demo_dataset(sample_root: Path, training_root: Path) -> None:
    """Ensure both sample and runtime dataset roots contain encrypted assets."""

    sample_root.mkdir(parents=True, exist_ok=True)
    training_root.mkdir(parents=True, exist_ok=True)

    generated = generate_encrypted_dataset(sample_root)
    for path in generated:
        relative = path.relative_to(sample_root)
        destination = training_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
