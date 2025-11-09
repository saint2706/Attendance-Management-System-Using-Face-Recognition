"""Shared pytest fixtures and test bootstrapping for the project."""

import sys
from unittest.mock import MagicMock


# Ensure OpenCV imports don't fail in environments without the native bindings.
sys.modules.setdefault("cv2", MagicMock(name="cv2"))
