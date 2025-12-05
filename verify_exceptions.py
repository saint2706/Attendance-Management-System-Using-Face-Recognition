"""Quick verification script for exception handling changes."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def verify_views_exceptions():
    """Verify exception handling in views.py."""
    views_path = Path(__file__).parent / "recognition" / "views.py"
    content = views_path.read_text()

    checks = {
        "ValueError for DeepFace": "except ValueError as exc:" in content
        and "# DeepFace raises ValueError" in content,
        "AttributeError for DeepFace": "except AttributeError as exc:" in content
        and "# DeepFace raises AttributeError" in content,
        "OSError for DeepFace": "except OSError as exc:" in content
        and "# DeepFace raises OSError" in content,
        "OSError for cache": "except OSError as exc:  # pragma: no cover - defensive file I/O"
        in content,
        "pickle.UnpicklingError": "except pickle.UnpicklingError as exc:" in content,
        "Pickle edge cases": "except (EOFError, AttributeError, ImportError) as exc:" in content,
    }

    print("✓ Checking views.py exception handling...")
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    return all_passed


def verify_webcam_exceptions():
    """Verify exception handling in webcam_manager.py."""
    webcam_path = Path(__file__).parent / "recognition" / "webcam_manager.py"
    content = webcam_path.read_text()

    checks = {
        "OSError/RuntimeError for camera start": "except (OSError, RuntimeError) as exc:" in content
        and "# OSError: Camera device not available" in content,
        "OSError/RuntimeError for camera stop": content.count(
            "except (OSError, RuntimeError) as exc:"
        )
        >= 2,
        "Fallback Exception catches": content.count(
            "except Exception as exc:  # pragma: no cover - catch unexpected errors"
        )
        >= 2,
    }

    print("\n✓ Checking webcam_manager.py exception handling...")
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("Exception Handling Verification\n" + "=" * 50)

    views_ok = verify_views_exceptions()
    webcam_ok = verify_webcam_exceptions()

    print("\n" + "=" * 50)
    if views_ok and webcam_ok:
        print("✓ All exception handling checks passed!")
        sys.exit(0)
    else:
        print("✗ Some exception handling checks failed!")
        sys.exit(1)
