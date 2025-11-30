"""Capture documentation screenshots using Playwright.

This script captures a canonical set of screenshots from the running application
for use in documentation. It assumes the demo environment has been bootstrapped
with `make demo` and the development server is running at http://127.0.0.1:8000/.

Usage:
    1. Bootstrap the demo: make demo
    2. Start the server: python manage.py runserver
    3. Capture screenshots: python scripts/capture_screenshots.py

The script saves screenshots to docs/screenshots/ with consistent filenames.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_django() -> None:
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture documentation screenshots from the running app."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running Django development server.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save screenshots (defaults to docs/screenshots/).",
    )
    parser.add_argument(
        "--admin-username",
        default="demo_admin",
        help="Admin username for authenticated screenshots.",
    )
    parser.add_argument(
        "--admin-password",
        default="demo_admin_pass",
        help="Admin password for authenticated screenshots.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode (shows browser window). Default is headless.",
    )
    return parser.parse_args()


def capture_screenshots(
    base_url: str,
    output_dir: Path,
    admin_username: str,
    admin_password: str,
    headless: bool = True,
) -> dict[str, Path]:
    """Capture the canonical set of documentation screenshots.

    Args:
        base_url: Base URL of the running Django server.
        output_dir: Directory where screenshots will be saved.
        admin_username: Username for admin login.
        admin_password: Password for admin login.
        headless: Whether to run browser in headless mode.

    Returns:
        Dictionary mapping screenshot names to their file paths.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: Playwright is not installed.")
        print("Install it with: pip install pytest-playwright && playwright install chromium")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots: dict[str, Path] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        # 1. Home page (landing page)
        print("Capturing: Home page...")
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        home_path = output_dir / "home.png"
        page.screenshot(path=str(home_path))
        screenshots["home"] = home_path

        # 2. Login page
        print("Capturing: Login page...")
        page.goto(f"{base_url}/login/")
        page.wait_for_load_state("networkidle")
        login_path = output_dir / "login.png"
        page.screenshot(path=str(login_path))
        screenshots["login"] = login_path

        # 3. Log in as admin
        print("Logging in as admin...")
        page.fill('input[name="username"]', admin_username)
        page.fill('input[name="password"]', admin_password)
        page.click('button[type="submit"]')
        page.wait_for_load_state("networkidle")

        # 4. Admin dashboard
        print("Capturing: Admin dashboard...")
        page.goto(f"{base_url}/dashboard/")
        page.wait_for_load_state("networkidle")
        admin_dashboard_path = output_dir / "admin-dashboard.png"
        page.screenshot(path=str(admin_dashboard_path))
        screenshots["admin-dashboard"] = admin_dashboard_path

        # 5. Employee registration page
        print("Capturing: Employee registration...")
        page.goto(f"{base_url}/register/")
        page.wait_for_load_state("networkidle")
        register_path = output_dir / "employee-registration.png"
        page.screenshot(path=str(register_path))
        screenshots["employee-registration"] = register_path

        # 6. Add photos page (employee enrollment)
        print("Capturing: Add photos / enrollment...")
        page.goto(f"{base_url}/add_photos/")
        page.wait_for_load_state("networkidle")
        enrollment_path = output_dir / "employee-enrollment.png"
        page.screenshot(path=str(enrollment_path))
        screenshots["employee-enrollment"] = enrollment_path

        # 7. Attendance session page
        print("Capturing: Attendance session...")
        page.goto(f"{base_url}/attendance-session/")
        page.wait_for_load_state("networkidle")
        session_path = output_dir / "attendance-session.png"
        page.screenshot(path=str(session_path))
        screenshots["attendance-session"] = session_path

        # 8. Attendance reports / view attendance home
        print("Capturing: Attendance reports...")
        page.goto(f"{base_url}/view_attendance_home")
        page.wait_for_load_state("networkidle")
        reports_path = output_dir / "reports.png"
        page.screenshot(path=str(reports_path))
        screenshots["reports"] = reports_path

        # 9. System health dashboard (admin only)
        print("Capturing: System health...")
        page.goto(f"{base_url}/admin/health/")
        page.wait_for_load_state("networkidle")
        health_path = output_dir / "system-health.png"
        page.screenshot(path=str(health_path))
        screenshots["system-health"] = health_path

        # 10. Fairness dashboard (if accessible)
        print("Capturing: Fairness dashboard...")
        page.goto(f"{base_url}/admin/fairness/")
        page.wait_for_load_state("networkidle")
        fairness_path = output_dir / "fairness-dashboard.png"
        page.screenshot(path=str(fairness_path))
        screenshots["fairness-dashboard"] = fairness_path

        # 11. Evaluation dashboard
        print("Capturing: Evaluation dashboard...")
        page.goto(f"{base_url}/admin/evaluation/")
        page.wait_for_load_state("networkidle")
        evaluation_path = output_dir / "evaluation-dashboard.png"
        page.screenshot(path=str(evaluation_path))
        screenshots["evaluation-dashboard"] = evaluation_path

        browser.close()

    return screenshots


def main() -> int:
    args = _parse_args()
    _ensure_django()

    project_root = _project_root()
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "docs" / "screenshots"

    headless = not args.headed

    print("=" * 60)
    print("Screenshot Capture for Documentation")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'headless' if headless else 'headed'}")
    print()

    try:
        screenshots = capture_screenshots(
            base_url=args.base_url,
            output_dir=output_dir,
            admin_username=args.admin_username,
            admin_password=args.admin_password,
            headless=headless,
        )
    except Exception as e:
        print(f"\nERROR: Failed to capture screenshots: {e}")
        print("\nMake sure:")
        print("  1. The development server is running: python manage.py runserver")
        print("  2. The demo environment is set up: make demo")
        print(
            "  3. Playwright is installed: pip install pytest-playwright && playwright install chromium"
        )
        return 1

    print()
    print("=" * 60)
    print("Screenshots captured successfully!")
    print("=" * 60)
    for name, path in screenshots.items():
        print(f"  {name}: {path}")
    print()
    print(f"Total: {len(screenshots)} screenshots saved to {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
