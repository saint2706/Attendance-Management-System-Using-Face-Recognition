# Documentation Screenshots

This directory contains screenshots used in the project documentation.

## Generating Screenshots

Screenshots can be regenerated using the capture script:

```bash
# With the development server running
make docs-screenshots
```

Or directly:

```bash
python scripts/capture_screenshots.py
```

See [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md#11-regenerating-documentation-screenshots) for detailed instructions.

## Screenshot Files

The following screenshots are captured by the script:

| File | Description |
|------|-------------|
| `home.png` | Landing page with primary actions |
| `login.png` | User authentication screen |
| `admin-dashboard.png` | Admin home with first-run checklist |
| `employee-registration.png` | Form for registering new employees |
| `employee-enrollment.png` | Photo capture for face recognition |
| `attendance-session.png` | Live recognition feed and logs |
| `reports.png` | Attendance reports dashboard |
| `system-health.png` | Operational health dashboard |
| `fairness-dashboard.png` | Per-group fairness metrics |
| `evaluation-dashboard.png` | Model performance metrics |

## Manual Updates

If you prefer to capture screenshots manually:

1. Run the demo environment: `make demo && python manage.py runserver`
2. Log in with `demo_admin` / `demo_admin_pass`
3. Navigate to each screen and capture using your browser or OS tools
4. Save images using the filenames listed above
5. Use a consistent resolution (1280×800 recommended)
