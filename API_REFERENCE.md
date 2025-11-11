# API Documentation

This document outlines all the URL patterns for the project and explains the purpose and functionality of each corresponding view.

## Core Pages

| URL Path      | View Function           | Name        | Description |
|---------------|-------------------------|------------|-------------|
| `/`           | `recog_views.home`      | `home`      | Landing page with quick actions for employees and admins. |
| `/dashboard/` | `recog_views.dashboard` | `dashboard` | Role-aware dashboard summarizing attendance insights. |

## Progressive Web App Assets

| URL Path        | View Function                        | Name             | Description |
|-----------------|--------------------------------------|------------------|-------------|
| `/manifest.json`| `progressive_web_app_manifest`       | `pwa-manifest`   | Serves the PWA manifest so browsers can install the dashboard. |
| `/sw.js`        | `progressive_web_app_service_worker` | `service-worker` | Exposes the service worker used for offline caching and push refreshes. |

## Authentication

| URL Path   | View Function/Class     | Name    | Description |
|------------|-------------------------|---------|-------------|
| `/login/`  | `auth_views.LoginView`  | `login` | Displays the login form and signs the user in. |
| `/logout/` | `auth_views.LogoutView` | `logout`| Logs the user out and redirects to the home page. |

## User and Photo Management (Admin-only)

| URL Path       | View Function            | Name         | Description |
|----------------|--------------------------|--------------|-------------|
| `/register/`   | `users_views.register`   | `register`   | Lets staff add employee accounts with default credentials. |
| `/add_photos/` | `recog_views.add_photos` | `add-photos` | Captures enrollment photos and stores embeddings for employees. |
| `/train/`      | `recog_views.train`      | `train`      | **Obsolete:** Redirects to the dashboard once automated training is available. |

## Evaluation and Analytics (Admin-only)

| URL Path             | View Function                            | Name                      | Description |
|----------------------|------------------------------------------|---------------------------|-------------|
| `/admin/evaluation/` | `recog_admin_views.evaluation_dashboard` | `admin_evaluation_dashboard` | Displays metrics, trend charts, and confidence intervals. See the [Evaluation Dashboard](docs/user-guide.md#evaluation-dashboard) for a guided tour. |
| `/admin/ablation/`   | `recog_admin_views.ablation_results`     | `admin_ablation_results`     | Compares feature flags and models via experiment matrices. Refer to the [Ablation Experiments Dashboard](docs/user-guide.md#ablation-experiments-dashboard) for usage notes. |
| `/admin/failures/`   | `recog_admin_views.failure_analysis`     | `admin_failure_analysis`     | Highlights misclassifications with evidence packs. See the [Failure Analysis Dashboard](docs/user-guide.md#failure-analysis-dashboard) for investigation workflows. |

## Face Recognition and Attendance Marking

| URL Path                   | View Function                          | Name                     | Description |
|----------------------------|----------------------------------------|--------------------------|-------------|
| `/mark_your_attendance`    | `recog_views.mark_your_attendance`     | `mark-your-attendance`   | Launches the camera workflow to mark a time-in event. |
| `/mark_your_attendance_out`| `recog_views.mark_your_attendance_out` | `mark-your-attendance-out` | Launches the camera workflow to mark a time-out event. |
| `/api/face-recognition/`   | `recog_views.FaceRecognitionAPI`       | `face-recognition-api`   | Accepts embeddings or frames and returns the nearest enrolled identity. |
| `/api/attendance/batch/`   | `recog_views.enqueue_attendance_batch` | `attendance-batch`       | Queues attendance records for asynchronous persistence via Celery. |

### `POST /api/attendance/batch/`

- **Authentication:** Required (session cookie).
- **Rate limiting:** Shares the attendance throttling applied to recognition flows.
- **Request body:** JSON object containing a `records` array. Each record must include:
  - `direction` (`"in"` or `"out"`) – selects the check-in or check-out pipeline.
  - `present` (object) or `payload` (object) – key/value pairs for employee identifiers and their attendance metadata (timestamps, device IDs, confidence scores, etc.).
- **Success response:** `202 Accepted` with JSON payload:

  ```json
  {
    "task_id": "4d7a2c64-3f37-4c5c-884f-0b9d27d9d6d3",
    "status": "PENDING",
    "total": 2
  }
  ```

  The response confirms that processing was enqueued in Celery. Poll the task result backend (e.g., `/celery-progress/`) for completion details.
- **Error responses:**
  - `400 Bad Request` – invalid JSON or malformed `records` payload.
  - `405 Method Not Allowed` – non-`POST` methods.
  - `503 Service Unavailable` – Celery queue failures.

## Attendance Viewing

| URL Path                     | View Function                               | Name                             | Description |
|------------------------------|---------------------------------------------|----------------------------------|-------------|
| `/view_attendance_home`      | `recog_views.view_attendance_home`          | `view-attendance-home`           | Overview of attendance analytics for administrators. |
| `/view_attendance_date`      | `recog_views.view_attendance_date`          | `view-attendance-date`           | Lists all attendance records for a selected date. |
| `/view_attendance_employee`  | `recog_views.view_attendance_employee`      | `view-attendance-employee`       | Filters attendance history for a single employee. |
| `/view_my_attendance`        | `recog_views.view_my_attendance_employee_login` | `view-my-attendance-employee-login` | Lets an employee review their own attendance timeline. |

## Error/Status Pages

| URL Path        | View Function                 | Name             | Description |
|-----------------|-------------------------------|------------------|-------------|
| `/not_authorised` | `recog_views.not_authorised` | `not-authorised` | Shown when a user lacks permission to view a page. |
## Command-Line Tools

The project includes several command-line tools for evaluation, testing, and analysis:

### Management Commands

All management commands are run using `python manage.py <command>`:

| Command | Description | Example |
|---------|-------------|---------|
| `prepare_splits` | Prepares stratified train/validation/test splits with identity-level grouping | `python manage.py prepare_splits --seed 42` |
| `eval` | Runs comprehensive evaluation with verification metrics and confidence intervals | `python manage.py eval --seed 42 --n-bootstrap 1000` |
| `threshold_select` | Selects optimal recognition threshold based on validation set | `python manage.py threshold_select --method eer` |
| `ablation` | Runs ablation experiments to test different component configurations | `python manage.py ablation --seed 42` |
| `export_reports` | Exports all generated reports and figures to consolidated directory | `python manage.py export_reports` |

### Standalone CLI Tool

| Tool | Description | Example |
|------|-------------|---------|
| `predict_cli.py` | Standalone prediction tool with policy-based action recommendations | `python predict_cli.py --image path/to/image.jpg` |

**Features:**
- Tests face recognition on individual images
- Applies policy configuration (action bands)
- Provides confidence scores and recommendations
- Supports JSON output for automation

**Options:**
- `--image`: Path to image file (required)
- `--threshold`: Custom similarity threshold (optional)
- `--json`: Output in JSON format (optional)
- `--policy`: Path to custom policy YAML file (optional)

**Example Output:**
```
Identity: john_doe
Score: 0.85
Band: Confident Accept
Action: Approve immediately
User Experience: < 2 second interaction
```

### Makefile Shortcuts

For convenience, common commands are aliased in the Makefile:

| Make Command | Equivalent Django Command | Description |
|--------------|---------------------------|-------------|
| `make setup` | `pip install && migrate` | Initial setup |
| `make run` | `python manage.py runserver` | Start dev server |
| `make test` | `python manage.py test` | Run all tests |
| `make evaluate` | `python manage.py eval` | Run evaluation |
| `make ablation` | `python manage.py ablation` | Run ablations |
| `make reproduce` | Multiple commands | Full reproducibility workflow |
| `make lint` | `black --check && isort --check && flake8` | Check code quality |
| `make format` | `black && isort` | Auto-format code |

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for detailed information on all commands and their options.
