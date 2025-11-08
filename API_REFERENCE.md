# API Documentation

This document outlines all the URL patterns for the project and explains the purpose and functionality of each corresponding view.

## Core Pages

| URL Path      | View Function        | Name                 | Description                                                                                                                                                             |
|---------------|----------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `/`           | `recog_views.home`   | `home`               | Renders the home page of the application.                                                                                                                               |
| `/dashboard/` | `recog_views.dashboard`| `dashboard`          | Renders the dashboard, which differs for admins and regular employees.                                                                                                    |

## Authentication

| URL Path   | View Function/Class             | Name    | Description                                                                 |
|------------|---------------------------------|---------|-----------------------------------------------------------------------------|
| `/login/`  | `auth_views.LoginView`          | `login` | Displays the login page and handles user authentication.                      |
| `/logout/` | `auth_views.LogoutView`         | `logout`| Logs the user out and redirects them to the home page.                        |

## User and Photo Management (Admin-only)

| URL Path        | View Function           | Name           | Description                                                                                                                                                    |
|-----------------|-------------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `/register/`    | `users_views.register`  | `register`     | Allows staff members or superusers to register new employee accounts.                                                                                           |
| `/add_photos/`  | `recog_views.add_photos`| `add-photos`   | Handles the 'Add Photos' functionality for admins to create face datasets for users.                                                                           |
| `/train/`       | `recog_views.train`     | `train`        | **Obsolete.** This view is no longer used, as the training process is now automatic. It redirects to the dashboard with an informational message.                  |

## Evaluation and Analytics (Admin-only)

| URL Path            | View Function                      | Name                      | Description                                                                 |
|---------------------|------------------------------------|---------------------------|-----------------------------------------------------------------------------|
| `/admin/evaluation/`| `recog_admin_views.evaluation_dashboard` | `admin:evaluation_dashboard` | Displays comprehensive evaluation metrics, confidence intervals, performance visualizations, and links to detailed reports. Shows ROC AUC, EER, FAR/TPR metrics with bootstrap confidence intervals. |

## Face Recognition and Attendance Marking

| URL Path                      | View Function                      | Name                         | Description                                            |
|-------------------------------|------------------------------------|------------------------------|--------------------------------------------------------|
| `/mark_your_attendance`       | `recog_views.mark_your_attendance` | `mark-your-attendance`       | Handles marking time-in using face recognition.        |
| `/mark_your_attendance_out`   | `recog_views.mark_your_attendance_out` | `mark-your-attendance-out`   | Handles marking time-out using face recognition.       |

## Attendance Viewing

| URL Path                     | View Function                                 | Name                                  | Description                                                                                                                                         |
|------------------------------|-----------------------------------------------|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `/view_attendance_home`      | `recog_views.view_attendance_home`            | `view-attendance-home`                | Renders the main attendance viewing page for admins.                                                                                                |
| `/view_attendance_date`      | `recog_views.view_attendance_date`            | `view-attendance-date`                | Admin view to see attendance for all employees on a specific date.                                                                                   |
| `/view_attendance_employee`  | `recog_views.view_attendance_employee`        | `view-attendance-employee`            | Admin view to see attendance for a specific employee over a date range.                                                                             |
| `/view_my_attendance`        | `recog_views.view_my_attendance_employee_login` | `view-my-attendance-employee-login`   | Employee-specific view to see their own attendance over a date range.                                                                               |

## Error/Status Pages

| URL Path           | View Function              | Name                 | Description                                                       |
|--------------------|----------------------------|----------------------|-------------------------------------------------------------------|
| `/not_authorised`  | `recog_views.not_authorised` | `not-authorised`     | Renders a page for users trying to access unauthorized areas.     |


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
