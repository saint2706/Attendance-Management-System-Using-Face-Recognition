# Modern Smart Attendance System

This project is a fully refactored and modernized smart attendance system that leverages deep learning for face recognition. It provides a seamless and automated way to track employee attendance, eliminating the need for manual record-keeping. The system is built with a responsive web interface for a great user experience on any device.

![Home Page Light Theme](docs/images/home-light.png)

## Features

- **Automated Attendance:** Mark time-in and time-out effortlessly using real-time face recognition.
- **Responsive Web Interface:** A clean, modern, and intuitive UI that works beautifully on desktops, tablets, and mobile devices.
- **Admin Dashboard:** A powerful dashboard for administrators to manage employees, add user photos, and view comprehensive attendance reports.
- **Employee Dashboard:** A personalized dashboard for employees to view their own attendance records.
- **Automatic Training:** The face recognition model updates automatically when new employee photos are added.
- **Offline-ready experience:** Installable progressive web app with background sync for attendance submissions and cached UI shell.
- **Performance Optimized:** Utilizes the efficient "Facenet" model and "SSD" detector for a fast and responsive recognition experience.
- **Continuous Integration:** Includes a GitHub Actions workflow to automatically run tests, ensuring code quality and stability.

## Technical Stack

- **Backend:** Django 5+
- **Face Recognition:** DeepFace (wrapping Facenet)
- **Frontend:** HTML5, CSS3, Bootstrap 5, Custom CSS Design System
- **JavaScript:** Vanilla JS (no framework dependencies)
- **Database:** Configurable via `DATABASE_URL` (PostgreSQL recommended; falls back to SQLite for local development)
- **Testing:** Django's built-in test framework, Playwright (planned)

## Getting Started

### Prerequisites

- Python 3.12 or higher
- A webcam for face recognition

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/smart-attendance-system.git
    cd smart-attendance-system
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    - Copy `.env.example` to `.env`.
    - (Optional) Start the bundled Postgres service if you want to run against PostgreSQL instead of SQLite:
      ```bash
      docker compose up -d postgres
      ```

5.  **Run database migrations:**
    ```bash
    python manage.py migrate
    ```

6.  **Create a superuser (admin account):**
    ```bash
    python manage.py createsuperuser
    ```
    Follow the prompts to create your admin username, email, and password.

7.  **Run the development server:**
    ```bash
    python manage.py runserver
    ```
    The application will be available at `http://127.0.0.1:8000/`.

8.  **Collect static assets (including the PWA shell) before packaging builds:**
    ```bash
    python manage.py collectstatic
    ```
    This ensures the generated icons, `manifest.json`, and `sw.js` are published alongside the rest of the static files when you deploy with WhiteNoise or another static file server.

## Performance Monitoring

Silk is bundled to profile database queries, view timings, and cache usage without leaving the Django admin. The dependency is already pinned in `requirements.txt`/`pyproject.toml`, so installing the project requirements pulls it in automatically.

1. **Apply Silk migrations** after installing dependencies any time new environments are set up:
   ```bash
   python manage.py migrate
   ```
   This creates the `silk_*` tables used to persist profiling results.
2. **Accessing the dashboard:**
   - In development (`DJANGO_DEBUG=1`) visit `http://127.0.0.1:8000/silk/` to inspect live profiles.
   - In non-debug deployments Silk requires authentication and staff status. Log in with a staff or superuser account before visiting `/silk/`; non-staff users receive a permission error and unauthenticated visitors are redirected to the login page.
3. **Production guardrails:** keep the middleware enabled only when you actively need profiling, and clear the Silk tables regularly in long-running environments to manage database size.

## Documentation

For more detailed information, please refer to the full documentation:

- **[User Guide](USER_GUIDE.md)**: A comprehensive guide for non-programmers on using and understanding the system.
- **[Developer Guide](DEVELOPER_GUIDE.md)**: Information for developers on the system's architecture, evaluation pipeline, and management commands.
- **[Contributing Guide](CONTRIBUTING.md)**: Instructions for setting up the development environment and contributing to the project.
- **[API Reference](API_REFERENCE.md)**: Details on URL patterns, API endpoints, and command-line tools.
- **[Architecture Overview](ARCHITECTURE.md)**: A high-level overview of the system architecture and data flows.
- **[Data Card](DATA_CARD.md)**: Comprehensive documentation on the dataset, including privacy policies and data splits.
- **[Deployment Guide](docs/deployment-guide.md)**: Step-by-step instructions for building the Docker image, configuring Compose services, managing environment variables, and hardening production deployments.

## Deployment Configuration

When deploying to staging or production, configure the following environment variables so that session cookies remain secure and expire after periods of inactivity. Boolean values accept `1`, `true`, `yes`, or `on` (case-insensitive). Use the development settings module (`attendance_system_facial_recognition.settings`) locally and for automated tests. Production deployments should set `DJANGO_SETTINGS_MODULE=attendance_system_facial_recognition.settings.production` so the hardened database configuration is loaded.

Review the [Security & Compliance Guide](docs/security.md) for secret management, HTTPS/SSL hardening, and operational checklists that build on these deployment notes.

The Progressive Web App resources are exposed at `/manifest.json` and `/sw.js`. Ensure these paths are routed to Django so the manifest and service worker can be cached by browsers during install.

| Environment variable | Purpose | Recommended staging value | Recommended production value |
| --- | --- | --- | --- |
| `DATABASE_URL` | Connection string parsed with [`dj-database-url`](https://github.com/jazzband/dj-database-url). | `postgres://user:pass@db:5432/attendance` | `postgres://user:pass@db:5432/attendance` |
| `DATABASE_CONN_MAX_AGE` | Persistent connection lifetime in seconds (`0` disables pooling). | `60` | `600` |
| `DATABASE_SSL_REQUIRE` | Force `sslmode=require` for managed Postgres providers. | `false` | `true` |
| `DJANGO_SESSION_COOKIE_SECURE` | Send the session cookie only over HTTPS. | `true` | `true` |
| `DJANGO_SESSION_COOKIE_HTTPONLY` | Prevent client-side scripts from reading the session cookie. | `true` | `true` |
| `DJANGO_CSRF_COOKIE_SECURE` | Send the CSRF cookie only over HTTPS. | `true` | `true` |
| `DJANGO_SESSION_COOKIE_SAMESITE` | Restrict cross-site cookie usage. | `Lax` | `Lax` |
| `DJANGO_SESSION_COOKIE_AGE` | Maximum session age (seconds) before inactivity timeout. | `1800` (30 minutes) | `1800` (30 minutes) |
| `DJANGO_SESSION_EXPIRE_AT_BROWSER_CLOSE` | Drop the session when the browser closes. | `true` | `true` |

Ensure these variables are present in the staging and production deployment manifests (e.g., `.env` files, container secrets, or platform configuration) before rolling out new builds.

### Observability and error tracking

The production settings initialise [Sentry](https://sentry.io/) automatically when the DSN is supplied. Configure the following environment variables to tailor telemetry to each deployment target:

| Environment variable | Purpose | Recommended staging value | Recommended production value |
| --- | --- | --- | --- |
| `SENTRY_DSN` | Enables Sentry ingestion for the project. | `https://<public>@sentry.io/<project>` | `https://<public>@sentry.io/<project>` |
| `SENTRY_ENVIRONMENT` | Distinguishes environments inside the Sentry dashboards. | `staging` | `production` |
| `SENTRY_RELEASE` | Associates events with build artefacts for source maps and regression tracking. | Git commit SHA | Git tag or release identifier |
| `SENTRY_TRACES_SAMPLE_RATE` | Fraction (0.0–1.0) of requests captured for APM tracing. | `0.1` | `0.2` |
| `SENTRY_PROFILES_SAMPLE_RATE` | Fraction (0.0–1.0) of traces that include profiling data. | `0.0` | `0.05` |
| `SENTRY_SEND_DEFAULT_PII` | Toggle for sending user-identifiable attributes; defaults to scrubbing PII. | `false` | `false` |

With `SENTRY_SEND_DEFAULT_PII` disabled, the integration strips cookies, authorisation headers, and user attributes before dispatching events so that production telemetry complies with internal privacy requirements. Operators that need richer context can opt in by setting `SENTRY_SEND_DEFAULT_PII=true` after completing a privacy impact assessment.

#### Operational dashboards

- **Issues**: Monitor unhandled exceptions and message breadcrumbs from the Sentry *Issues* dashboard. Pin the view filtered by `environment:production` to quickly detect regressions after each deployment.
- **Performance**: Track request latency, throughput, and slow transactions with the *Performance* dashboard. Enable sampling via `SENTRY_TRACES_SAMPLE_RATE` to populate the charts and configure alerts for p95 latency regressions.
- **Real-time**: For incident response, use Sentry's *Releases* view to correlate deploys with spikes in error volume, and subscribe the operations channel to release health alerts.

### Containerized deployment workflow

The repository ships with a production-ready `Dockerfile` and Compose definition so you can build and run the stack with minimal host dependencies. The commands below assume Docker Engine 24+ and Docker Compose v2 are installed locally.

1. **Prepare environment variables**
   - Create a secrets file (for example, `.env.production`) that exports:
     - `DJANGO_SECRET_KEY` — a strong, unique secret key.
     - `DATA_ENCRYPTION_KEY` and `FACE_DATA_ENCRYPTION_KEY` — Fernet keys. Generate each with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.
     - `DJANGO_ALLOWED_HOSTS` — comma-separated hostnames served by the deployment.
     - Database credentials (`POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`) if you override the defaults.
   - Load the file automatically by placing it next to `docker-compose.yml` and referencing it with `docker compose --env-file .env.production ...`.

2. **Build the application image**
   ```bash
   docker compose --env-file .env.production build web
   ```

3. **Apply database migrations inside the container image**
   ```bash
   docker compose --env-file .env.production run --rm web python manage.py migrate
   ```

4. **Start the web and worker services**
   ```bash
   docker compose --env-file .env.production up -d web celery
   ```

5. **Tail logs for troubleshooting**
   ```bash
   docker compose logs -f web celery
   ```

The `web` service serves the Django application through Gunicorn on port `8000`, while the `celery` service reuses the same image to execute asynchronous jobs. Static assets are collected during the image build, so no additional setup is required before exposing the container behind a reverse proxy.

### Database migration & testing workflow

1.  **Local development:**
    - Copy `.env.example` to `.env` and adjust credentials.
    - Start the Postgres service from the provided Docker Compose profile:
      ```bash
      docker compose up -d postgres
      ```
    - Apply schema migrations against Postgres and run the Django test suite:
      ```bash
      python manage.py migrate
      pytest
      ```

2.  **Continuous Integration:** Configure the CI job to export `DATABASE_URL` (for example, `postgres://postgres:postgres@localhost:5432/postgres`) before invoking `pytest` so the same migrations and tests execute against Postgres automatically.

## Evaluation & Benchmarking

The repository ships with an evaluation harness that reuses the exact face-recognition pipeline deployed in production. It loads the encrypted dataset, computes embeddings through DeepFace, and aggregates the metrics required for referee-quality reports (accuracy, precision, recall, macro F1, FAR, FRR, confusion matrix, and a threshold sweep).

1. **Prepare dataset splits (optional but recommended):**
   ```bash
   python manage.py prepare_splits --seed 42
   ```
   This command writes `reports/splits.csv`, which identifies the test split consumed during evaluation. If the file is missing the evaluator falls back to scanning the entire `face_recognition_data/training_dataset/` tree.
2. **Run the evaluation:**
   ```bash
   python manage.py eval --split-csv reports/splits.csv
   ```
   The command accepts additional knobs such as `--threshold`, `--threshold-start/stop/step`, `--max-samples`, and `--reports-dir` for ad-hoc experiments. A convenience shortcut is also available via `make evaluate`.
3. **Inspect the reports:** artifacts are written to `reports/evaluation/`:
   - `metrics_summary.json` – accuracy/precision/recall/F1/FAR/FRR plus bookkeeping stats.
   - `sample_predictions.csv` – per-image ground truth, candidate match, distance, and predicted label.
   - `confusion_matrix.csv` and `confusion_matrix.png` – tabular and visual confusion matrices.
   - `threshold_sweep.csv` and `threshold_sweep.png` – FAR/FRR/accuracy/F1 for each distance threshold in the sweep.

Because the evaluator defers to the same dataset cache used during attendance marking, results remain reproducible and consistent with the live service.

3.  **Production deployments:** Run `python manage.py migrate` as part of the release pipeline after setting the new database variables. Review logs for schema drift and keep a recent backup of the managed Postgres instance before upgrading.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
