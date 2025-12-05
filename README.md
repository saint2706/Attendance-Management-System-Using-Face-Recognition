# Attendance Management System Using Face Recognition

> **A production-ready, privacy-first attendance solution that just works.**

**Attendance-Management-System-Using-Face-Recognition** is a modern, full-stack solution for automated time-tracking. It combines state-of-the-art deep learning (FaceNet) with a robust Django backend to deliver a seamless experience for employees and admins alike.

![Home Page Light Theme](docs/screenshots/home-light.png)

## Why this project?

- **🚀 Zero-Touch Attendance:** Employees just walk up to the camera. No badges, no PINs.
- **🔒 Privacy First:** Face data is encrypted at rest. Liveness detection prevents spoofing.
- **📱 Any Device:** Responsive PWA design works on tablets, mobiles, and desktops.
- **⚡ Production Ready:** Dockerized, tested, and scalable with Redis + Celery.

## Key Features

- **Real-time Recognition:** Instant identification using the efficient "Facenet" model.
- **Anti-Spoofing:** Two-stage liveness detection (motion + texture) rejects photos and screens.
- **Smart Dashboard:** Comprehensive admin panel for reports, user management, and system health.
- **Offline Capable:** PWA architecture ensures basic functionality even with spotty internet.
- **Automated Training:** The model retrains itself in the background when new users are added.

## Technical Stack

- **Backend:** Django 5+ with Celery workers for async training/evaluation jobs
- **Face Recognition:** DeepFace (Facenet) + SSD detector with a motion-based liveness gate
- **Frontend:** HTML5, CSS3, Bootstrap 5, Custom CSS Design System (installable PWA)
- **JavaScript:** Vanilla JS (no framework dependencies)
- **Database & cache:** Configurable via `DATABASE_URL` (PostgreSQL recommended; SQLite for local development) and Redis for Celery/async queues
- **Observability:** Sentry integration plus Silk for request profiling
- **Testing & CI:** Pytest with coverage + Playwright UI checks, executed in GitHub Actions

## Docker Image

Pre-built Docker images are automatically published to GitHub Container Registry (GHCR) on every push to `main` and on tagged releases.

### Quick Start with Docker

```bash
# Pull the latest image
docker pull ghcr.io/saint2706/attendance-management-system-using-face-recognition:latest

# Run with minimal configuration
docker run -d \
  -p 8000:8000 \
  -e DJANGO_SECRET_KEY="your-secret-key" \
  -e DATA_ENCRYPTION_KEY="your-fernet-key" \
  -e FACE_DATA_ENCRYPTION_KEY="your-fernet-key" \
  -e DJANGO_ALLOWED_HOSTS="localhost,127.0.0.1" \
  ghcr.io/saint2706/attendance-management-system-using-face-recognition:latest
```

### Available Tags

| Tag | Description |
|-----|-------------|
| `latest` | Latest build from `main` branch |
| `main-<sha>` | Specific commit from `main` branch (e.g., `main-abc1234`) |
| `vX.Y.Z` | Semantic versioned release (e.g., `v1.0.0`) |
| `X.Y.Z` | Semantic versioned release without `v` prefix |
| `sha-<sha>` | Specific commit SHA |

### Using Docker Compose

For a complete local setup with PostgreSQL, Redis, and Celery workers:

```bash
# Clone the repository
git clone https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition.git
cd Attendance-Management-System-Using-Face-Recognition

# Create environment file
cp .env.example .env
# Edit .env with your secrets (see Configuration section below)

# Start all services
docker compose up -d

# Apply database migrations
docker compose exec web python manage.py migrate

# Create admin user
docker compose exec web python manage.py createsuperuser
```

The application will be available at `http://localhost:8000`.

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `DJANGO_SECRET_KEY` | Django secret key (generate with `secrets.token_urlsafe(50)`) |
| `DATA_ENCRYPTION_KEY` | Fernet key for data encryption |
| `FACE_DATA_ENCRYPTION_KEY` | Fernet key for face data encryption |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated list of allowed hosts |

For complete deployment instructions including production setup, SSL configuration, and Nginx reverse proxy, see the [Deployment Guide](DEPLOYMENT.md).

## Beginner Setup (fast demo, no prior experience required)

This section shows the **quickest way** to run a local demo of the system and explore its main screens. You do not need prior knowledge of Python, Django, Docker, or virtual environments—just follow the numbered steps below.

> **⚠️ Demo only:** This setup is designed for exploring features, not for production use. For real deployments, see the [Deployment Guide](DEPLOYMENT.md) and [Security & Compliance Guide](docs/security.md).

### What you will be able to do

- Run a local demo with synthetic data and pre-created accounts.
- Log in and explore the dashboard, attendance session, and reports screens.
- Understand how the face recognition workflow operates.

### Step-by-step instructions

### 1. Install Python

Download and install Python 3.12 or newer from [python.org](https://www.python.org/downloads/). During installation on Windows, check "Add Python to PATH".

### 2. Download this repository

Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run:

```bash
git clone https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition.git
cd Attendance-Management-System-Using-Face-Recognition
```

> **Tip:** If you do not have Git installed, download the repository as a ZIP from GitHub and extract it.

### 3. Create a virtual environment (isolates project dependencies)

```bash
python -m venv venv
```

This creates a folder called `venv` that keeps project libraries separate from your system.

### 4. Activate the virtual environment

- **macOS / Linux:**

  ```bash
  source venv/bin/activate
  ```

- **Windows (Command Prompt):**

  ```cmd
  venv\Scripts\activate
  ```

- **Windows (PowerShell):**

  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

You should see `(venv)` at the start of your terminal prompt.

### 5. Install project dependencies

```bash
pip install -r requirements.txt
```

This downloads all the libraries the project needs. It may take a few minutes.

### 6. Set up environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

> On Windows Command Prompt, use `copy .env.example .env` instead.

The default values in `.env.example` are pre-configured for local demos.

### 7. Bootstrap the demo (creates database, sample data, and demo accounts)

```bash
make demo
```

> **Windows note:** If `make` is not available, run these commands instead:
>
> ```bash
> python manage.py migrate --noinput
> python scripts/bootstrap_demo.py
> ```

This command:

- Creates the database tables.
- Generates synthetic face images for three demo users.
- Creates a demo admin account and three demo employee accounts.

### 8. Start the development server

```bash
python manage.py runserver
```

You should see output like:

```text
Starting development server at http://127.0.0.1:8000/
```

### 9. Open the app in your browser

Navigate to <http://127.0.0.1:8000/> in your web browser.

### Demo login credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | `demo_admin` | `demo_admin_pass` |
| Employee 1 | `user_001` | `demo_user_pass` |
| Employee 2 | `user_002` | `demo_user_pass` |
| Employee 3 | `user_003` | `demo_user_pass` |

### What you should see

1. **Home page** – The landing page with "Mark Time-In", "Mark Time-Out", and "Dashboard Login" cards.
2. **Login page** – Click "Dashboard Login" and enter the demo admin credentials above.
3. **Admin dashboard** – A first-run checklist (if no employees are registered) and action cards for registering employees, adding photos, viewing reports, and more.
4. **Attendance session** – Navigate to "Attendance Session" to see the live recognition feed and recent attendance logs.
5. **Reports** – Click "View Attendance" to see attendance records by date or employee.

Screenshots of these screens are available in the [Screenshots](#screenshots) section below and in the [User Guide](USER_GUIDE.md).

### Stopping the demo

- Press `Ctrl+C` in the terminal to stop the development server.
- To deactivate the virtual environment, run: `deactivate`

### Next steps

- **Explore further:** Try registering a new employee and adding photos via the webcam.
- **Read the User Guide:** See [USER_GUIDE.md](USER_GUIDE.md) for detailed instructions on all features.
- **Production deployment:** When ready to deploy for real use, follow the [Deployment Guide](DEPLOYMENT.md) and [Security & Compliance Guide](docs/security.md).

## 10-Minute Demo Script (for Stakeholders)

Use this script to demonstrate the system's core value in a presentation or meeting.

**Prerequisite:** Follow the [Beginner Setup](#beginner-setup-fast-demo-no-prior-experience-required) above to get the app running.

### 1. The "Zero-Touch" Experience (2 mins)

- **Action:** Go to the Home Page. Click "Attendance Session" (login as `demo_admin` if prompted).
- **Narrative:** "This is the kiosk view. In a real office, this runs on a tablet at the entrance."
- **Action:** Show the live camera feed. (If you have a webcam, try to check in. If not, explain: "It detects faces in real-time.")
- **Highlight:** Point out the "Liveness: Active" indicator. "We use advanced anti-spoofing so you can't cheat with a photo."

### 2. The Admin Power View (3 mins)

- **Action:** Go to **Dashboard**.
- **Narrative:** "Managers get a complete bird's-eye view."
- **Action:** Click **System Health**.
- **Highlight:** Show the "Model Status" and "Dataset" cards. "The system monitors its own health and retrains automatically when new employees are added."

### 3. Onboarding a New Employee (3 mins)

- **Action:** Go to **Register Employee**.
- **Input:** Create a user (e.g., `alice_wonder`, Role: `Employee`).
- **Action:** Click **Add Photos**.
- **Narrative:** "Onboarding is instant. We take 5-10 snapshots right here, and the system learns their face immediately."
- **Note:** If you don't have a camera, explain that the system uses the pre-loaded synthetic data for the demo users (`user_001`, etc.).

### 4. Reporting & Analytics (2 mins)

- **Action:** Go to **Reports** -> **Attendance by Date**.
- **Action:** Select today's date.
- **Narrative:** "Payroll is automated. We see exact clock-in/out times and total hours calculated instantly."

### 5. Conclusion

- "It's secure, privacy-focused, and requires zero manual data entry."

## Screenshots

Here are some key screens from the application:

### Home Page

![Home page](docs/screenshots/home-light.png)

*The landing page where users can mark attendance or access the dashboard.*

### Login Page

![Login page](docs/screenshots/login.png)

*The secure login page for accessing dashboards and admin features.*

### Admin Dashboard

![Admin dashboard](docs/screenshots/admin-dashboard-light.png)

*The admin dashboard showing the first-run checklist and quick actions for managing employees and viewing reports.*

## Releases & Versioning

This project follows [Semantic Versioning](https://semver.org/).

- **Stable:** The `main` branch is always stable and deployable.
- **Tags:** Releases are tagged (e.g., `v1.0.0`).
- **Docker:** Images are automatically published to GHCR with matching tags.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- A webcam for face recognition

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition.git
    cd Attendance-Management-System-Using-Face-Recognition
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure environment variables:**
    - Copy `.env.example` to `.env`.
    - Generate secrets if you don't already have them:

      ```bash
      python - <<'PY'
      from cryptography.fernet import Fernet
      import secrets

      print('DJANGO_SECRET_KEY=', secrets.token_urlsafe(50))
      print('DATA_ENCRYPTION_KEY=', Fernet.generate_key().decode())
      print('FACE_DATA_ENCRYPTION_KEY=', Fernet.generate_key().decode())
      PY
      ```

    - Paste the values into `.env` and keep the same keys across runs so encrypted face data remains readable locally.
    - (Optional) Start the bundled Postgres service if you want to run against PostgreSQL instead of SQLite:

      ```bash
      docker compose up -d postgres
      ```

5. **Run database migrations:**

    ```bash
    python manage.py migrate
    ```

6. **Create a superuser (admin account):**

    ```bash
    python manage.py createsuperuser
    ```

    Follow the prompts to create your admin username, email, and password.

7. **Run the development server:**

    ```bash
    python manage.py runserver
    ```

    The application will be available at `http://127.0.0.1:8000/`.

8. **Collect static assets (including the PWA shell) before packaging builds:**

    ```bash
    python manage.py collectstatic
    ```

    This ensures the generated icons, `manifest.json`, and `sw.js` are published alongside the rest of the static files when you deploy with WhiteNoise or another static file server.

## Quick demo (synthetic data)

Use the bundled helper to generate an **encrypted synthetic** dataset and matching demo accounts:

```bash
make demo
```

This will:

- Apply migrations.
- Generate synthetic avatars into `sample_data/face_recognition_data/training_dataset/` and copy them to `face_recognition_data/training_dataset/` so the runtime pipeline can load them immediately.
- Create a demo superuser (`demo_admin` / `demo_admin_pass`).
- Create three matching demo users (`user_001`, `user_002`, `user_003`), all with password `demo_user_pass`.

Start the server with `python manage.py runserver` and sign in with the demo credentials above. The synthetic dataset is fully encrypted with the configured `DATA_ENCRYPTION_KEY`, so embeddings and caching behave the same as production assets.

Prefer to inspect or regenerate the dataset manually? Run `python scripts/bootstrap_demo.py --help` for options, or consult [sample_data/README.md](sample_data/README.md) for a deeper walkthrough of how the encrypted JPEGs are produced and reused across demo/test runs.

### Run a live attendance session

- Visit **Dashboard → Attendance Session** to view a structured, auto-refreshing log of recent recognition attempts with liveness outcomes and match confidence.
- The first-run checklist on the dashboard will prompt you to register an employee, add photos, and train the model if any prerequisites are missing.
- Use the check-in/check-out controls on the session page to start the same webcam-based flow used elsewhere in the app while keeping an eye on the live feed.

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
- **[Developer Guide](DEVELOPER_GUIDE.md)**: Information for developers on the system's architecture, evaluation pipeline, management commands, and environment configuration (including encryption keys).
- **[Contributing Guide](CONTRIBUTING.md)**: Instructions for setting up the development environment and contributing to the project.
- **[API Reference](API_REFERENCE.md)**: Details on URL patterns, API endpoints, and command-line tools.
- **[Architecture Overview](ARCHITECTURE.md)**: A high-level overview of the system architecture and data flows.
- **[Data Card](DATA_CARD.md)**: Comprehensive documentation on the dataset, including privacy policies and data splits.
- **[Liveness Evaluation](docs/liveness_evaluation.md)**: Methodology and results for the new motion-based anti-spoofing stage plus guidance for running `manage.py evaluate_liveness` locally.
- **[Deployment Guide](DEPLOYMENT.md)**: Step-by-step instructions for building the Docker image, configuring Compose services, managing environment variables, and hardening production deployments.
- **[Fairness & Limitations](docs/FAIRNESS_AND_LIMITATIONS.md)**: Methodology, findings, and follow-up actions for the fairness and robustness audit plus guidance on how to rerun it locally.

## Reproducibility

The repository can generate a tiny, synthetic dataset under
`sample_data/face_recognition_data/training_dataset/` so reviewers can exercise
the full recognition pipeline without requesting encrypted production assets.
Run the following command after installing dependencies to regenerate the
metrics referenced in the documentation (the dataset will be created on the fly if missing):

```bash
make reproduce
```

The target launches `scripts/reproduce_sample_results.py`, which points the
evaluation harness at the bundled dataset, seeds the random number generators,
and saves artifacts (metrics JSON, confusion matrix, per-sample CSV) to
`reports/sample_repro/`. You can supply your own dataset or split CSV by
invoking the script directly:

```bash
python scripts/reproduce_sample_results.py --dataset-root /path/to/your/data --split-csv custom.csv
```

The script overwrites neither `face_recognition_data/` nor production caches; it
patches the dataset root in-memory so regular deployments remain unchanged.

## Limitations & Responsible Use

Face biometrics introduce safety and ethical constraints that require explicit monitoring. Run `python manage.py fairness_audit --split-csv reports/splits.csv --reports-dir reports/fairness` whenever the dataset changes to capture per-role, per-site, per-source, and per-lighting metrics. The resulting tables in `reports/fairness/summary.md` highlight buckets where the False Acceptance Rate (FAR) or False Rejection Rate (FRR) deviates from the global average. Review the [Fairness & Limitations](docs/FAIRNESS_AND_LIMITATIONS.md) report alongside the [DATA_CARD.md](DATA_CARD.md) before rolling out models to new locations or populations.

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
- **In-app health**: Use the Django admin System Health page to confirm camera connectivity, dataset freshness, model recency, recent recognition outcomes, and Celery worker reachability without leaving the UI.

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

1. **Local development:**
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

2. **Continuous Integration:** Configure the CI job to export `DATABASE_URL` (for example, `postgres://postgres:postgres@localhost:5432/postgres`) before invoking `pytest` so the same migrations and tests execute against Postgres automatically.
    - Coverage is enforced in CI; the suite fails if overall coverage drops below 60%, and the Codecov badge at the top of this README reflects the latest run.

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

## Face-Matching Metric

The recognition pipeline compares embeddings with cosine similarity:

- **Similarity:** `sim(A, B) = (A · B) / (||A|| ||B||)`
- **Cosine distance:** `d(A, B) = 1 − sim(A, B)`

Attendance is accepted when the cosine distance is less than or equal to `RECOGNITION_DISTANCE_THRESHOLD`, which defaults to **0.4** in this repository. Tightening the threshold reduces false accepts while loosening it mitigates false rejects. The evaluation harness (`python manage.py eval` or `make evaluate`) sweeps a configurable range via `--threshold-start/stop/step` and records FAR/FRR trade-offs in `reports/evaluation/threshold_sweep.csv`, making it easy to justify any threshold adjustment before shipping.

1. **Production deployments:** Run `python manage.py migrate` as part of the release pipeline after setting the new database variables. Review logs for schema drift and keep a recent backup of the managed Postgres instance before upgrading.

## Contributing

We welcome contributions from the community! Whether you're fixing a bug, improving documentation, or proposing new features, your help is appreciated.

- **[Contributing Guide](CONTRIBUTING.md)**: How to set up your development environment and submit changes
- **[Code of Conduct](CODE_OF_CONDUCT.md)**: Our community standards and expectations
- **[Support](SUPPORT.md)**: How to get help and where to ask questions
- **[Security Policy](SECURITY.md)**: How to report security vulnerabilities

### Quick Start for Contributors

1. Fork the repository
2. Clone your fork and set up the development environment (see [CONTRIBUTING.md](CONTRIBUTING.md))
3. Create a feature branch
4. Make your changes with tests
5. Submit a pull request

Look for issues labeled [`good first issue`](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/labels/good%20first%20issue) if you're new to the project.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
