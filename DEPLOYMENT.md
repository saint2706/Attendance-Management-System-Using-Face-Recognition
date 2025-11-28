# Deployment Guide

This guide describes how to build, configure, and deploy the Attendance Management System Using Face Recognition using Docker and Docker Compose. It also covers running a local demo.

## Prerequisites

-   Docker Engine 24 or newer
-   Docker Compose v2
-   Python 3.12+ (for local execution)

## 1. Local Demo Environment

For a quick local demonstration with synthetic data:

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the demo bootstrap:**
    ```bash
    make demo
    ```
    This command will migrate the database, generate synthetic employee records, and create the following accounts:
    -   **Admin:** `demo_admin` / `demo_admin_pass`
    -   **Users:** `user_001`, `user_002`, `user_003` (password: `demo_user_pass`)

3.  **Launch the server:**
    Follow the on-screen instructions (typically `python manage.py runserver`).

## 2. Building the Docker Image

The `Dockerfile` in the project root builds a production-ready image. It uses a multi-stage build to create a lean final image with all the necessary dependencies.

To build the image, run the following command from the project root:

```bash
docker compose build
```

This will create an image named `attendance-system:latest` that will be used by the `web` and `celery` services.

## 3. Configuration

The application is configured using environment variables. Create a `.env.production` file in the project root with the following variables:

```
# Django settings
DJANGO_DEBUG=0
DJANGO_SECRET_KEY='your-strong-secret-key'
DJANGO_ALLOWED_HOSTS='your-domain.com,www.your-domain.com'
DATA_ENCRYPTION_KEY='your-data-encryption-key'
FACE_DATA_ENCRYPTION_KEY='your-face-data-encryption-key'

# Database credentials
POSTGRES_DB=attendance
POSTGRES_USER=attendance
POSTGRES_PASSWORD='your-secure-password'

# Production settings
DJANGO_SESSION_COOKIE_SECURE=True
DJANGO_CSRF_COOKIE_SECURE=True
```

**Note:** For a full list of available configuration variables, see the `.env.example` file.

## 4. Running with Docker (Staging/Production)

### Running Migrations

Before starting the application for the first time, you need to run the database migrations:

```bash
docker compose --env-file .env.production run --rm web python manage.py migrate
```

### Staging Environment

For staging or quality assurance, you can initialize the Docker stack with demo data:

1.  Start the services:
    ```bash
    docker compose --env-file .env.production up -d
    ```
2.  Initialize the demo data (creates admin and synthetic users):
    ```bash
    docker compose --env-file .env.production exec web python scripts/bootstrap_demo.py
    ```
3.  Access the application at `http://localhost:8000`.

### Starting the Services (Production)

To start the `web`, `celery`, `postgres`, and `redis` services for production usage, run:

```bash
docker compose --env-file .env.production up -d
```

The application will be available at `http://localhost:8000`.

## 5. Common Deployment Issues

-   **Missing Environment Variables:** The application will fail to start if any of the required environment variables are missing. Ensure that your `.env.production` file is complete and correctly formatted.
-   **Static Files Not Collected Correctly:** The `Dockerfile` runs `collectstatic` during the build process. If you are having issues with static files, ensure that the `DJANGO_SETTINGS_MODULE` is set to `attendance_system_facial_recognition.settings.production` in your `.env.production` file.
-   **Incorrect Database Host:** When running with Docker Compose, the database host is `postgres`. If you are deploying to a different environment, you will need to update the `DB_HOST` environment variable.
-   **Celery Worker Failing:** The Celery worker depends on Redis. Ensure that the Redis container is running before the Celery container starts.
-   **HTTPS Misconfiguration:** In a production environment, you should run the application behind a reverse proxy that handles HTTPS. Ensure that you have correctly configured your reverse proxy and have set `DJANGO_SESSION_COOKIE_SECURE=True` and `DJANGO_CSRF_COOKIE_SECURE=True` in your `.env.production` file.

## 6. PWA and Service Worker

The application is a Progressive Web App (PWA) and uses a service worker to cache assets and enable offline functionality. The service worker is served from the root of the application and is controlled by the `progressive_web_app_service_worker` view.

## 7. Reproducibility Smoke Test

Before promoting a new image, run the bundled synthetic evaluation to verify that DeepFace, OpenCV, and the encrypted embedding cache function correctly inside the target environment:

```bash
docker compose --env-file .env.production run --rm web make reproduce
```

Before running the full evaluation suite (`make evaluate`), prepare deterministic splits so staging and production use the exact same hold-out set:

```bash
docker compose --env-file .env.production run --rm web python manage.py prepare_splits --seed 42
docker compose --env-file .env.production run --rm web make evaluate
```

The command routes the evaluation pipeline through `sample_data/` instead of the encrypted dataset, so no customer photos are required. Review the metrics and artifacts saved under `reports/sample_repro/` to confirm the build is healthy before replacing production assets with the real `face_recognition_data/` volume.
