# Deployment Guide

This guide describes how to package the Attendance Management System into a Docker image, configure the Docker Compose services, provide the required environment variables, and run database migrations for production-like environments.

## Prerequisites

- Docker Engine 24 or newer
- Docker Compose v2 (bundled with recent Docker Engine releases)
- Access to generate required secrets (for example, Fernet keys for data encryption)

## Build the application image

The repository ships with a `Dockerfile` that builds a production-ready image using Python 3.12, installs system packages required by OpenCV/DeepFace, installs Python dependencies from `requirements.txt`, and collects static assets via the production Django settings module. Build the image locally with Compose so the `web` and `celery` services share the same artifact:

```bash
docker compose --env-file .env.production build web
```

This command uses the `build` section defined on the `web` service in `docker-compose.yml`, caching layers appropriately and tagging the result as `attendance-system:latest`.

## Configure Compose services

The Compose definition includes four services:

- **postgres** – Runs `postgres:16-alpine` and seeds a database, user, and password via environment variables. The service persists data in the `postgres_data` named volume and publishes the configured port (`POSTGRES_PORT`, default `5432`).
- **redis** – Runs `redis:7-alpine`, persists data in the `redis_data` named volume, and exposes the configured port (`REDIS_PORT`, default `6379`).
- **web** – Builds from the project context using the Dockerfile, runs Gunicorn bound to `0.0.0.0:8000`, depends on Postgres and Redis, and exposes port `WEB_PORT` (default `8000`).
- **celery** – Reuses the `attendance-system:latest` image, starts the Celery worker entrypoint, and depends on Postgres and Redis.

When you start the stack, Compose automatically provisions the `postgres_data` and `redis_data` volumes to keep database state and Redis keys across container recreations.

## Seed environment variables

Create an environment file (for example, `.env.production`) next to `docker-compose.yml` and populate the secrets and configuration toggles required by the application. At minimum, export:

- `DJANGO_SECRET_KEY` – A strong, unique secret key for the Django instance.【F:README.md†L118-L122】
- `DATA_ENCRYPTION_KEY` and `FACE_DATA_ENCRYPTION_KEY` – Fernet keys used to encrypt stored data and biometric templates.【F:README.md†L122-L124】
- `DJANGO_ALLOWED_HOSTS` – Comma-separated hostnames that serve the application.【F:README.md†L124-L124】
- `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` – Database credentials if you do not want the defaults baked into the Compose file.【F:README.md†L124-L126】
- Production hardening variables such as `DJANGO_SESSION_COOKIE_SECURE`, `DJANGO_SESSION_COOKIE_HTTPONLY`, `DJANGO_CSRF_COOKIE_SECURE`, `DJANGO_SESSION_COOKIE_SAMESITE`, and `DJANGO_SESSION_EXPIRE_AT_BROWSER_CLOSE` to align with the recommendations in the deployment configuration table.【F:README.md†L134-L150】

Reference additional settings like `DATABASE_URL`, `DATABASE_CONN_MAX_AGE`, and `DATABASE_SSL_REQUIRE` from the deployment configuration section when you deploy to managed database providers.【F:README.md†L130-L140】 Load the file automatically with `docker compose --env-file .env.production <command>` so every Compose invocation receives the same variables.

## Run database migrations

After building the image and exporting the environment variables, apply schema migrations using the `web` service image:

```bash
docker compose --env-file .env.production run --rm web python manage.py migrate
```

This launches a one-off container using the production settings module (`attendance_system_facial_recognition.settings.production`) configured in the Dockerfile and Compose file, ensuring migrations run against the same code that serves traffic.

## Start the services

Bring the web application and Celery worker online once migrations complete:

```bash
docker compose --env-file .env.production up -d web celery
```

The `web` container listens on port `8000` inside the Compose network and maps to the host port defined by `WEB_PORT`. Tail logs for troubleshooting or health checks as needed:

```bash
docker compose logs -f web celery
```

## Production hardening checklist

Beyond the base Compose stack, review the following practices before promoting a deployment to production:

1. **TLS termination and reverse proxy** – Place a hardened reverse proxy (for example, Nginx, Caddy, or a cloud load balancer) in front of the `web` service. Configure HTTPS, HTTP/2, request buffering, and static asset caching. Terminate TLS at the proxy and forward traffic to the Gunicorn container on the internal network.
2. **Secrets management** – Store sensitive variables such as `DJANGO_SECRET_KEY`, `DATA_ENCRYPTION_KEY`, database credentials, and Fernet keys in a secure secret manager (Docker Swarm/Kubernetes secrets, HashiCorp Vault, AWS Secrets Manager) instead of plaintext files. Inject them into the container at runtime.
3. **Persistent storage** – Use managed storage (for example, cloud block storage) or bind mounts for `postgres_data` and `redis_data` so container recreation does not wipe state. Back up the Postgres volume regularly.
4. **Database connectivity** – Set `DATABASE_SSL_REQUIRE=true` and adjust `DATABASE_URL` to point at the managed database endpoint when running outside the Compose network. Tune connection pooling via `DATABASE_CONN_MAX_AGE` based on workload.【F:README.md†L130-L137】
5. **Session security** – Ensure `DJANGO_SESSION_COOKIE_SECURE`, `DJANGO_SESSION_COOKIE_HTTPONLY`, `DJANGO_CSRF_COOKIE_SECURE`, `DJANGO_SESSION_COOKIE_SAMESITE`, `DJANGO_SESSION_COOKIE_AGE`, and `DJANGO_SESSION_EXPIRE_AT_BROWSER_CLOSE` are configured as recommended to protect user sessions.【F:README.md†L134-L150】
6. **Static asset delivery** – Serve the `/static/` files collected during the image build through the reverse proxy or a CDN. Configure long-lived cache headers and invalidation rules that fit your deployment.
7. **Monitoring and logging** – Forward Gunicorn, Celery, Postgres, and Redis logs to a centralized log aggregator. Add uptime monitoring and application health checks (for example, `/health/` endpoints or Celery worker heartbeats).
8. **Background workers** – Scale the `celery` service replicas based on queue depth. Configure the broker (`CELERY_BROKER_URL`) and result backend (`CELERY_RESULT_BACKEND`) to use managed Redis or another durable backend in production.
9. **Upgrades and rollbacks** – Tag images with semantic versions instead of `latest` once you move to production. Maintain migration scripts and test upgrades in staging before promoting to production.

Following this checklist ensures the Dockerized deployment remains resilient, secure, and maintainable as you move from local testing to production.
