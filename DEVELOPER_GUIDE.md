# Developer Guide

This guide provides a comprehensive overview of the project's structure, architecture, and advanced usage. It is intended for developers who want to understand the project's inner workings. For information on how to contribute, please see the [Contributing Guide](CONTRIBUTING.md).

## 1. Project Structure

The project is organized into the following directories:

-   `attendance_system_facial_recognition`: The main Django project directory.
-   `recognition`: The Django app that handles face recognition and attendance tracking.
-   `users`: The Django app that handles user management.
-   `face_recognition_data`: The directory where the face recognition data is stored.

## 2. Architecture Overview

For a detailed overview of the system architecture, including diagrams and data flow, please see the [Architecture Overview](ARCHITECTURE.md) document.

## 3. Management Commands

The project includes several custom Django management commands for evaluation and analysis.

### Data Preparation

To prepare the data for training and evaluation, use the `prepare_splits` command:

```bash
python manage.py prepare_splits --seed 42
```

### Evaluation

To run a comprehensive evaluation of the model, use the `eval` command:

```bash
python manage.py eval --seed 42 --n-bootstrap 1000
```

### Threshold Selection

To select the optimal recognition threshold, use the `threshold_select` command:

```bash
python manage.py threshold_select --method eer --seed 42
```

### Ablation Experiments

To run ablation experiments, use the `ablation` command:

```bash
python manage.py ablation --seed 42
```

### Export Reports

To export all generated reports and figures, use the `export_reports` command:

```bash
python manage.py export_reports
```

## 4. Makefile Targets

The project includes a comprehensive `Makefile` for common development tasks.

### Development

-   `make run`: Start the Django development server
-   `make migrate`: Run database migrations

### Testing and Evaluation

-   `make test`: Run all Django tests
-   `make evaluate`: Run performance evaluation with metrics
-   `make ablation`: Run ablation experiments
-   `make report`: Generate comprehensive reports

## 5. API Reference

For a complete reference of all API endpoints and command-line tools, please see the [API Reference](API_REFERENCE.md).

## 6. Configuration

The system can be configured using environment variables. For a detailed list of all configuration options, please see the [main README file](README.md#deployment-configuration).

## 7. Database backends

### Local PostgreSQL with Docker Compose

1. Copy `.env.example` to `.env` and update the secrets as required. The default values match the credentials exported by the Compose profile.
2. Start the containerised database:
   ```bash
   docker compose up -d postgres
   ```
3. Run migrations and tests against the running Postgres instance:
   ```bash
   python manage.py migrate
   pytest
   ```
4. Tear the database down when you are finished:
   ```bash
   docker compose down
   ```

### Continuous Integration

CI pipelines must export `DATABASE_URL` before running `pytest` so Django connects to Postgres instead of the default SQLite fallback. A typical GitHub Actions job includes a Postgres service container and the following step:

```yaml
- name: Run tests
  env:
    DATABASE_URL: postgresql://postgres:postgres@localhost:5432/postgres
  run: pytest
```

Running tests against Postgres ensures migrations stay compatible with the production backend and catches issues that do not appear with SQLite.
