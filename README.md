# Modern Smart Attendance System

This project is a fully refactored and modernized smart attendance system that leverages deep learning for face recognition. It provides a seamless and automated way to track employee attendance, eliminating the need for manual record-keeping. The system is built with a responsive web interface for a great user experience on any device.

![Home Page Light Theme](docs/images/home-light.png)

## Features

- **Automated Attendance:** Mark time-in and time-out effortlessly using real-time face recognition.
- **Responsive Web Interface:** A clean, modern, and intuitive UI that works beautifully on desktops, tablets, and mobile devices.
- **Admin Dashboard:** A powerful dashboard for administrators to manage employees, add user photos, and view comprehensive attendance reports.
- **Employee Dashboard:** A personalized dashboard for employees to view their own attendance records.
- **Automatic Training:** The face recognition model updates automatically when new employee photos are added.
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

## Documentation

For more detailed information, please refer to the full documentation:

- **[User Guide](USER_GUIDE.md)**: A comprehensive guide for non-programmers on using and understanding the system.
- **[Developer Guide](DEVELOPER_GUIDE.md)**: Information for developers on the system's architecture, evaluation pipeline, and management commands.
- **[Contributing Guide](CONTRIBUTING.md)**: Instructions for setting up the development environment and contributing to the project.
- **[API Reference](API_REFERENCE.md)**: Details on URL patterns, API endpoints, and command-line tools.
- **[Architecture Overview](ARCHITECTURE.md)**: A high-level overview of the system architecture and data flows.
- **[Data Card](DATA_CARD.md)**: Comprehensive documentation on the dataset, including privacy policies and data splits.

## Deployment Configuration

When deploying to staging or production, configure the following environment variables so that session cookies remain secure and expire after periods of inactivity. Boolean values accept `1`, `true`, `yes`, or `on` (case-insensitive).

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

3.  **Production deployments:** Run `python manage.py migrate` as part of the release pipeline after setting the new database variables. Review logs for schema drift and keep a recent backup of the managed Postgres instance before upgrading.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
