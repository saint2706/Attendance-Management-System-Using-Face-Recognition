# Developer Guide

This guide provides technical details for developers working on the Attendance Management System.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [Backend Architecture](#backend-architecture)
- [Frontend Architecture](#frontend-architecture)
- [Face Recognition Pipeline](#face-recognition-pipeline)
- [Asynchronous Tasks with Celery](#asynchronous-tasks-with-celery)
- [Logging and Error Handling](#logging-and-error-handling)
- [Testing](#testing)
- [Deployment](#deployment)

## Project Structure

The project is organized into several Django apps:

- `attendance_system_facial_recognition`: The main Django project.
- `recognition`: The core app for face recognition, attendance marking, and views.
- `users`: Manages user authentication and registration.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/Attendance-Management-System-Using-Face-Recognition.git
    cd Attendance-Management-System-Using-Face-Recognition
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Apply migrations:**
    ```bash
    python manage.py migrate
    ```
2.  **Create a superuser:**
    ```bash
    python manage.py createsuperuser
    ```
3.  **Run the development server:**
    ```bash
    python manage.py runserver
    ```
4.  **Run the Celery worker:**
    In a separate terminal, run:
    ```bash
    celery -A attendance_system_facial_recognition worker -l info
    ```

## Backend Architecture

The backend is built with Django 5. Key components include:

- **Models:** `users/models.py` and `recognition/models.py` define the database schema.
- **Views:** `recognition/views.py` contains the logic for handling web requests.
- **URLs:** `attendance_system_facial_recognition/urls.py` and `recognition/urls.py` define the URL routing.
- **Celery Tasks:** `recognition/tasks.py` contains asynchronous tasks for face recognition and attendance processing.

## Frontend Architecture

The frontend is built with Django templates and Bootstrap 5. Key components include:

- **Templates:** HTML files in `recognition/templates/` and `users/templates/`.
- **Static Files:** CSS, JavaScript, and images in `recognition/static/`.

## Face Recognition Pipeline

The face recognition pipeline is implemented in the `recognition` app. The core logic is in `recognition/pipeline.py`.

## Asynchronous Tasks with Celery

Heavy operations like face recognition are offloaded to Celery tasks to avoid blocking web requests.

- **`recognize_face`:** Takes an image and performs face recognition.
- **`process_attendance_batch`:** Processes a batch of attendance records.

To run the Celery worker:
```bash
celery -A attendance_system_facial_recognition worker -l info
```

## Logging and Error Handling

The application uses Python's `logging` module for structured logging. Logs are written to the console in JSON format in production.

Errors are tracked with Sentry. The Sentry integration is configured in `attendance_system_facial_recognition/settings/sentry.py`.

## Testing

To run the test suite:
```bash
pytest
```

## Deployment

The application is deployed with Docker. See `docker-compose.yml` for the service configuration.
