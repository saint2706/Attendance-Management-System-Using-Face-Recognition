# Developer Guide

This guide provides a comprehensive overview of the project's structure, development setup, coding conventions, and testing procedures. It is intended for developers who want to contribute to the project or understand its inner workings.

## 1. Project Structure

The project is organized into the following directories:

-   `attendance_system_facial_recognition`: The main Django project directory.
    -   `settings.py`: The project's settings.
    -   `urls.py`: The project's URL patterns.
-   `recognition`: The Django app that handles face recognition and attendance tracking.
    -   `views.py`: The views for the recognition app.
    -   `models.py`: The models for the recognition app.
    -   `forms.py`: The forms for the recognition app.
    -   `tests.py`: The tests for the recognition app.
    -   `static/`: The static files for the recognition app.
    -   `templates/`: The templates for the recognition app.
-   `users`: The Django app that handles user management.
    -   `views.py`: The views for the users app.
    -   `models.py`: The models for the users app.
    -   `tests.py`: The tests for the users app.
    -   `templates/`: The templates for the users app.
-   `face_recognition_data`: The directory where the face recognition data is stored.
    -   `training_dataset`: The directory where the training dataset is stored. Each subdirectory is named after a user and contains their face images.

## 2. Development Setup

To set up the project for development, follow these steps:

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

4.  **Run database migrations:**
    ```bash
    python manage.py migrate
    ```

5.  **Create a superuser (admin account):**
    ```bash
    python manage.py createsuperuser
    ```
    Follow the prompts to create your admin username, email, and password.

6.  **Run the development server:**
    ```bash
    python manage.py runserver
    ```
    The application will be available at `http://127.0.0.1:8000/`.

## 3. Coding Conventions

The project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. Please ensure that your code adheres to these conventions.

In addition, the project uses the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings. Please ensure that all functions and methods have a comprehensive docstring that follows this style.

## 4. Testing

The project uses Django's built-in test framework for testing. To run the tests, use the following command:

```bash
python manage.py test
```

Please ensure that all new features are accompanied by a comprehensive set of tests.
