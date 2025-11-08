# Contributing to the Smart Attendance System

Thank you for your interest in contributing to the Modern Smart Attendance System! This document provides a comprehensive guide for developers who want to contribute to the project.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- A webcam for face recognition

### Development Setup

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

## Coding Conventions

### Python Style Guide

The project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. Please ensure that your code adheres to these conventions.

### Docstrings

The project uses the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings. Please ensure that all functions and methods have a comprehensive docstring that follows this style.

### Pre-commit Hooks

The project uses pre-commit hooks to maintain code quality. To install the hooks, run the following command:

```bash
make install-hooks
```

The hooks will automatically check your code for style violations before you commit your changes.

## Testing

### Running the Tests

The project uses Django's built-in test framework for testing. To run the tests, use the following command:

```bash
python manage.py test
```

Please ensure that all new features are accompanied by a comprehensive set of tests.

### Test Coverage

We aim for a high test coverage. Please ensure that your contributions are well-tested and that the test coverage does not decrease.

## Contributing Guidelines

### Submitting a Pull Request

1.  **Fork the repository:**
    Create a fork of the repository on GitHub.

2.  **Create a new branch:**
    Create a new branch for your changes.

3.  **Make your changes:**
    Make your changes to the code.

4.  **Run the tests:**
    Run the tests to ensure that your changes do not break anything.

5.  **Commit your changes:**
    Commit your changes with a clear and descriptive commit message.

6.  **Push your changes:**
    Push your changes to your fork.

7.  **Create a pull request:**
    Create a pull request from your fork to the main repository.

### Code Review

All pull requests will be reviewed by at least one other developer. Please ensure that you address all comments and suggestions from the reviewer.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.
