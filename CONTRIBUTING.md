# Contributing to the Attendance Management System

Thank you for your interest in contributing to the Attendance Management System Using Face Recognition! This document provides a comprehensive guide for developers who want to contribute to the project.

## Code of Conduct

This project has a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- A webcam for face recognition (optional for most development tasks)
- Docker and Docker Compose (optional, for containerized development)

For comprehensive setup instructions, see the [Developer Guide](docs/DEVELOPER_GUIDE.md).

### Development Setup

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

4. **Run database migrations:**

    ```bash
    python manage.py migrate
    ```

5. **Create a superuser (admin account):**

    ```bash
    python manage.py createsuperuser
    ```

    Follow the prompts to create your admin username, email, and password.

6. **Run the development server:**

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

The project uses pytest for testing. To run the full test suite:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=recognition --cov=users --cov-report=term-missing
```

You can also use Django's test runner:

```bash
python manage.py test
```

### Running Linters

Before submitting your changes, ensure they pass the pre-commit hooks:

```bash
# Install pre-commit hooks (first time only)
make install-hooks

# Run all linters manually
pre-commit run --all-files
```

Please ensure that all new features are accompanied by a comprehensive set of tests.

### Test Coverage

We aim for a minimum of 60% test coverage. Please ensure that your contributions are well-tested and that the test coverage does not decrease. The CI pipeline will fail if coverage drops below this threshold.

## Filing Issues

### Before Opening an Issue

1. **Search existing issues** to ensure your issue hasn't already been reported.
2. **Check the documentation** ([README](README.md), [User Guide](docs/USER_GUIDE.md), [Developer Guide](docs/DEVELOPER_GUIDE.md)) to see if your question is already answered.

### Bug Reports

When reporting a bug, use the [Bug Report template](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/issues/new?template=bug_report.yml) and include:

- A clear description of the bug
- Steps to reproduce the behavior
- Expected vs. actual behavior
- Your environment details (OS, Python version, browser, etc.)
- Relevant logs or error messages

### Feature Requests

For new feature ideas, use the [Feature Request template](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/issues/new?template=feature_request.yml) and include:

- A clear problem statement
- Your proposed solution
- Any alternatives you've considered

## Contributing Guidelines

### Submitting a Pull Request

1. **Fork the repository:**
    Create a fork of the repository on GitHub.

2. **Create a new branch:**
    Create a new branch for your changes.

3. **Make your changes:**
    Make your changes to the code.

4. **Run the tests:**
    Run the tests to ensure that your changes do not break anything.

5. **Commit your changes:**
    Commit your changes with a clear and descriptive commit message.

6. **Push your changes:**
    Push your changes to your fork.

7. **Create a pull request:**
    Create a pull request from your fork to the main repository.

### Code Review

All pull requests will be reviewed by at least one other developer. Please ensure that you address all comments and suggestions from the reviewer.

## Labels

We use labels to categorize issues and pull requests. Here's what each label means:

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working as expected |
| `enhancement` | New feature or improvement request |
| `good first issue` | Good for newcomers to the project |
| `help wanted` | Extra attention is needed |
| `documentation` | Improvements or additions to documentation |
| `security` | Security-related issues |
| `question` | Further information is requested |
| `wontfix` | This will not be worked on |

### For Maintainers

When triaging issues:

- Add `good first issue` to issues that are well-scoped and beginner-friendly
- Add `help wanted` to issues where community contributions are welcome
- Add appropriate component labels (e.g., `recognition`, `ui`, `api`) when available

## Questions and Support

For questions about using the project, see [SUPPORT.md](docs/SUPPORT.md).

## Contributor Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
