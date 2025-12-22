# Contributing to Attendance Management System

Thank you for your interest in contributing! This document provides comprehensive guidelines for contributing code, documentation, and other improvements to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Community](#community)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior by opening a GitHub issue or contacting the maintainers.

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.12+** installed
- **Git** for version control
- **A GitHub account**
- **Basic knowledge** of Django, Python, or React (depending on your contribution area)

### Finding Work

1. **Browse [Good First Issues](docs/GOOD_FIRST_ISSUES.md)** for beginner-friendly tasks
2. **Check [open issues](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/issues)** for bugs and feature requests
3. **Review the [Roadmap](docs/ROADMAP.md)** for planned features
4. **Propose new features** by opening a discussion issue first

> **Tip**: Comment on an issue to indicate you're working on it. This prevents duplicate effort.

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Attendance-Management-System-Using-Face-Recognition.git
cd Attendance-Management-System-Using-Face-Recognition

# Add upstream remote
git remote add upstream https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition.git
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies (includes testing, linting, etc.)
pip install -r requirements-dev.txt
```

### 4. Set Up Pre-Commit Hooks

Pre-commit hooks automatically check code formatting before each commit:

```bash
pre-commit install
```

This installs hooks that run:
- **Black** (code formatting)
- **isort** (import sorting)
- **flake8** (linting)

### 5. Configure Environment

```bash
cp .env.example .env
# Edit .env with your local settings if needed
```

### 6. Initialize Database

```bash
python manage.py migrate
python manage.py createsuperuser
```

### 7. Verify Setup

```bash
# Run fast tests to ensure everything works
make test-fast

# Start the development server
python manage.py runserver
```

Visit `http://localhost:8000` to confirm the setup is working.

---

## Making Changes

### Branching Strategy

We use a **feature branch workflow**:

```bash
# Create a new branch for your changes
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

**Branch Naming Conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests
- `chore/` - Maintenance tasks

### Making Commits

Write clear, descriptive commit messages following [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <subject>

feat(recognition): add multi-face detection support
fix(auth): resolve session timeout issue
docs(api): update endpoint documentation
test(users): add test for password reset flow
refactor(pipeline): optimize embedding generation
chore(deps): update Django to 6.0
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, no code change
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

---

## Code Standards

### Python Code

We follow **PEP 8** with some customizations:

**Style Requirements:**
- **Line length**: 100 characters (not 79)
- **Formatter**: Black (automatically applied by pre-commit)
- **Import sorting**: isort (automatically applied)
- **Type hints**: Required for all function signatures
- **Docstrings**: Required for public functions and classes

**Example:**

```python
from typing import Optional

def calculate_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine distance between two face embeddings.
    
    Args:
        embedding1: First face embedding vector
        embedding2: Second face embedding vector
        
    Returns:
        Cosine distance as a float between 0.0 and 2.0
        
    Raises:
        ValueError: If embeddings have different dimensions
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError("Embeddings must have the same dimension")
    
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    return 1.0 - similarity
```

### TypeScript/React Code

**Style Requirements:**
- **Formatter**: ESLint (configured in `frontend/eslint.config.js`)
- **Type annotations**: Required for function parameters and returns
- **Component style**: Functional components with hooks
- **File naming**: PascalCase for components, camelCase for utilities

### Django Code

**Conventions:**
- Use **class-based views** where appropriate
- Keep views thin, logic in models/services
- Use Django ORM (avoid raw SQL unless necessary)
- Follow Django's security best practices
- Use Django's built-in validators

---

## Testing Requirements

**All code changes must include tests.** We maintain 60%+ test coverage.

### Running Tests

```bash
# Fast tests (unit + integration, ~2 minutes)
make test-fast

# Slow tests (model loading, ~10 minutes)
make test-slow

# UI tests (Playwright, ~5 minutes)
make test-ui

# Full suite with coverage report
make test-all
```

### Writing Tests

**Use pytest** with appropriate markers:

```python
import pytest
from django.contrib.auth import get_user_model

User = get_user_model()

@pytest.mark.django_db
def test_user_creation():
    """Test creating a new user."""
    user = User.objects.create_user(
        username="testuser",
        email="test@example.com",
        password="testpass123"
    )
    assert user.username == "testuser"
    assert user.check_password("testpass123")

@pytest.mark.slow
def test_face_recognition_pipeline():
    """Test the complete face recognition pipeline."""
    # Slow test that loads ML models
    pass

@pytest.mark.ui
def test_login_flow(page):
    """Test the login user interface."""
    # Playwright UI test
    pass
```

### Test Coverage Requirements

- **New features**: Must have 80%+ coverage
- **Bug fixes**: Must include regression test
- **Refactoring**: Existing tests must pass

---

## Submitting Changes

### Before Submitting

**Checklist:**
- [ ] Code follows project style guidelines (Black, isort, flake8)
- [ ] All tests pass locally (`make test-fast` at minimum)
- [ ] New tests added for new functionality
- [ ] Documentation updated (README, docs/, docstrings)
- [ ] Commit messages follow Conventional Commits
- [ ] No merge conflicts with `main` branch
- [ ] `.env` and secrets not committed

### Creating a Pull Request

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub

3. **Fill out the PR template** completely:
   - Description of changes
   - Related issue number (if applicable)
   - Testing performed
   - Screenshots (for UI changes)

4. **Request review** from maintainers

### Pull Request Title Format

Use Conventional Commits format:

```
feat(recognition): Add multi-face detection support
fix(auth): Resolve session timeout on mobile devices
docs(deployment): Update Kubernetes configuration guide
```

---

## Review Process

### What to Expect

1. **Automated Checks** run via GitHub Actions:
   - Code formatting (Black, isort)
   - Linting (flake8)
   - Fast tests
   - Security scanning (CodeQL)

2. **Manual Review** by maintainers:
   - Code quality and style
   - Test coverage
   - Documentation accuracy
   - Security considerations

3. **Feedback Cycle**:
   - Reviewers may request changes
   - Address feedback by pushing new commits
   - Reviewers re-review updated code

4. **Merge**:
   - Once approved, maintainers will merge
   - Your contribution will be included in the next release!

### Review Timeline

- **Simple changes** (docs, small fixes): 1-2 days
- **Medium changes** (new features): 3-7 days
- **Large changes** (architecture): 1-2 weeks

> **Note**: These are estimates. Complex changes may take longer to review thoroughly.

---

## Community

### Getting Help

- **Documentation**: Start with [docs/README.md](docs/README.md)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/discussions)
- **Issues**: Report bugs via [GitHub Issues](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/issues)
- **Chat**: (Coming soon - Discord/Slack link)

### Recognition

Contributors are recognized in:
- **CHANGELOG.md**: Each release credits contributors
- **GitHub Contributors**: Automatically tracked
- **Special thanks**: Major contributors listed in README

---

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

---

## Questions?

If you have questions about contributing, please:
1. Check the [Developer Guide](docs/DEVELOPER_GUIDE.md)
2. Search [existing issues](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/issues)
3. Ask in [GitHub Discussions](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/discussions)
4. Open a new issue with the "question" label

**Thank you for contributing to make this project better!** 🎉

---

*Last Updated: December 2025*
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

We use labels to categorize issues and pull requests.

### Type Labels

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working as expected |
| `enhancement` | New feature or improvement request |
| `documentation` | Improvements or additions to documentation |
| `security` | Security-related issues |
| `question` | Further information is requested |
| `wontfix` | This will not be worked on |

### Component Labels

| Label | Description |
|-------|-------------|
| `backend` | Django backend, API endpoints, Celery tasks |
| `frontend` | React SPA, TypeScript, Vite, UI components |
| `recognition` | Face recognition pipeline, liveness detection |
| `docs` | Documentation improvements |
| `infra` | Docker, Kubernetes, CI/CD, deployment |

### Difficulty Labels

| Label | Description |
|-------|-------------|
| `good first issue` | Good for newcomers to the project |
| `help wanted` | Extra attention is needed |

### For Maintainers

When triaging issues:

- Add `good first issue` to issues that are well-scoped and beginner-friendly
- Add `help wanted` to issues where community contributions are welcome
- Add a component label (`backend`, `frontend`, `recognition`, `docs`, `infra`) to every issue

## Questions and Support

For questions about using the project, see [SUPPORT.md](docs/SUPPORT.md).

## Contributor Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
