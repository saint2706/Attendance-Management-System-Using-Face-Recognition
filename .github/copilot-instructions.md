# GitHub Copilot Instructions

This file provides custom instructions for GitHub Copilot when working in this repository.

## About This Project

This is a Django 5 + Celery + Docker + PWA application implementing a smart attendance management system using face recognition. The system includes:

- **Backend**: Django 5 with face recognition pipeline using DeepFace (Facenet model)
- **Frontend**: Progressive Web App with Bootstrap 5, offline-ready with service workers
- **Infrastructure**: Docker + docker-compose, Celery workers, Redis, PostgreSQL
- **Security**: Encrypted data fields, Sentry integration for error tracking
- **Testing**: pytest, Django test framework, Playwright (planned)

## Core Principles

When contributing to this codebase, always follow these principles:

1. **Think step-by-step**: Plan before executing changes
2. **Production-quality code**: Complete, idiomatic, and maintainable
3. **Minimal changes**: Make surgical, precise modifications
4. **Security first**: No secrets in code, validate inputs, use encryption helpers
5. **Architecture awareness**: Respect existing patterns and structure
6. **Documentation**: Update docs when behavior changes
7. **Testing**: Add tests for new functionality

## Project Structure

```
├── attendance_system_facial_recognition/  # Core Django project
├── recognition/                           # Face recognition pipeline
├── users/                                 # User management app
├── tests/                                 # Test suite
├── docs/                                  # Documentation
├── .github/                               # CI/CD workflows
├── Dockerfile                             # Container image
└── docker-compose.yml                     # Service orchestration
```

## Code Style & Quality

### Python

- Follow PEP8 standards
- Use type hints everywhere
- Add docstrings for all public functions/classes
- Prefer pure functions where possible
- Use logging instead of print statements
- Follow Django best practices (CBVs, proper serializers)

### HTML/CSS/JS

- Use semantic HTML
- Maintain Bootstrap 5 UI consistency
- Minimal JavaScript; avoid inline scripts
- Group reusable UI patterns into partials

### Docker

- Multi-stage builds
- Small runtime images
- Proper build caching

## Security Requirements

All code changes must:

- Use environment variables for configuration (never hardcode secrets)
- Use the project's Fernet encryption helpers for sensitive data
- Validate all user inputs
- Maintain secure production settings (CSRF_COOKIE_SECURE, SESSION_COOKIE_SECURE, etc.)
- Not weaken existing security measures

## Testing Guidelines

- Add tests in `tests/` directory
- Use pytest-style tests
- Cover all new functionality:
  - Face recognition pipeline changes
  - Attendance marking logic
  - Employee & admin flows
  - API endpoints
  - Permissions
- Use Playwright for frontend testing when applicable

## Face Recognition Pipeline

When modifying `recognition/` code:

- Keep embeddings reproducible and deterministic
- Never degrade accuracy without explicit approval
- Validate threshold logic with synthetic tests
- Use Celery for heavy tasks (don't block Django request thread)
- Document changes in matching logic

## Documentation Requirements

When behavior changes, update relevant documentation:

- **USER_GUIDE.md**: End-user instructions
- **DEVELOPER_GUIDE.md**: Technical documentation for contributors
- **API_REFERENCE.md**: API endpoints and CLI tools
- **docs/deployment-guide.md**: Deployment procedures
- **docs/security.md**: Security-related changes
- **DATA_CARD.md**: Data model changes

Ensure instructions are:

- Accurate and up-to-date
- Step-by-step and beginner-friendly
- Aligned with actual UI and behavior

## Deployment Considerations

Changes must ensure:

- Docker builds pass cleanly
- Static files collect without errors (`python manage.py collectstatic`)
- Celery workers boot successfully
- PWA continues to function (manifest.json, service worker)
- Sentry captures errors correctly

## Common Commands

```bash
# Development setup
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver

# Testing
pytest
python manage.py test

# Docker
docker compose up -d postgres
docker compose build web
docker compose up -d web celery

# Collect static files
python manage.py collectstatic --noinput
```

## Environment Variables

Key environment variables used in this project:

- `DJANGO_SECRET_KEY`: Django secret key
- `DATA_ENCRYPTION_KEY`: Fernet key for general data encryption
- `FACE_DATA_ENCRYPTION_KEY`: Fernet key for face data encryption
- `DATABASE_URL`: Database connection string
- `DJANGO_ALLOWED_HOSTS`: Comma-separated list of allowed hosts
- `SENTRY_DSN`: Sentry DSN for error tracking
- `DJANGO_DEBUG`: Enable/disable debug mode

## Additional Resources

For comprehensive agent instructions and detailed guidelines, see:

- **[AGENTS.md](docs\AGENTS.md)**: Detailed agent behavior and responsibilities
- **[ARCHITECTURE.md](docs\ARCHITECTURE.md)**: System architecture overview
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Contribution guidelines
- **[README.md](README.md)**: Project overview and getting started

## Communication Style

When providing assistance:

- Be concise and accurate
- Be direct, no fluff
- No hallucinations or assumptions
- Ask for clarification on risky or ambiguous tasks
- Explain reasoning when making architectural decisions
