# Good First Issues

This document contains a curated list of beginner-friendly tasks for new contributors. Each issue is scoped, has clear acceptance criteria, and points to the relevant files.

> **Maintainers:** Convert these into real GitHub issues and add the `good first issue` and appropriate component labels.

---

## Documentation

### 1. Add Missing Screenshots to User Guide

**Description:** Some sections of the User Guide reference screenshots that may be outdated or missing. Update or capture new screenshots using `make docs-screenshots`.

**Labels:** `good first issue`, `docs`, `documentation`

**Files involved:**

- `docs/USER_GUIDE.md`
- `docs/screenshots/`
- `scripts/capture_screenshots.py`

**Skill level:** Beginner  
**Estimated effort:** 1-2 hours

---

### 2. Improve Troubleshooting Guide

**Description:** Add more common issues and solutions to the troubleshooting guide based on existing GitHub issues.

**Labels:** `good first issue`, `docs`

**Files involved:**

- `docs/troubleshooting.md`

**Skill level:** Beginner  
**Estimated effort:** 1-2 hours

---

## Backend

### 3. Add Type Hints to Core Models

**Description:** Add type hints to model fields and methods in `recognition/models.py` to improve IDE support and documentation.

**Labels:** `good first issue`, `backend`

**Files involved:**

- `recognition/models.py`
- `users/models.py`

**Skill level:** Beginner (Python)  
**Estimated effort:** 2-3 hours

---

### 4. Add Docstrings to Management Commands

**Description:** Ensure all management commands in `recognition/management/commands/` have comprehensive docstrings explaining usage and options.

**Labels:** `good first issue`, `backend`, `docs`

**Files involved:**

- `recognition/management/commands/*.py`

**Skill level:** Beginner  
**Estimated effort:** 2-3 hours

---

## Frontend

### 5. Improve Loading States in React Dashboard

**Description:** Add skeleton loaders or spinner states to dashboard components while data is loading.

**Labels:** `good first issue`, `frontend`

**Files involved:**

- `frontend/src/components/`
- `frontend/src/pages/`

**Skill level:** Beginner (React/TypeScript)  
**Estimated effort:** 2-4 hours

---

### 6. Add Dark Mode Toggle Persistence

**Description:** Ensure the dark mode preference is saved to localStorage and restored on page load.

**Labels:** `good first issue`, `frontend`

**Files involved:**

- `frontend/src/contexts/` or theme-related files
- `frontend/src/App.tsx`

**Skill level:** Beginner  
**Estimated effort:** 1-2 hours

---

## Testing

### 7. Add Edge Case Tests for Liveness Detection

**Description:** Extend test coverage for edge cases in liveness detection (low light, multiple faces, rapid movement).

**Labels:** `good first issue`, `backend`, `recognition`

**Files involved:**

- `tests/recognition/test_liveness.py`
- `tests/recognition/test_enhanced_liveness.py`

**Skill level:** Intermediate  
**Estimated effort:** 3-4 hours

---

## Infrastructure

### 8. Add Health Check Endpoint Documentation

**Description:** Document the `/admin/health/` endpoint and its response format in the API Reference.

**Labels:** `good first issue`, `docs`, `infra`

**Files involved:**

- `docs/API_REFERENCE.md`
- `recognition/health.py`

**Skill level:** Beginner  
**Estimated effort:** 1-2 hours

---

## How to Get Started

1. Comment on an issue to claim it
2. Fork the repository and create a feature branch
3. Follow the [Developer Guide](DEVELOPER_GUIDE.md) for setup
4. Submit a PR and reference the issue number
5. Respond to review feedback

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.
