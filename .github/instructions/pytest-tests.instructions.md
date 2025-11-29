---
applyTo: "**/tests/**/*.py"
---

# Pytest Test Requirements

When writing or modifying tests in this project, follow these guidelines to ensure consistency and maintainability:

## Test Structure

1. **Use pytest-style tests** - Prefer pytest fixtures and assertions over Django's TestCase
2. **Follow naming conventions** - Use descriptive test names with `test_` prefix and `_test.py` or `test_*.py` file naming
3. **Write isolated tests** - Each test should be independent and not rely on other tests' state
4. **Use fixtures** - Leverage pytest fixtures for setup and teardown, defined in `conftest.py` files

## Test Organization

- Place unit tests in `tests/` directory organized by app/module
- Place UI tests in `tests/ui/` directory
- Mark UI tests with `@pytest.mark.ui` decorator
- Mark slow tests with `@pytest.mark.slow` decorator

## Django Testing

1. **Use Django test client** - Use `client` fixture for view tests
2. **Use `db` marker** - Mark tests that need database access with `@pytest.mark.django_db`
3. **Mock external services** - Use `unittest.mock` to mock external API calls
4. **Test both authenticated and unauthenticated scenarios**

## Face Recognition Tests

- Use synthetic test data from `sample_data/` directory
- Mock DeepFace calls for unit tests to avoid long processing times
- Test encryption/decryption of face data separately
- Validate embedding generation determinism

## Assertions

```python
# Preferred - explicit assertions
assert response.status_code == 200
assert "expected_key" in response.json()

# With helpful error messages
assert user.is_active, f"User {user.username} should be active"
```

## Fixtures Example

```python
import pytest
from django.contrib.auth import get_user_model

User = get_user_model()

@pytest.fixture
def admin_user(db):
    return User.objects.create_superuser(
        username="admin",
        email="admin@example.com",
        password="adminpass"
    )

@pytest.fixture
def authenticated_client(client, admin_user):
    client.login(username="admin", password="adminpass")
    return client
```

## Running Tests

```bash
# Run all tests except UI
pytest -m "not ui"

# Run only UI tests
pytest -m "ui" tests/ui

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_recognition.py -v
```
