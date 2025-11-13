# UI Tests

This directory contains UI tests for the Smart Attendance System using Playwright.

## Overview

These tests verify the user interface functionality, including:
- Theme toggle and dark mode persistence
- Mobile navigation behavior
- Table enhancements (search, sort, export)
- Form interactions
- Accessibility features
- Keyboard navigation

## Prerequisites

Before running the tests, you need to:

1. **Install Playwright:**
   ```bash
   pip install pytest-playwright
   playwright install chromium  # Or firefox, webkit
   ```

2. **Start the Django development server:**
   ```bash
   python manage.py runserver
   ```
   
   The tests expect the application to be running at `http://localhost:8000/`

## Running Tests

### Run All UI Tests

```bash
pytest tests/ui/
```

### Run Specific Test Files

```bash
# Theme toggle tests
pytest tests/ui/test_theme_toggle.py

# Mobile navigation tests
pytest tests/ui/test_mobile_navigation.py
```

### Run Specific Tests

```bash
# Run a single test
pytest tests/ui/test_theme_toggle.py::test_theme_toggle_switches_to_dark_mode

# Run tests matching a pattern
pytest tests/ui/ -k "mobile"
```

### Run with Browser UI (Headed Mode)

```bash
pytest tests/ui/ --headed
```

This will show the browser window during test execution, useful for debugging.

### Run with Specific Browser

```bash
# Chromium (default)
pytest tests/ui/ --browser chromium

# Firefox
pytest tests/ui/ --browser firefox

# WebKit (Safari)
pytest tests/ui/ --browser webkit
```

### Run with Slow Motion

```bash
pytest tests/ui/ --headed --slowmo=1000
```

This slows down test execution by 1000ms between actions, helpful for debugging.

### Run Tests by Marker

```bash
# Accessibility tests only
pytest tests/ui/ -m accessibility

# Mobile tests only
pytest tests/ui/ -m mobile

# Theme tests only
pytest tests/ui/ -m theme
```

### Run in Parallel

```bash
pip install pytest-xdist
pytest tests/ui/ -n auto
```

## Test Structure

### Test Files

- `test_theme_toggle.py` - Theme switching and dark mode tests
- `test_mobile_navigation.py` - Mobile menu and responsive design tests
- `test_admin_navigation.py` - End-to-end admin flows for dashboard, employee list, and attendance views
- `conftest.py` - Shared fixtures and configuration

### Fixtures

- `page` - Standard browser page
- `mobile_page` - Page with mobile viewport settings
- `browser` - Browser instance
- `browser_context_args` - Browser context configuration
- `admin_account` - Creates a temporary superuser for admin navigation flows

### Markers

Tests are marked with categories for easy filtering:

- `@pytest.mark.accessibility` - Accessibility feature tests
- `@pytest.mark.mobile` - Mobile-specific tests
- `@pytest.mark.theme` - Theme-related tests
- `@pytest.mark.table` - Table functionality tests

## Writing New Tests

### Basic Test Template

```python
import pytest
from playwright.sync_api import Page, expect

def test_something(page: Page):
    """Test description."""
    page.goto('http://localhost:8000/')
    
    # Your test code here
    element = page.locator('#element-id')
    expect(element).to_be_visible()
```

### Mobile Test Template

```python
@pytest.mark.mobile
def test_mobile_feature(mobile_page: Page):
    """Test mobile-specific feature."""
    mobile_page.goto('http://localhost:8000/')
    
    # Your mobile test code here
```

### Best Practices

1. **Use Descriptive Test Names**: Make test names clearly describe what they test
2. **Use Locators Wisely**: Prefer ID and data-testid over CSS classes
3. **Wait for Elements**: Use `expect()` which includes auto-waiting
4. **Clean Up**: Fixtures handle cleanup automatically
5. **Mark Tests**: Use markers to categorize tests
6. **Independent Tests**: Each test should be independent
7. **Test One Thing**: Each test should verify one specific behavior

## Debugging Tests

### Common Issues

**Issue: "Connection refused" error**
- Make sure Django server is running at `http://localhost:8000/`

**Issue: Tests fail but work in browser**
- Try running with `--headed` to see what's happening
- Add `page.pause()` in your test to stop and inspect

**Issue: Element not found**
- Check that the element exists on the page
- Ensure you're on the correct URL
- Wait for dynamic content to load

### Debug Mode

```python
def test_debug_example(page: Page):
    page.goto('http://localhost:8000/')
    
    # Pause execution for manual inspection
    page.pause()
    
    # Take a screenshot
    page.screenshot(path="debug.png")
    
    # Print page content
    print(page.content())
```

### VS Code Integration

Add to `.vscode/settings.json`:

```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ]
}
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Install Playwright
  run: |
    pip install pytest-playwright
    playwright install chromium

- name: Run UI Tests
  run: |
    python manage.py runserver &
    sleep 5  # Wait for server to start
    pytest tests/ui/
```

## Coverage

To generate coverage reports:

```bash
pip install pytest-cov
pytest tests/ui/ --cov=recognition --cov-report=html
```

View the report by opening `htmlcov/index.html`.

## Resources

- [Playwright Documentation](https://playwright.dev/python/)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-playwright Plugin](https://github.com/microsoft/playwright-pytest)

## Adding More Tests

When adding new features to the UI, add corresponding tests:

1. Create a new test file: `test_feature_name.py`
2. Add appropriate markers
3. Write tests for the new functionality
4. Update this README if needed

## Questions?

See the main project [Developer Guide](../../docs/developer-guide.md) for more information about the UI architecture and testing strategies.
