# Smart Attendance System - Developer Guide

**Version 2.0** | Last Updated: November 2024

## Overview

This guide is for developers and technical users who want to customize, extend, or understand the UI/UX implementation of the Smart Attendance System.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design System](#design-system)
3. [Customizing the UI](#customizing-the-ui)
4. [Adding New Pages](#adding-new-pages)
5. [JavaScript Features](#javascript-features)
6. [Accessibility Guidelines](#accessibility-guidelines)
7. [Testing](#testing)
8. [Best Practices](#best-practices)

---

## Architecture Overview

### Technology Stack

- **Backend**: Django 5+
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Styling**: Custom CSS with CSS Variables + Bootstrap 5 (compatibility layer)
- **Icons**: Font Awesome 6
- **Face Recognition**: DeepFace (Facenet model)

### File Structure

```
attendance_system/
├── recognition/
│   ├── static/
│   │   ├── css/
│   │   │   ├── app.css       # Main design system
│   │   │   └── styles.css    # Legacy styles (compatibility)
│   │   └── js/
│   │       └── ui.js         # UI enhancements
│   ├── templates/
│   │   └── recognition/
│   │       ├── base.html     # Base template
│   │       └── *.html        # Page templates
│   └── ...
├── users/
│   └── templates/
│       └── users/
│           ├── login.html
│           └── register.html
├── docs/
│   ├── user-guide.md
│   ├── developer-guide.md
│   ├── theme-customization.md
│   └── change-log.md
└── ...
```

---

## Design System

### CSS Variables

The design system is built using CSS Custom Properties (variables) defined in `recognition/static/css/app.css`.

#### Color Tokens

```css
/* Primary colors */
--color-primary: #0d6efd;
--color-primary-dark: #0b5ed7;
--color-primary-light: #6ea8fe;

/* Semantic colors */
--color-success: #198754;
--color-danger: #dc3545;
--color-warning: #ffc107;
--color-info: #0dcaf0;

/* Neutral colors */
--color-gray-50: #f8f9fa;
--color-gray-100: #e9ecef;
/* ... up to gray-900 */

/* Contextual colors */
--color-background: var(--color-white);
--color-surface: var(--color-white);
--color-text: var(--color-gray-900);
--color-text-muted: var(--color-gray-600);
```

#### Spacing Scale

```css
--space-1: 0.25rem;  /* 4px */
--space-2: 0.5rem;   /* 8px */
--space-3: 0.75rem;  /* 12px */
--space-4: 1rem;     /* 16px */
--space-5: 1.25rem;  /* 20px */
--space-6: 1.5rem;   /* 24px */
--space-8: 2rem;     /* 32px */
--space-10: 2.5rem;  /* 40px */
--space-12: 3rem;    /* 48px */
--space-16: 4rem;    /* 64px */
```

#### Typography Scale

```css
--font-size-xs: 0.75rem;    /* 12px */
--font-size-sm: 0.875rem;   /* 14px */
--font-size-base: 1rem;     /* 16px */
--font-size-lg: 1.125rem;   /* 18px */
--font-size-xl: 1.25rem;    /* 20px */
--font-size-2xl: 1.5rem;    /* 24px */
--font-size-3xl: 1.875rem;  /* 30px */
--font-size-4xl: 2.25rem;   /* 36px */
```

### Dark Theme

Dark theme is implemented using a `.theme-dark` class on the root `<html>` element.

```css
.theme-dark {
  --color-primary: #6ea8fe;
  --color-background: #0d1117;
  --color-surface: #161b22;
  --color-text: #e6edf3;
  --color-text-muted: #8b949e;
  --color-border: #30363d;
  /* ... other overrides */
}
```

---

## Customizing the UI

### Changing Colors

#### Method 1: Update CSS Variables (Recommended)

Edit `recognition/static/css/app.css`:

```css
:root {
  --color-primary: #your-color;
  --color-primary-dark: #your-darker-color;
}
```

See [Theme Customization Guide](theme-customization.md) for detailed instructions.

#### Method 2: Override in Custom CSS

Create a custom CSS file and load it after `app.css`:

```html
<!-- In base.html -->
<link rel="stylesheet" href="{% static 'css/app.css' %}">
<link rel="stylesheet" href="{% static 'css/custom.css' %}">
```

### Changing Fonts

Update the font family variable:

```css
:root {
  --font-family-base: 'Your Font', -apple-system, BlinkMacSystemFont, sans-serif;
}
```

Don't forget to load the font:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Your+Font&display=swap" rel="stylesheet">
```

### Changing Spacing

Adjust the spacing scale:

```css
:root {
  --space-base: 1rem;     /* Change base unit */
  --space-4: calc(var(--space-base) * 1);
  --space-8: calc(var(--space-base) * 2);
  /* ... */
}
```

### Adding Custom Styles

Add custom utility classes in `app.css` or a custom CSS file:

```css
.custom-card {
  background: linear-gradient(135deg, var(--color-primary), var(--color-primary-dark));
  padding: var(--space-8);
  border-radius: var(--radius-xl);
}
```

---

## Adding New Pages

### Step 1: Create Template

Create a new template in `recognition/templates/recognition/`:

```html
{% extends "recognition/base.html" %}

{% block title %}Your Page Title - Smart Attendance System{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-12 col-md-10 col-lg-8">
        <article class="card" style="box-shadow: var(--shadow-lg);">
            <div class="card-body" style="padding: var(--space-8);">
                <h1 style="font-size: var(--font-size-3xl); font-weight: var(--font-weight-bold); color: var(--color-text);">
                    Your Page Title
                </h1>
                <p style="color: var(--color-text-muted);">
                    Your content here...
                </p>
            </div>
        </article>
    </div>
</div>
{% endblock %}
```

### Step 2: Add View

In `recognition/views.py`:

```python
def your_view(request):
    context = {
        # Your context data
    }
    return render(request, 'recognition/your_template.html', context)
```

### Step 3: Add URL

In `recognition/urls.py` or main `urls.py`:

```python
path('your-url/', views.your_view, name='your-view-name'),
```

### Step 4: Add Navigation Link

Update `base.html` or relevant template:

```html
<li class="nav-item" role="none">
    <a class="nav-link" href="{% url 'your-view-name' %}" role="menuitem">
        <i class="fas fa-your-icon" aria-hidden="true"></i>
        <span>Your Link Text</span>
    </a>
</li>
```

---

## JavaScript Features

### Available Modules

The `ui.js` file provides several modules:

#### 1. Theme Manager

Handles dark mode toggle and persistence:

```javascript
ThemeManager.enableDarkMode();
ThemeManager.enableLightMode();
ThemeManager.toggle();
ThemeManager.isDarkMode(); // Returns boolean
```

#### 2. Table Enhancer

Automatically enhances tables with search, sort, and CSV export:

```html
<table class="table" data-enhance="true">
  <!-- Table content -->
</table>
```

Features added:
- Search/filter input
- Sortable columns (click headers)
- CSV export button
- Keyboard navigation

To disable sorting on specific columns:

```html
<th data-sortable="false">Column Name</th>
```

#### 3. Mobile Navigation

Handles collapsible mobile menu:

```html
<button id="mobile-menu-toggle">
  <i class="fas fa-bars"></i>
</button>
<ul class="navbar-nav" id="navbar-nav">
  <!-- Nav items -->
</ul>
```

#### 4. Form Enhancer

Adds validation and floating label effects:

```html
<form data-validate="true">
  <!-- Form fields -->
</form>
```

#### 5. Alert Manager

Auto-dismisses alerts after 5 seconds:

```html
<div class="alert alert-success alert-dismissible" data-auto-dismiss="false">
  <!-- Alert content -->
</div>
```

Set `data-auto-dismiss="false"` to prevent auto-dismiss.

### Adding Custom JavaScript

Create a custom JavaScript file:

```javascript
// recognition/static/js/custom.js

(function() {
  'use strict';
  
  function init() {
    // Your custom initialization
    console.log('Custom JS loaded');
  }
  
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
```

Load it in your template:

```html
{% block extra_scripts %}
<script src="{% static 'js/custom.js' %}"></script>
{% endblock %}
```

---

## Accessibility Guidelines

### Semantic HTML

Always use semantic HTML elements:

```html
<!-- Good -->
<nav role="navigation" aria-label="Main navigation">
<main role="main">
<article>
<section aria-label="Section description">
<header>
<footer>

<!-- Avoid -->
<div class="nav">
<div class="main">
```

### ARIA Attributes

Add ARIA attributes for screen readers:

```html
<!-- Buttons -->
<button aria-label="Toggle dark mode" title="Toggle dark mode">
  <i class="fas fa-moon" aria-hidden="true"></i>
</button>

<!-- Icons (decorative) -->
<i class="fas fa-check" aria-hidden="true"></i>

<!-- Menus -->
<ul role="menubar">
  <li role="none">
    <a role="menuitem" href="#">Menu Item</a>
  </li>
</ul>

<!-- Tables -->
<table aria-label="Attendance records table">
  <thead>
    <tr>
      <th scope="col">Date</th>
      <th scope="col">Status</th>
    </tr>
  </thead>
</table>

<!-- Form labels -->
<label for="username">Username</label>
<input type="text" id="username" name="username" aria-required="true">
```

### Keyboard Navigation

Ensure all interactive elements are keyboard accessible:

```html
<!-- Add tabindex for custom interactive elements -->
<div role="button" tabindex="0" @keydown.enter="handleClick" @keydown.space="handleClick">
  Click me
</div>
```

### Focus Styles

The design system includes focus-visible styles:

```css
:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}
```

### Color Contrast

Ensure sufficient color contrast (WCAG AA minimum):
- Normal text: 4.5:1
- Large text: 3:1
- UI components: 3:1

Test with browser DevTools or online tools.

### Skip Links

Always include skip-to-content link (already in base.html):

```html
<a href="#main-content" class="skip-to-main">Skip to main content</a>
```

---

## Testing

### Manual Testing Checklist

- [ ] Test on multiple browsers (Chrome, Firefox, Safari, Edge)
- [ ] Test responsive design (mobile, tablet, desktop)
- [ ] Test dark mode toggle and persistence
- [ ] Test keyboard navigation (Tab, Enter, Escape, Arrow keys)
- [ ] Test screen reader compatibility
- [ ] Test table filters and CSV export
- [ ] Test form validation
- [ ] Test mobile menu behavior

### Automated Testing

#### Playwright Tests

Create tests in `tests/ui/`:

```python
# tests/ui/test_theme_toggle.py
import pytest
from playwright.sync_api import Page, expect

def test_theme_toggle_persistence(page: Page):
    """Test that dark mode persists across page loads."""
    page.goto('http://localhost:8000/')
    
    # Click theme toggle
    page.click('#theme-toggle')
    
    # Check dark class is applied
    html = page.locator('html')
    expect(html).to_have_class('theme-dark')
    
    # Reload page
    page.reload()
    
    # Dark mode should persist
    expect(html).to_have_class('theme-dark')
```

Run tests:

```bash
pytest tests/ui/ --headed  # With browser UI
pytest tests/ui/           # Headless
```

#### Lighthouse Audits

**Automated CI Audits**

Lighthouse CI runs automatically on every push and pull request via GitHub Actions. The audit results are uploaded as workflow artifacts and can be viewed in the Actions tab.

**Running Locally**

To run Lighthouse audits locally:

```bash
# Install Lighthouse CLI
npm install -g @lhci/cli

# Run audit with configuration
lhci autorun

# Or manually specify URLs
lhci collect --url=http://localhost:8000
```

The repository includes a `.lighthouserc.js` configuration file that:
- Automatically starts the Django server
- Tests multiple pages (home, login)
- Runs 3 audits and takes the median score
- Enforces target score thresholds

Target scores (enforced in CI):
- Accessibility: ≥ 95
- Best Practices: ≥ 95
- Performance: ≥ 80
- SEO: ≥ 90

### Browser DevTools

Use browser DevTools for:
- **Accessibility**: Check ARIA, contrast, tab order
- **Responsive**: Test different screen sizes
- **Console**: Check for JavaScript errors
- **Network**: Monitor load times
- **Lighthouse**: Built-in auditing tool

---

## Best Practices

### CSS Best Practices

1. **Use CSS Variables**: Always use design tokens instead of hard-coded values

```css
/* Good */
color: var(--color-primary);
padding: var(--space-4);

/* Avoid */
color: #0d6efd;
padding: 16px;
```

2. **Mobile-First**: Write mobile styles first, then add desktop overrides

```css
/* Mobile first */
.card {
  padding: var(--space-4);
}

/* Desktop override */
@media (min-width: 768px) {
  .card {
    padding: var(--space-8);
  }
}
```

3. **Avoid !important**: Use specific selectors instead

4. **Use Logical Properties**: Use `inline` and `block` instead of left/right/top/bottom when possible

```css
/* Good (supports RTL languages) */
margin-inline-start: var(--space-4);
padding-block: var(--space-6);

/* Less flexible */
margin-left: var(--space-4);
padding-top: var(--space-6);
padding-bottom: var(--space-6);
```

### HTML Best Practices

1. **Semantic Elements**: Use appropriate HTML5 elements
2. **Accessibility First**: Add ARIA attributes from the start
3. **Valid HTML**: Use W3C validator
4. **Progressive Enhancement**: Ensure functionality without JavaScript

### JavaScript Best Practices

1. **Vanilla JS**: Keep dependencies minimal
2. **Module Pattern**: Use IIFE to avoid global pollution
3. **Event Delegation**: For dynamic elements
4. **Error Handling**: Always handle errors gracefully

```javascript
try {
  // Your code
} catch (error) {
  console.error('Error:', error);
  // Show user-friendly message
}
```

5. **Performance**: Debounce frequent events (scroll, resize, input)

```javascript
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
```

### Django Template Best Practices

1. **Extend Base**: Always extend base.html
2. **Block Organization**: Use blocks appropriately
3. **Static Files**: Use `{% static %}` tag
4. **URL Reversal**: Use `{% url %}` tag instead of hard-coded URLs
5. **Escaping**: Django auto-escapes, use `|safe` sparingly

```html
<!-- Good -->
<a href="{% url 'dashboard' %}">Dashboard</a>
<img src="{% static 'images/logo.png' %}" alt="Logo">

<!-- Avoid -->
<a href="/dashboard/">Dashboard</a>
<img src="/static/images/logo.png" alt="Logo">
```

---

## Common Customization Scenarios

### Adding a New Color Theme

1. Define new color variables in `:root`
2. Create theme class (e.g., `.theme-blue`)
3. Add theme toggle option
4. Update ThemeManager in ui.js

### Creating Custom Components

```html
<!-- Custom info card component -->
<div class="custom-info-card">
  <div class="custom-info-card__icon">
    <i class="fas fa-info-circle"></i>
  </div>
  <div class="custom-info-card__content">
    <h3 class="custom-info-card__title">Title</h3>
    <p class="custom-info-card__text">Content</p>
  </div>
</div>
```

```css
/* Component styles */
.custom-info-card {
  display: flex;
  gap: var(--space-4);
  padding: var(--space-6);
  background-color: var(--color-surface);
  border-radius: var(--radius-lg);
  border-left: 4px solid var(--color-info);
}

.custom-info-card__icon {
  font-size: 2rem;
  color: var(--color-info);
}

.custom-info-card__title {
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  margin-bottom: var(--space-2);
}
```

### Adding Animation

```css
@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(-20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.animated-element {
  animation: slideIn var(--transition-base) ease;
}
```

---

## Resources

### Documentation Links

- [Django Documentation](https://docs.djangoproject.com/)
- [MDN Web Docs](https://developer.mozilla.org/)
- [W3C Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Font Awesome Icons](https://fontawesome.com/icons)

### Tools

- [CSS Variables Reference](https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_custom_properties)
- [Playwright Documentation](https://playwright.dev/)
- [Lighthouse CI](https://github.com/GoogleChrome/lighthouse-ci)
- [Color Contrast Checker](https://webaim.org/resources/contrastchecker/)

---

## Getting Help

For more information:

- **User Guide**: See [user-guide.md](user-guide.md) for end-user documentation
- **Theme Customization**: See [theme-customization.md](theme-customization.md)
- **Change Log**: See [change-log.md](change-log.md)
- **Issues**: Open an issue on GitHub

---

## Contributing

When contributing UI/UX changes:

1. Follow the design system
2. Maintain accessibility standards
3. Test on multiple browsers and devices
4. Update documentation
5. Add tests for new features
6. Run linters and formatters

---

*Last updated: November 2024 | Version 2.0*
