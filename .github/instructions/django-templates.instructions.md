---
applyTo: "**/templates/**/*.html"
---

# Django Template Guidelines

When creating or modifying Django templates, follow these guidelines for consistency:

## Structure

1. **Extend base templates** - Always extend from `base.html` or appropriate parent
2. **Use semantic HTML** - Use proper HTML5 semantic elements
3. **Follow Bootstrap 5 conventions** - Use Bootstrap classes consistently
4. **Maintain accessibility** - Include ARIA labels, alt text, and proper heading hierarchy

## Template Inheritance

```html
{% extends "base.html" %}
{% load static %}

{% block title %}Page Title{% endblock %}

{% block content %}
<!-- Page-specific content -->
{% endblock %}
```

## Static Files

```html
{% load static %}

<!-- CSS -->
<link rel="stylesheet" href="{% static 'css/custom.css' %}">

<!-- JavaScript -->
<script src="{% static 'js/main.js' %}"></script>

<!-- Images -->
<img src="{% static 'images/logo.png' %}" alt="Logo description">
```

## Bootstrap 5 Components

```html
<!-- Cards -->
<div class="card shadow-sm">
    <div class="card-header">
        <h5 class="card-title mb-0">Title</h5>
    </div>
    <div class="card-body">
        Content
    </div>
</div>

<!-- Buttons -->
<button type="submit" class="btn btn-primary">Submit</button>
<a href="{% url 'name' %}" class="btn btn-outline-secondary">Cancel</a>

<!-- Forms -->
{% load crispy_forms_tags %}
{{ form|crispy }}
```

## Accessibility Requirements

1. **Proper heading hierarchy** - Use h1-h6 in order
2. **Form labels** - Every input must have an associated label
3. **Alt text** - All images must have descriptive alt attributes
4. **ARIA labels** - Add aria-label for icon buttons and interactive elements
5. **Focus indicators** - Don't remove focus outlines

```html
<!-- Good -->
<button type="button" class="btn btn-icon" aria-label="Close dialog">
    <i class="bi bi-x" aria-hidden="true"></i>
</button>

<!-- Bad -->
<button type="button" class="btn btn-icon">
    <i class="bi bi-x"></i>
</button>
```

## PWA Considerations

- Templates should work offline with service worker caching
- Use skeleton screens for loading states
- Keep JavaScript minimal and progressive enhancement friendly

## Django Template Tags

```html
<!-- URLs -->
<a href="{% url 'app:view_name' pk=object.pk %}">Link</a>

<!-- Conditionals -->
{% if user.is_authenticated %}
    <p>Welcome, {{ user.username }}</p>
{% endif %}

<!-- Loops -->
{% for item in items %}
    <li>{{ item.name }}</li>
{% empty %}
    <li>No items found</li>
{% endfor %}

<!-- CSRF Token (required for forms) -->
<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
</form>
```

## Responsive Design

- Use Bootstrap's responsive classes (col-md-*, d-none d-md-block, etc.)
- Test on mobile, tablet, and desktop viewports
- Maintain consistent spacing and alignment
