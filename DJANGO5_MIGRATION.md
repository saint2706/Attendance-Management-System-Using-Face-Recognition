# Django 5.x Migration Notes

## force_text to force_str Migration

### Background
Django deprecated `force_text` in version 3.0 and completely removed it in version 4.0. The modern equivalent is `force_str` from `django.utils.encoding`.

### Changes Made

#### 1. Updated django-pandas Dependency
- **Previous version**: `django-pandas==0.6.1`
- **New version**: `django-pandas>=0.6.7`
- **Reason**: Version 0.6.1 used the deprecated `force_text` function, which causes ImportError with Django 5.x

The older version (0.6.1) contained the following incompatible code in `django_pandas/utils.py`:
```python
from django.utils.encoding import force_text  # This no longer exists in Django 5.x
```

Version 0.6.7 and later have been updated to use `force_str` instead, which is compatible with Django 5.x.

### Verification
To verify the fix works, you can run:
```bash
python manage.py check
```

This should complete without ImportError related to `force_text`.

### No Changes Required in Project Code
The project's own codebase does not use `force_text` anywhere. The issue was solely in the third-party `django-pandas` dependency.
