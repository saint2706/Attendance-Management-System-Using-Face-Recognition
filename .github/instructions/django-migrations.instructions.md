---
applyTo: "**/migrations/**/*.py"
---

# Django Migration Guidelines

When working with Django migrations, follow these guidelines:

## General Rules

1. **Never edit existing migrations** - Create new migrations for changes
2. **Keep migrations reversible** - Always provide reverse operations when possible
3. **Test migrations** - Run `makemigrations --check --dry-run` before committing
4. **Review auto-generated migrations** - Check for unexpected changes

## Creating Migrations

```bash
# Generate migrations for changes
python manage.py makemigrations

# Generate migration for specific app
python manage.py makemigrations app_name

# Create empty migration for custom operations
python manage.py makemigrations app_name --empty --name=descriptive_name
```

## Migration File Structure

```python
from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('app_name', 'previous_migration'),
    ]

    operations = [
        # Operations go here
    ]
```

## Data Migrations

For data migrations, always include reverse operations:

```python
from django.db import migrations

def forward_func(apps, schema_editor):
    Model = apps.get_model('app_name', 'ModelName')
    # Forward operation
    Model.objects.filter(field='old').update(field='new')

def reverse_func(apps, schema_editor):
    Model = apps.get_model('app_name', 'ModelName')
    # Reverse operation
    Model.objects.filter(field='new').update(field='old')

class Migration(migrations.Migration):
    dependencies = [
        ('app_name', 'previous'),
    ]

    operations = [
        migrations.RunPython(forward_func, reverse_func),
    ]
```

## Best Practices

1. **Small, focused migrations** - One logical change per migration
2. **Descriptive names** - Use meaningful names for custom migrations
3. **No model imports** - Use `apps.get_model()` in data migrations
4. **Database compatibility** - Consider PostgreSQL as the target database

## Squashing Migrations

Only squash migrations when necessary and with team agreement:

```bash
python manage.py squashmigrations app_name start_migration end_migration
```

## CI/CD Integration

The CI workflow checks for missing migrations:

```bash
python manage.py makemigrations --check --dry-run
```

Always ensure this check passes before submitting changes.
