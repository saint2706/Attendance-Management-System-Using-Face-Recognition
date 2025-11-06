"""
Defines the database models for the recognition app.

This file is the central place to define the data structure of the recognition
app. Each model in this file corresponds to a table in the database. Django's
Object-Relational Mapping (ORM) uses these models to interact with the
underlying database, allowing you to create, retrieve, update, and delete
records using Python objects.

Attributes:
    Each model contains fields that represent database columns, such as
    `CharField` for text, `DateTimeField` for timestamps, and `ForeignKey`
    for relationships with other models. These fields are defined with
    various options to control their behavior, such as `max_length`,
    `default` values, and `on_delete` policies.

Example Usage:
    To create a new model, you would define a class that inherits from
    `django.db.models.Model` and add the desired fields. For example:

    ```python
    from django.contrib.auth.models import User

    class Attendance(models.Model):
        user = models.ForeignKey(User, on_delete=models.CASCADE)
        timestamp = models.DateTimeField(auto_now_add=True)
        is_time_in = models.BooleanField(default=True)
    ```

After defining or modifying a model, you must run the following commands
to apply the changes to the database:
    - `python manage.py makemigrations`
    - `python manage.py migrate`

This ensures that the database schema is kept in sync with your models.
"""

# Create your models here.
