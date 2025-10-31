"""
This file contains the app configuration for the users app.

The `AppConfig` class is used to configure the app's behavior, such as
its name and other settings. Django automatically discovers this
configuration when the app is included in the `INSTALLED_APPS` list
in the project's settings.
"""
from django.apps import AppConfig


class UsersConfig(AppConfig):
    """
    Configuration class for the users app.

    This class defines the configuration for the users app, including its
    name. Django uses this class to manage the app and its models.
    """
    name = 'users'
