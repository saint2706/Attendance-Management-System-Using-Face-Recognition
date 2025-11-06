"""
This file contains the app configuration for the recognition app.

The `AppConfig` class is used to configure the app's behavior, such as
its name and other settings. Django automatically discovers this
configuration when the app is included in the `INSTALLED_APPS` list
in the project's settings.
"""

from django.apps import AppConfig


class RecognitionConfig(AppConfig):
    """
    Configuration class for the recognition app.

    This class defines the configuration for the recognition app, including
    its name. Django uses this class to manage the app and its models.
    """

    name = "recognition"
