"""
Django settings for the Smart Attendance System project.

This file contains the configuration for the Django project, including database settings,
installed applications, middleware, and custom application-specific parameters.
It is configured to read sensitive values from environment variables for security.
"""

import os
from pathlib import Path

# Define the project's base directory.
# `BASE_DIR` points to the root of the Django project.
BASE_DIR = Path(__file__).resolve().parent.parent


# --- Security Settings ---

# SECRET_KEY: A secret key used for cryptographic signing.
# It is crucial to keep this key secret in a production environment.
# The value is read from an environment variable, with a default for development.
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY", "a-secure-default-key-for-development-only"
)

# DEBUG: A boolean that turns on/off debug mode.
# Never run with debug mode turned on in a production environment.
# The value is read from an environment variable, defaulting to True for development.
DEBUG = os.environ.get("DJANGO_DEBUG", "True") == "True"

# ALLOWED_HOSTS: A list of strings representing the host/domain names that this Django site can serve.
# In development, '*' is permissive, but this should be locked down in production.
ALLOWED_HOSTS = ["*"]


# --- Application Configuration ---

INSTALLED_APPS = [
    # Custom applications for this project
    "users.apps.UsersConfig",
    "recognition.apps.RecognitionConfig",
    # Third-party packages
    "crispy_forms",
    "crispy_bootstrap5",
    # Core Django applications
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# The root URL configuration module for the project.
ROOT_URLCONF = "attendance_system_facial_recognition.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# WSGI application entry point for production servers.
WSGI_APPLICATION = "attendance_system_facial_recognition.wsgi.application"


# --- Database Configuration ---
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# --- Password Validation ---
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# --- Internationalization ---
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"  # Set to the appropriate timezone
USE_I18N = True
USE_TZ = True  # Enable timezone-aware datetimes


# --- Static Files Configuration ---
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = "/static/"
STATICFILES_DIRS = [
    BASE_DIR / "static",
]

# --- Crispy Forms Configuration ---

# Specifies that django-crispy-forms should use Bootstrap 5 templates.
CRISPY_TEMPLATE_PACK = "bootstrap5"

# --- Authentication and Redirects ---

# The URL to redirect to for login when using the @login_required decorator.
LOGIN_URL = "login"

# The URL to redirect to after a user logs out.
LOGOUT_REDIRECT_URL = "home"

# The default URL to redirect to after a user logs in.
LOGIN_REDIRECT_URL = "dashboard"

# --- Model Field Configuration ---

# Specifies the default primary key field type for new models.
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Custom Application Settings ---

# Threshold for accepting DeepFace matches when marking attendance.
# Lower values (e.g., 0.3) mean stricter matching, while higher values (e.g., 0.5)
# are more permissive. This can be overridden via an environment variable.
RECOGNITION_DISTANCE_THRESHOLD = float(
    os.environ.get("RECOGNITION_DISTANCE_THRESHOLD", "0.4")
)
