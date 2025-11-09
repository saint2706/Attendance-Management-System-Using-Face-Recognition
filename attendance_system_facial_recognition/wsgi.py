"""
WSGI config for attendance_system_facial_recognition project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

# WSGI is typically invoked by the production application server, so default to
# the hardened production settings. Developers running local WSGI servers can
# override this by exporting DJANGO_SETTINGS_MODULE=attendance_system_facial_recognition.settings.
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "attendance_system_facial_recognition.settings.production",
)

application = get_wsgi_application()
