"""
Admin site configuration for the users app.

This file registers the `Time` and `Present` models with the Django admin site,
making them accessible and manageable through the admin interface. This allows
administrators to view, add, edit, and delete attendance records directly.
"""

from django.contrib import admin
from .models import Time, Present

# Register the Time model to make it available in the Django admin panel.
admin.site.register(Time)

# Register the Present model to make it available in the Django admin panel.
admin.site.register(Present)