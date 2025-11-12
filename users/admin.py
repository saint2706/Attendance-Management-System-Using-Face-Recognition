"""
Admin site configuration for the users app.

This file registers the `Time` and `Present` models with the Django admin site,
making them accessible and manageable through the admin interface. This allows
administrators to view, add, edit, and delete attendance records directly.
"""

from django.contrib import admin

from .models import Present, RecognitionAttempt, Time


@admin.register(RecognitionAttempt)
class RecognitionAttemptAdmin(admin.ModelAdmin):
    """Admin configuration for recognition attempt records."""

    list_display = (
        "created_at",
        "username",
        "direction",
        "site",
        "source",
        "successful",
        "spoof_detected",
        "latency_ms",
    )
    list_filter = ("direction", "site", "source", "successful", "spoof_detected")
    search_fields = ("username", "user__username", "site", "source")
    readonly_fields = ("created_at", "updated_at")


# Register the Time model to make it available in the Django admin panel.
admin.site.register(Time)

# Register the Present model to make it available in the Django admin panel.
admin.site.register(Present)
