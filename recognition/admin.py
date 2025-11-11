"""Admin registrations for the recognition app."""

from django.contrib import admin

from .models import RecognitionOutcome


@admin.register(RecognitionOutcome)
class RecognitionOutcomeAdmin(admin.ModelAdmin):
    """Expose persisted recognition outcomes for auditing."""

    list_display = ("created_at", "username", "direction", "accepted", "confidence", "source")
    list_filter = ("accepted", "direction", "source")
    search_fields = ("username", "source")
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)
