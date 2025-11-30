"""Admin registrations for the recognition app."""

from django.contrib import admin

from .models import LivenessResult, RecognitionOutcome, ThresholdProfile


@admin.register(RecognitionOutcome)
class RecognitionOutcomeAdmin(admin.ModelAdmin):
    """Expose persisted recognition outcomes for auditing."""

    list_display = (
        "created_at",
        "username",
        "direction",
        "accepted",
        "confidence",
        "liveness_confidence",
        "liveness_passed",
        "profile_name",
        "source",
    )
    list_filter = ("accepted", "direction", "source", "liveness_passed", "profile_name")
    search_fields = ("username", "source", "profile_name")
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)


@admin.register(ThresholdProfile)
class ThresholdProfileAdmin(admin.ModelAdmin):
    """Manage threshold profiles for per-site recognition tuning."""

    list_display = (
        "name",
        "distance_threshold",
        "selection_method",
        "sites",
        "is_default",
        "updated_at",
    )
    list_filter = ("is_default", "selection_method")
    search_fields = ("name", "description", "sites")
    ordering = ("-is_default", "name")
    readonly_fields = ("created_at", "updated_at")
    fieldsets = (
        (
            None,
            {
                "fields": ("name", "description", "is_default"),
            },
        ),
        (
            "Threshold Settings",
            {
                "fields": ("distance_threshold", "selection_method", "target_far", "target_frr"),
            },
        ),
        (
            "Site Assignment",
            {
                "fields": ("sites",),
                "description": "Comma-separated list of site codes where this profile applies.",
            },
        ),
        (
            "Metadata",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )


@admin.register(LivenessResult)
class LivenessResultAdmin(admin.ModelAdmin):
    """Audit liveness check results."""

    list_display = (
        "created_at",
        "username",
        "challenge_type",
        "challenge_status",
        "liveness_confidence",
        "motion_score",
        "frames_analyzed",
        "source",
    )
    list_filter = ("challenge_type", "challenge_status", "source")
    search_fields = ("username", "site")
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)
