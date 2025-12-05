"""
Management command to display current feature flag status.
"""

from django.core.management.base import BaseCommand

from recognition.features import FeatureFlags


class Command(BaseCommand):
    help = "Display current feature flag configuration and active profile"

    def handle(self, *args, **options):
        """Display feature flag status."""
        self.stdout.write(self.style.SUCCESS("=== Feature Flags Configuration ===\n"))

        # Display active profile
        profile = FeatureFlags.get_profile()
        self.stdout.write(f"Active Profile: {self.style.WARNING(profile.value.upper())}\n")

        # Profile descriptions
        profile_descriptions = {
            "basic": "Minimal features for simple deployments (<50 employees)",
            "standard": "Recommended for most production deployments",
            "advanced": "Full feature set for enterprise with compliance requirements",
        }
        if profile.value in profile_descriptions:
            self.stdout.write(f"  → {profile_descriptions[profile.value]}\n")

        # Display all feature flags
        self.stdout.write("\n=== Feature Status ===\n")
        flags = FeatureFlags.get_all_flags()

        feature_names = {
            "liveness_detection": "Motion-based Liveness Detection",
            "deepface_antispoofing": "DeepFace Anti-Spoofing",
            "scheduled_evaluations": "Automated Model Evaluations",
            "fairness_audits": "Scheduled Fairness Audits",
            "liveness_evaluations": "Liveness Detection Evaluations",
            "performance_profiling": "Performance Profiling (Silk)",
            "encryption": "Face Data Encryption",
            "sentry": "Sentry Error Tracking",
        }

        for flag_key, enabled in flags.items():
            flag_name = feature_names.get(flag_key, flag_key)
            status = self.style.SUCCESS("✓ Enabled") if enabled else self.style.ERROR("✗ Disabled")
            self.stdout.write(f"{flag_name:.<40} {status}")

        # Recommendations
        self.stdout.write("\n=== Recommendations ===\n")
        if profile.value == "basic":
            self.stdout.write(
                self.style.WARNING(
                    "⚠  Basic profile detected. Consider enabling encryption for production deployments."
                )
            )
            self.stdout.write(
                self.style.WARNING(
                    "⚠  Liveness detection is disabled. Users may attempt spoofing attacks."
                )
            )
        elif profile.value == "standard":
            self.stdout.write(
                self.style.SUCCESS(
                    "✓ Standard profile is recommended for most production deployments."
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    "✓ Advanced profile enabled. All features active for enterprise compliance."
                )
            )

        self.stdout.write("\n=== Configuration ===\n")
        self.stdout.write("To change profiles, set environment variable:")
        self.stdout.write("  export FEATURE_PROFILE=basic|standard|advanced\n")
        self.stdout.write("To override individual features:")
        self.stdout.write("  export ENABLE_LIVENESS_DETECTION=true")
        self.stdout.write("  export ENABLE_SCHEDULED_EVALUATIONS=false\n")
