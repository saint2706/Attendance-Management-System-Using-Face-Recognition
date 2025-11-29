"""
Django management command to manage threshold profiles for recognition.

Usage:
    python manage.py threshold_profile list
    python manage.py threshold_profile create --name "strict_office" --threshold 0.3 --default
    python manage.py threshold_profile create --name "lenient_lab" --threshold 0.5 --sites "lab1,lab2"
    python manage.py threshold_profile update "strict_office" --threshold 0.35
    python manage.py threshold_profile delete "old_profile"
    python manage.py threshold_profile import --file reports/selected_threshold.json
    python manage.py threshold_profile export --name "strict_office"
    python manage.py threshold_profile set-default "lenient_lab"
"""

import json
import sys
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from recognition.models import ThresholdProfile


class Command(BaseCommand):
    help = "Manage threshold profiles for face recognition"

    def add_arguments(self, parser):
        subparsers = parser.add_subparsers(dest="action", help="Action to perform")

        # List profiles
        list_parser = subparsers.add_parser("list", help="List all threshold profiles")
        list_parser.add_argument(
            "--json",
            action="store_true",
            help="Output as JSON",
        )

        # Create profile
        create_parser = subparsers.add_parser("create", help="Create a new threshold profile")
        create_parser.add_argument(
            "--name",
            type=str,
            required=True,
            help="Profile name (e.g., 'strict_office')",
        )
        create_parser.add_argument(
            "--threshold",
            type=float,
            required=True,
            help="Recognition distance threshold (lower = stricter)",
        )
        create_parser.add_argument(
            "--description",
            type=str,
            default="",
            help="Profile description",
        )
        create_parser.add_argument(
            "--sites",
            type=str,
            default="",
            help="Comma-separated list of site codes",
        )
        create_parser.add_argument(
            "--method",
            type=str,
            choices=["eer", "f1", "far", "frr", "manual"],
            default="manual",
            help="Threshold selection method",
        )
        create_parser.add_argument(
            "--target-far",
            type=float,
            default=None,
            help="Target False Accept Rate",
        )
        create_parser.add_argument(
            "--target-frr",
            type=float,
            default=None,
            help="Target False Reject Rate",
        )
        create_parser.add_argument(
            "--default",
            action="store_true",
            help="Set as default profile",
        )

        # Update profile
        update_parser = subparsers.add_parser("update", help="Update an existing profile")
        update_parser.add_argument(
            "name",
            type=str,
            help="Profile name to update",
        )
        update_parser.add_argument(
            "--threshold",
            type=float,
            default=None,
            help="New threshold value",
        )
        update_parser.add_argument(
            "--description",
            type=str,
            default=None,
            help="New description",
        )
        update_parser.add_argument(
            "--sites",
            type=str,
            default=None,
            help="New site codes (comma-separated)",
        )
        update_parser.add_argument(
            "--method",
            type=str,
            choices=["eer", "f1", "far", "frr", "manual"],
            default=None,
            help="New selection method",
        )

        # Delete profile
        delete_parser = subparsers.add_parser("delete", help="Delete a threshold profile")
        delete_parser.add_argument(
            "name",
            type=str,
            help="Profile name to delete",
        )
        delete_parser.add_argument(
            "--force",
            action="store_true",
            help="Skip confirmation",
        )

        # Import from JSON
        import_parser = subparsers.add_parser(
            "import", help="Import a profile from threshold selection output"
        )
        import_parser.add_argument(
            "--file",
            type=Path,
            required=True,
            help="Path to threshold JSON file (e.g., reports/selected_threshold.json)",
        )
        import_parser.add_argument(
            "--name",
            type=str,
            default=None,
            help="Profile name (auto-generated if not specified)",
        )
        import_parser.add_argument(
            "--sites",
            type=str,
            default="",
            help="Site codes for this profile",
        )
        import_parser.add_argument(
            "--default",
            action="store_true",
            help="Set as default profile",
        )

        # Export profile
        export_parser = subparsers.add_parser("export", help="Export a profile to JSON")
        export_parser.add_argument(
            "--name",
            type=str,
            required=True,
            help="Profile name to export",
        )
        export_parser.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Output file path (defaults to stdout)",
        )

        # Set default
        default_parser = subparsers.add_parser("set-default", help="Set a profile as default")
        default_parser.add_argument(
            "name",
            type=str,
            help="Profile name to set as default",
        )

        # Get threshold for site
        get_parser = subparsers.add_parser(
            "get-threshold", help="Get threshold for a specific site"
        )
        get_parser.add_argument(
            "--site",
            type=str,
            default="",
            help="Site code",
        )

    def handle(self, *args, **options):
        action = options.get("action")

        if not action:
            self.stdout.write(self.style.ERROR("Please specify an action. Use --help for options."))
            return

        handler = getattr(self, f"handle_{action.replace('-', '_')}", None)
        if handler:
            handler(options)
        else:
            raise CommandError(f"Unknown action: {action}")

    def handle_list(self, options: dict[str, Any]):
        """List all threshold profiles."""
        profiles = ThresholdProfile.objects.all()

        if options.get("json"):
            data = []
            for profile in profiles:
                data.append(
                    {
                        "name": profile.name,
                        "description": profile.description,
                        "distance_threshold": profile.distance_threshold,
                        "target_far": profile.target_far,
                        "target_frr": profile.target_frr,
                        "selection_method": profile.selection_method,
                        "sites": profile.sites,
                        "is_default": profile.is_default,
                        "created_at": profile.created_at.isoformat(),
                        "updated_at": profile.updated_at.isoformat(),
                    }
                )
            self.stdout.write(json.dumps(data, indent=2))
        else:
            if not profiles.exists():
                self.stdout.write(self.style.WARNING("No threshold profiles configured."))
                return

            self.stdout.write("\n" + "=" * 70)
            self.stdout.write("Threshold Profiles")
            self.stdout.write("=" * 70 + "\n")

            for profile in profiles:
                default_mark = " [DEFAULT]" if profile.is_default else ""
                self.stdout.write(
                    self.style.SUCCESS(f"  {profile.name}{default_mark}")
                )
                self.stdout.write(f"    Threshold: {profile.distance_threshold:.4f}")
                if profile.description:
                    self.stdout.write(f"    Description: {profile.description}")
                if profile.sites:
                    self.stdout.write(f"    Sites: {profile.sites}")
                if profile.selection_method:
                    self.stdout.write(f"    Method: {profile.selection_method}")
                if profile.target_far is not None:
                    self.stdout.write(f"    Target FAR: {profile.target_far:.4f}")
                if profile.target_frr is not None:
                    self.stdout.write(f"    Target FRR: {profile.target_frr:.4f}")
                self.stdout.write("")

    def handle_create(self, options: dict[str, Any]):
        """Create a new threshold profile."""
        name = options["name"]

        if ThresholdProfile.objects.filter(name=name).exists():
            raise CommandError(f"Profile '{name}' already exists. Use 'update' instead.")

        profile = ThresholdProfile.objects.create(
            name=name,
            description=options.get("description", ""),
            distance_threshold=options["threshold"],
            target_far=options.get("target_far"),
            target_frr=options.get("target_frr"),
            selection_method=options.get("method", "manual"),
            sites=options.get("sites", ""),
            is_default=options.get("default", False),
        )

        self.stdout.write(
            self.style.SUCCESS(f"\n✓ Created profile '{profile.name}' with threshold {profile.distance_threshold:.4f}")
        )
        if profile.is_default:
            self.stdout.write(self.style.SUCCESS("  Set as default profile"))

    def handle_update(self, options: dict[str, Any]):
        """Update an existing threshold profile."""
        name = options["name"]

        try:
            profile = ThresholdProfile.objects.get(name=name)
        except ThresholdProfile.DoesNotExist:
            raise CommandError(f"Profile '{name}' does not exist.")

        updated_fields = []

        if options.get("threshold") is not None:
            profile.distance_threshold = options["threshold"]
            updated_fields.append("threshold")

        if options.get("description") is not None:
            profile.description = options["description"]
            updated_fields.append("description")

        if options.get("sites") is not None:
            profile.sites = options["sites"]
            updated_fields.append("sites")

        if options.get("method") is not None:
            profile.selection_method = options["method"]
            updated_fields.append("method")

        if updated_fields:
            profile.save()
            self.stdout.write(
                self.style.SUCCESS(f"\n✓ Updated profile '{name}': {', '.join(updated_fields)}")
            )
        else:
            self.stdout.write(self.style.WARNING("No fields to update."))

    def handle_delete(self, options: dict[str, Any]):
        """Delete a threshold profile."""
        name = options["name"]

        try:
            profile = ThresholdProfile.objects.get(name=name)
        except ThresholdProfile.DoesNotExist:
            raise CommandError(f"Profile '{name}' does not exist.")

        if not options.get("force"):
            confirm = input(f"Delete profile '{name}'? [y/N] ")
            if confirm.lower() != "y":
                self.stdout.write("Cancelled.")
                return

        profile.delete()
        self.stdout.write(self.style.SUCCESS(f"\n✓ Deleted profile '{name}'"))

    def handle_import(self, options: dict[str, Any]):
        """Import a profile from threshold selection JSON output."""
        file_path = options["file"]

        if not file_path.exists():
            raise CommandError(f"File not found: {file_path}")

        try:
            with open(file_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise CommandError(f"Invalid JSON: {exc}")

        threshold = data.get("threshold")
        if threshold is None:
            raise CommandError("JSON file must contain a 'threshold' field.")

        method = data.get("method", "manual")
        name = options.get("name")
        if not name:
            name = f"{method}_profile_{threshold:.4f}".replace(".", "_")

        if ThresholdProfile.objects.filter(name=name).exists():
            raise CommandError(f"Profile '{name}' already exists.")

        profile = ThresholdProfile.objects.create(
            name=name,
            description=f"Imported from {file_path.name}",
            distance_threshold=threshold,
            target_far=data.get("actual_far") or data.get("target_far"),
            target_frr=data.get("frr"),
            selection_method=method,
            sites=options.get("sites", ""),
            is_default=options.get("default", False),
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"\n✓ Imported profile '{profile.name}' with threshold {profile.distance_threshold:.4f}"
            )
        )

    def handle_export(self, options: dict[str, Any]):
        """Export a profile to JSON."""
        name = options["name"]

        try:
            profile = ThresholdProfile.objects.get(name=name)
        except ThresholdProfile.DoesNotExist:
            raise CommandError(f"Profile '{name}' does not exist.")

        data = {
            "name": profile.name,
            "description": profile.description,
            "threshold": profile.distance_threshold,
            "target_far": profile.target_far,
            "target_frr": profile.target_frr,
            "method": profile.selection_method,
            "sites": profile.sites,
            "is_default": profile.is_default,
        }

        output_path = options.get("output")
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            self.stdout.write(self.style.SUCCESS(f"\n✓ Exported to {output_path}"))
        else:
            self.stdout.write(json.dumps(data, indent=2))

    def handle_set_default(self, options: dict[str, Any]):
        """Set a profile as the default."""
        name = options["name"]

        try:
            profile = ThresholdProfile.objects.get(name=name)
        except ThresholdProfile.DoesNotExist:
            raise CommandError(f"Profile '{name}' does not exist.")

        profile.is_default = True
        profile.save()

        self.stdout.write(self.style.SUCCESS(f"\n✓ Set '{name}' as default profile"))

    def handle_get_threshold(self, options: dict[str, Any]):
        """Get the threshold for a specific site."""
        site_code = options.get("site", "")

        profile = ThresholdProfile.get_for_site(site_code)
        threshold = ThresholdProfile.get_threshold_for_site(site_code)

        if profile:
            self.stdout.write(f"Site: {site_code or '(default)'}")
            self.stdout.write(f"Profile: {profile.name}")
            self.stdout.write(f"Threshold: {threshold:.4f}")
        else:
            self.stdout.write(f"Site: {site_code or '(default)'}")
            self.stdout.write(f"Profile: (system default)")
            self.stdout.write(f"Threshold: {threshold:.4f}")
