"""Tests for threshold profile management."""

import os
from io import StringIO

import django
from django.core.management import call_command

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")
if not django.apps.apps.ready:
    django.setup()

from django.test import TestCase  # noqa: E402

from recognition.models import ThresholdProfile  # noqa: E402


class TestThresholdProfileModel(TestCase):
    """Tests for the ThresholdProfile model."""

    def test_create_profile(self):
        """Test creating a basic threshold profile."""
        profile = ThresholdProfile.objects.create(
            name="test_profile",
            distance_threshold=0.35,
            description="Test profile",
        )
        assert profile.name == "test_profile"
        assert profile.distance_threshold == 0.35
        assert profile.is_default is False

    def test_default_profile_uniqueness(self):
        """Test that only one default profile can exist."""
        profile1 = ThresholdProfile.objects.create(
            name="profile1",
            distance_threshold=0.3,
            is_default=True,
        )
        assert profile1.is_default is True

        profile2 = ThresholdProfile.objects.create(
            name="profile2",
            distance_threshold=0.5,
            is_default=True,
        )

        # Refresh profile1 from database
        profile1.refresh_from_db()

        # profile1 should no longer be default
        assert profile1.is_default is False
        assert profile2.is_default is True

    def test_get_for_site(self):
        """Test retrieving profile by site code."""
        ThresholdProfile.objects.create(
            name="site_profile",
            distance_threshold=0.4,
            sites="site1, site2, site3",
        )

        profile = ThresholdProfile.get_for_site("site2")
        assert profile is not None
        assert profile.name == "site_profile"

    def test_get_for_site_returns_default(self):
        """Test that get_for_site returns default when no site match."""
        ThresholdProfile.objects.create(
            name="default_profile",
            distance_threshold=0.4,
            is_default=True,
        )
        ThresholdProfile.objects.create(
            name="site_profile",
            distance_threshold=0.5,
            sites="other_site",
        )

        profile = ThresholdProfile.get_for_site("unknown_site")
        assert profile is not None
        assert profile.name == "default_profile"

    def test_get_threshold_for_site(self):
        """Test getting threshold value for a site."""
        ThresholdProfile.objects.create(
            name="site_profile",
            distance_threshold=0.45,
            sites="lab1, lab2",
        )

        threshold = ThresholdProfile.get_threshold_for_site("lab1")
        assert threshold == 0.45

    def test_get_threshold_for_site_uses_system_default(self):
        """Test that system default is used when no profile matches."""
        from django.conf import settings

        expected = float(getattr(settings, "RECOGNITION_DISTANCE_THRESHOLD", 0.4))
        threshold = ThresholdProfile.get_threshold_for_site("unknown")
        assert threshold == expected

    def test_profile_str_representation(self):
        """Test the string representation of a profile."""
        profile = ThresholdProfile.objects.create(
            name="test",
            distance_threshold=0.35,
        )
        assert "test" in str(profile)
        assert "0.3500" in str(profile)


class TestThresholdProfileCommand(TestCase):
    """Tests for the threshold_profile management command."""

    def test_list_empty(self):
        """Test listing profiles when none exist."""
        out = StringIO()
        call_command("threshold_profile", "list", stdout=out)
        output = out.getvalue()
        assert "No threshold profiles" in output

    def test_list_profiles(self):
        """Test listing existing profiles."""
        ThresholdProfile.objects.create(
            name="test_profile",
            distance_threshold=0.4,
        )

        out = StringIO()
        call_command("threshold_profile", "list", stdout=out)
        output = out.getvalue()
        assert "test_profile" in output
        assert "0.4000" in output

    def test_list_json_output(self):
        """Test JSON output format."""
        ThresholdProfile.objects.create(
            name="json_test",
            distance_threshold=0.35,
        )

        out = StringIO()
        call_command("threshold_profile", "list", "--json", stdout=out)
        import json

        data = json.loads(out.getvalue())
        assert len(data) == 1
        assert data[0]["name"] == "json_test"
        assert data[0]["distance_threshold"] == 0.35

    def test_create_profile(self):
        """Test creating a profile via command."""
        out = StringIO()
        call_command(
            "threshold_profile",
            "create",
            "--name",
            "cmd_profile",
            "--threshold",
            "0.38",
            "--description",
            "Created via CLI",
            stdout=out,
        )

        output = out.getvalue()
        assert "Created profile" in output

        profile = ThresholdProfile.objects.get(name="cmd_profile")
        assert profile.distance_threshold == 0.38
        assert profile.description == "Created via CLI"

    def test_update_profile(self):
        """Test updating a profile via command."""
        ThresholdProfile.objects.create(
            name="update_test",
            distance_threshold=0.4,
        )

        out = StringIO()
        call_command(
            "threshold_profile",
            "update",
            "update_test",
            "--threshold",
            "0.45",
            stdout=out,
        )

        output = out.getvalue()
        assert "Updated profile" in output

        profile = ThresholdProfile.objects.get(name="update_test")
        assert profile.distance_threshold == 0.45

    def test_get_threshold(self):
        """Test get-threshold command."""
        ThresholdProfile.objects.create(
            name="get_test",
            distance_threshold=0.42,
            sites="test_site",
        )

        out = StringIO()
        call_command(
            "threshold_profile",
            "get-threshold",
            "--site",
            "test_site",
            stdout=out,
        )

        output = out.getvalue()
        assert "get_test" in output
        assert "0.4200" in output
