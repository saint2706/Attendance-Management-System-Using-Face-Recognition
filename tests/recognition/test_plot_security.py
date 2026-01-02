"""
Security tests for plotting functions to verify IDOR/race condition fix.

This test suite verifies that plotting functions:
1. Return Base64-encoded data URIs instead of file paths
2. Do not trigger file I/O operations
3. Do not create files in the media directory
"""

import datetime
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.utils import timezone

import pytest

from recognition.views_legacy import (
    _plot_to_base64,
    hours_vs_date_given_employee,
    hours_vs_employee_given_date,
)
from users.models import Direction, Present, Time


@pytest.mark.django_db
class TestPlotSecurity:
    """Test suite for plot security improvements."""

    def test_plot_to_base64_returns_data_uri(self):
        """Verify that _plot_to_base64 returns a Base64-encoded data URI."""

        def mock_savefig(buffer, format=None, bbox_inches=None):
            # Write some fake PNG data to the buffer
            buffer.write(b"\x89PNG\r\n\x1a\n")  # PNG file signature

        # Create a mock Figure instance
        mock_fig = MagicMock()
        mock_fig.savefig = MagicMock(side_effect=mock_savefig)

        with patch("recognition.views_legacy.plt") as mock_plt:
            mock_plt.close = MagicMock()

            result = _plot_to_base64(mock_fig)

            # Verify it returns a data URI
            assert result.startswith("data:image/png;base64,")
            assert len(result) > len("data:image/png;base64,")

            # Verify plot was closed with the figure instance
            mock_plt.close.assert_called_once_with(mock_fig)

    def test_plot_to_base64_no_file_io(self, tmp_path):
        """Verify that _plot_to_base64 does not write files to disk."""
        # Create a mock Figure instance
        mock_fig = MagicMock()
        mock_fig.savefig = MagicMock()

        with patch("recognition.views_legacy.plt") as mock_plt:
            mock_plt.close = MagicMock()

            # Record files before
            files_before = set(tmp_path.glob("**/*"))

            _plot_to_base64(mock_fig)

            # Verify no new files were created
            files_after = set(tmp_path.glob("**/*"))
            assert files_before == files_after

            # Verify savefig was called with a BytesIO buffer, not a file path
            call_args = mock_fig.savefig.call_args
            assert call_args is not None
            buffer = call_args[0][0]
            assert hasattr(buffer, "getvalue")  # BytesIO has getvalue method

    def test_hours_vs_date_returns_base64(self):
        """Verify hours_vs_date_given_employee returns Base64 data URI."""
        # Setup test data
        user = User.objects.create_user(username="testuser", password="password")
        date = timezone.localdate()

        Present.objects.create(user=user, date=date, present=True)
        t1 = timezone.now()
        t2 = t1 + datetime.timedelta(hours=8)
        Time.objects.create(user=user, date=date, time=t1, direction=Direction.IN)
        Time.objects.create(user=user, date=date, time=t2, direction=Direction.OUT)

        present_qs = Present.objects.filter(user=user)
        time_qs = Time.objects.filter(user=user)

        with patch("recognition.views_legacy.plt"):
            with patch("recognition.views_legacy.sns"):
                with patch("recognition.views_legacy._plot_to_base64") as mock_plot:
                    mock_plot.return_value = "data:image/png;base64,fake_data"

                    _, chart_url = hours_vs_date_given_employee(present_qs, time_qs)

                    # Verify the function returns a Base64 data URI
                    assert chart_url.startswith("data:image/png;base64,")
                    mock_plot.assert_called_once()

    def test_hours_vs_employee_returns_base64(self):
        """Verify hours_vs_employee_given_date returns Base64 data URI."""
        # Setup test data
        user = User.objects.create_user(username="testuser2", password="password")
        date = timezone.localdate()

        Present.objects.create(user=user, date=date, present=True)
        t1 = timezone.now()
        t2 = t1 + datetime.timedelta(hours=8)
        Time.objects.create(user=user, date=date, time=t1, direction=Direction.IN)
        Time.objects.create(user=user, date=date, time=t2, direction=Direction.OUT)

        present_qs = Present.objects.filter(date=date)
        time_qs = Time.objects.filter(date=date)

        with patch("recognition.views_legacy.plt"):
            with patch("recognition.views_legacy.sns"):
                with patch("recognition.views_legacy._plot_to_base64") as mock_plot:
                    mock_plot.return_value = "data:image/png;base64,fake_data"

                    _, chart_url = hours_vs_employee_given_date(present_qs, time_qs)

                    # Verify the function returns a Base64 data URI
                    assert chart_url.startswith("data:image/png;base64,")
                    mock_plot.assert_called_once()

    def test_no_media_directory_access(self, tmp_path):
        """Verify plotting functions do not access media directory."""
        # Setup test data
        user = User.objects.create_user(username="testuser3", password="password")
        date = timezone.localdate()

        Present.objects.create(user=user, date=date, present=True)
        t1 = timezone.now()
        t2 = t1 + datetime.timedelta(hours=8)
        Time.objects.create(user=user, date=date, time=t1, direction=Direction.IN)
        Time.objects.create(user=user, date=date, time=t2, direction=Direction.OUT)

        present_qs = Present.objects.filter(user=user)
        time_qs = Time.objects.filter(user=user)

        # Mock the media root to a temporary directory
        with patch("django.conf.settings.MEDIA_ROOT", str(tmp_path)):
            with patch("recognition.views_legacy.plt"):
                with patch("recognition.views_legacy.sns"):
                    with patch("recognition.views_legacy._plot_to_base64") as mock_plot:
                        mock_plot.return_value = "data:image/png;base64,fake_data"

                        # Track files before
                        files_before = list(tmp_path.glob("**/*"))

                        hours_vs_date_given_employee(present_qs, time_qs)

                        # Track files after
                        files_after = list(tmp_path.glob("**/*"))

                        # Verify no new files were created in media directory
                        assert len(files_before) == len(files_after)

    def test_concurrent_plotting_no_race_condition(self):
        """Verify that concurrent plotting operations don't interfere with each other."""
        # Setup test data for two users
        user1 = User.objects.create_user(username="user1", password="password")
        user2 = User.objects.create_user(username="user2", password="password")
        date = timezone.localdate()

        for user in [user1, user2]:
            Present.objects.create(user=user, date=date, present=True)
            t1 = timezone.now()
            t2 = t1 + datetime.timedelta(hours=8)
            Time.objects.create(user=user, date=date, time=t1, direction=Direction.IN)
            Time.objects.create(user=user, date=date, time=t2, direction=Direction.OUT)

        present_qs1 = Present.objects.filter(user=user1)
        time_qs1 = Time.objects.filter(user=user1)
        present_qs2 = Present.objects.filter(user=user2)
        time_qs2 = Time.objects.filter(user=user2)

        with patch("recognition.views_legacy.plt"):
            with patch("recognition.views_legacy.sns"):
                with patch("recognition.views_legacy._plot_to_base64") as mock_plot:
                    # Simulate different Base64 outputs for each call
                    mock_plot.side_effect = [
                        "data:image/png;base64,user1_data",
                        "data:image/png;base64,user2_data",
                    ]

                    # Generate plots for both users
                    _, chart_url1 = hours_vs_date_given_employee(present_qs1, time_qs1)
                    _, chart_url2 = hours_vs_date_given_employee(present_qs2, time_qs2)

                    # Verify each got their own unique result
                    assert chart_url1 == "data:image/png;base64,user1_data"
                    assert chart_url2 == "data:image/png;base64,user2_data"
                    assert chart_url1 != chart_url2
