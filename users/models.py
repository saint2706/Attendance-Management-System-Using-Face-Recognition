"""
Database models for the users app.

This module defines the data models for tracking employee attendance, including
daily presence status and specific time-in/time-out events.
"""
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class Present(models.Model):
    """
    Represents the daily attendance status of a user.

    This model tracks whether a user was marked present on a specific date.
    It provides a simple way to query for daily presence without needing to
    analyze the more granular Time model.
    """

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, help_text="The user this attendance record belongs to."
    )
    date = models.DateField(
        default=timezone.localdate, help_text="The date of the attendance record."
    )
    present = models.BooleanField(
        default=False, help_text="Indicates if the user was present on this date."
    )

    def __str__(self):
        """Return a string representation of the attendance record."""
        return f"{self.user.username} - {self.date} - {'Present' if self.present else 'Absent'}"


class Time(models.Model):
    """
    Records a specific time-in or time-out event for a user.

    Each instance of this model represents a single timestamp when a user
    either checked in or checked out. The 'out' field distinguishes between
    the two types of events.
    """

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, help_text="The user this time entry belongs to."
    )
    date = models.DateField(
        default=timezone.localdate, help_text="The date of the time entry."
    )
    time = models.DateTimeField(
        null=True, blank=True, help_text="The exact time of the event."
    )
    out = models.BooleanField(
        default=False, help_text="False for a time-in event, True for a time-out event."
    )

    def __str__(self):
        """Return a string representation of the time entry."""
        event_type = "Time-Out" if self.out else "Time-In"
        if self.time is not None:
            formatted_time = self.time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            formatted_time = "No timestamp recorded"

        return f"{self.user.username} - {formatted_time} - {event_type}"
