"""
Database models for the users app.

This module defines the data models for tracking employee attendance, including
daily presence status and specific time-in/time-out events.
"""

import datetime

from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone


class Present(models.Model):
    """
    Represents the daily attendance status of a user.

    This model tracks whether a user was marked present on a specific date.
    It provides a simple way to query for daily presence without needing to
    analyze the more granular Time model.
    """

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        help_text="The user this attendance record belongs to.",
        db_index=True,
    )
    date = models.DateField(
        default=timezone.localdate,
        help_text="The date of the attendance record.",
        db_index=True,
    )
    present = models.BooleanField(
        default=False, help_text="Indicates if the user was present on this date."
    )

    # Non-persistent fields for calculated values
    time_in: datetime.datetime | None = None
    time_out: datetime.datetime | None = None
    hours: str | int = "0 hrs 0 mins"
    break_hours: str | float = "0 hrs 0 mins"

    class Meta:
        indexes = [
            models.Index(fields=["user", "date"], name="users_present_user_date_idx"),
            models.Index(fields=["date", "user"], name="users_present_date_user_idx"),
        ]

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
        User,
        on_delete=models.CASCADE,
        help_text="The user this time entry belongs to.",
        db_index=True,
    )
    date = models.DateField(
        default=timezone.localdate,
        help_text="The date of the time entry.",
        db_index=True,
    )
    time = models.DateTimeField(
        null=True,
        blank=True,
        help_text="The exact time of the event.",
        db_index=True,
    )
    out = models.BooleanField(
        default=False, help_text="False for a time-in event, True for a time-out event."
    )

    class Meta:
        indexes = [
            models.Index(fields=["user", "date"], name="users_time_user_date_idx"),
            models.Index(fields=["date", "user"], name="users_time_date_user_idx"),
        ]

    def __str__(self):
        """Return a string representation of the time entry."""
        event_type = "Time-Out" if self.out else "Time-In"
        if self.time is not None:
            formatted_time = self.time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            formatted_time = "No timestamp recorded"

        return f"{self.user.username} - {formatted_time} - {event_type}"


class SetupWizardProgress(models.Model):
    """
    Tracks the progress of the setup wizard for each admin user.

    This model stores the wizard state including the current step,
    completion status, and any collected data from each step.
    """

    class Step(models.IntegerChoices):
        """Wizard steps."""

        ORG_DETAILS = 1, "Organization Details"
        CAMERA_TEST = 2, "Camera & Liveness Test"
        ADD_EMPLOYEE = 3, "Add First Employee"
        TRAIN_MODEL = 4, "Train Recognition Model"
        START_SESSION = 5, "Start Attendance Session"

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="setup_wizard_progress",
        help_text="The admin user this wizard progress belongs to.",
    )
    current_step = models.IntegerField(
        choices=Step.choices,
        default=Step.ORG_DETAILS,
        help_text="The current step in the setup wizard.",
    )
    completed = models.BooleanField(
        default=False,
        help_text="Whether the wizard has been completed.",
    )
    org_name = models.CharField(
        max_length=255,
        blank=True,
        help_text="Organization name entered in step 1.",
    )
    org_timezone = models.CharField(
        max_length=100,
        blank=True,
        help_text="Organization timezone entered in step 1.",
    )
    camera_tested = models.BooleanField(
        default=False,
        help_text="Whether the camera test was passed in step 2.",
    )
    liveness_tested = models.BooleanField(
        default=False,
        help_text="Whether the liveness test was passed in step 2.",
    )
    first_employee_username = models.CharField(
        max_length=150,
        blank=True,
        help_text="Username of the first employee added in step 3.",
    )
    first_employee_photos_captured = models.BooleanField(
        default=False,
        help_text="Whether photos were captured for the first employee in step 3.",
    )
    model_trained = models.BooleanField(
        default=False,
        help_text="Whether the model was trained in step 4.",
    )
    training_task_id = models.CharField(
        max_length=255,
        blank=True,
        help_text="Celery task ID for the training job.",
    )
    first_session_started = models.BooleanField(
        default=False,
        help_text="Whether the first attendance session was started in step 5.",
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When the wizard progress was created.",
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="When the wizard progress was last updated.",
    )

    class Meta:
        verbose_name = "Setup Wizard Progress"
        verbose_name_plural = "Setup Wizard Progress"

    def __str__(self) -> str:
        """Return a human readable representation."""
        step_display = self.get_current_step_display()
        status = "completed" if self.completed else f"step {self.current_step}"
        return f"{self.user.username} - {status} ({step_display})"

    def can_proceed_to_step(self, step: int) -> bool:
        """Check if the user can proceed to the given step."""
        if step == self.Step.ORG_DETAILS:
            return True
        if step == self.Step.CAMERA_TEST:
            return bool(self.org_name and self.org_timezone)
        if step == self.Step.ADD_EMPLOYEE:
            return self.camera_tested and self.liveness_tested
        if step == self.Step.TRAIN_MODEL:
            return bool(self.first_employee_username and self.first_employee_photos_captured)
        if step == self.Step.START_SESSION:
            return self.model_trained
        return False

    def get_step_status(self, step: int) -> str:
        """Return the status of a specific step."""
        if step < self.current_step:
            return "completed"
        if step == self.current_step:
            return "current"
        if self.can_proceed_to_step(step):
            return "available"
        return "locked"


class RecognitionAttempt(models.Model):
    """Persist metadata for each recognition attempt."""

    class Direction(models.TextChoices):
        """Supported attendance directions for recognition attempts."""

        IN = "in", "Check-in"
        OUT = "out", "Check-out"

    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When the recognition attempt was recorded.",
        db_index=True,
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp for the last update to this record.",
    )
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="recognition_attempts",
        help_text="Resolved user for the attempt, if available.",
    )
    username = models.CharField(
        max_length=150,
        blank=True,
        help_text="Username inferred at recognition time (may be empty on failure).",
    )
    direction = models.CharField(
        max_length=3,
        choices=Direction.choices,
        help_text="Whether the attempt was for a check-in or check-out event.",
    )
    site = models.CharField(
        max_length=255,
        blank=True,
        help_text="Physical location or site identifier for the attempt.",
    )
    source = models.CharField(
        max_length=64,
        blank=True,
        help_text="System component that initiated the attempt (e.g. webcam, API).",
    )
    successful = models.BooleanField(
        default=False,
        help_text="True when the attempt resulted in a successful recognition.",
    )
    spoof_detected = models.BooleanField(
        default=False,
        help_text="True when anti-spoofing blocked the attempt.",
    )
    latency_ms = models.FloatField(
        null=True,
        blank=True,
        help_text="End-to-end latency in milliseconds measured for the attempt.",
    )
    present_record = models.ForeignKey(
        Present,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="recognition_attempts",
        help_text="Linked daily presence record when one was created.",
    )
    time_record = models.ForeignKey(
        Time,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="recognition_attempts",
        help_text="Linked time entry created for the attempt, if applicable.",
    )
    error_message = models.TextField(
        blank=True,
        help_text="Optional diagnostic information describing failures.",
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["site", "direction"], name="users_attempt_site_dir_idx"),
            models.Index(fields=["user", "direction"], name="users_attempt_user_dir_idx"),
            models.Index(fields=["created_at"], name="users_attempt_created_idx"),
        ]

    def __str__(self) -> str:
        """Return a human readable representation for the attempt."""

        label = self.username or (self.user.username if self.user else "unknown")
        status = "success" if self.successful else "failure"
        direction = self.get_direction_display()
        return f"{label} - {direction} - {status}"
