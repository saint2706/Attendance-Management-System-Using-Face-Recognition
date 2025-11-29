"""
Forms for the users app.

This module defines forms used in the setup wizard for onboarding
new admin users to the attendance system.
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


# Get common timezones for the dropdown
COMMON_TIMEZONES = [
    ("UTC", "UTC"),
    ("America/New_York", "Eastern Time (US)"),
    ("America/Chicago", "Central Time (US)"),
    ("America/Denver", "Mountain Time (US)"),
    ("America/Los_Angeles", "Pacific Time (US)"),
    ("Europe/London", "London (UK)"),
    ("Europe/Paris", "Paris (France)"),
    ("Europe/Berlin", "Berlin (Germany)"),
    ("Asia/Tokyo", "Tokyo (Japan)"),
    ("Asia/Shanghai", "Shanghai (China)"),
    ("Asia/Kolkata", "Kolkata (India)"),
    ("Asia/Singapore", "Singapore"),
    ("Asia/Dubai", "Dubai (UAE)"),
    ("Australia/Sydney", "Sydney (Australia)"),
    ("Pacific/Auckland", "Auckland (New Zealand)"),
]


class OrgDetailsForm(forms.Form):
    """
    Form for Step 1: Organization Details.

    Captures the organization name and timezone for the attendance system.
    """

    org_name = forms.CharField(
        label="Organization Name",
        max_length=255,
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "e.g., Acme Corporation",
                "autofocus": True,
            }
        ),
        help_text="The name of your company or organization.",
    )
    org_timezone = forms.ChoiceField(
        label="Time Zone",
        choices=COMMON_TIMEZONES,
        widget=forms.Select(attrs={"class": "form-select"}),
        help_text="Select the primary timezone for attendance tracking.",
    )


class CameraTestForm(forms.Form):
    """
    Form for Step 2: Camera & Liveness Test.

    Confirms that the camera and liveness detection are working.
    """

    camera_tested = forms.BooleanField(
        label="Camera is working correctly",
        required=True,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
        help_text="Check this after verifying the camera preview works.",
    )
    liveness_tested = forms.BooleanField(
        label="Liveness detection is working",
        required=True,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
        help_text="Check this after successfully passing the liveness test.",
    )


class AddEmployeeForm(UserCreationForm):
    """
    Form for Step 3: Add First Employee.

    Extended user creation form for adding the first employee.
    """

    class Meta:
        model = User
        fields = ["username", "first_name", "last_name", "email", "password1", "password2"]
        widgets = {
            "username": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "e.g., john.doe"}
            ),
            "first_name": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "First name"}
            ),
            "last_name": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "Last name"}
            ),
            "email": forms.EmailInput(
                attrs={"class": "form-control", "placeholder": "employee@company.com"}
            ),
        }

    def __init__(self, *args, **kwargs):
        """Add Bootstrap classes to password fields."""
        super().__init__(*args, **kwargs)
        self.fields["password1"].widget.attrs.update({"class": "form-control"})
        self.fields["password2"].widget.attrs.update({"class": "form-control"})


class PhotoCaptureConfirmForm(forms.Form):
    """
    Form to confirm photo capture was successful.
    """

    photos_captured = forms.BooleanField(
        label="Photos captured successfully",
        required=True,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
        help_text="Check this after capturing employee photos.",
    )


class TrainingConfirmForm(forms.Form):
    """
    Form for Step 4: Train Model confirmation.
    """

    start_training = forms.BooleanField(
        label="Start model training",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )


class StartSessionForm(forms.Form):
    """
    Form for Step 5: Start First Attendance Session.
    """

    session_type = forms.ChoiceField(
        label="Session Type",
        choices=[
            ("check_in", "Check-in Session"),
            ("check_out", "Check-out Session"),
        ],
        initial="check_in",
        widget=forms.RadioSelect(attrs={"class": "form-check-input"}),
        help_text="Select the type of attendance session to start.",
    )
