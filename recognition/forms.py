"""
Forms for the recognition app.

This module defines the forms used to capture user input for various features,
such as adding photos for a user, viewing attendance by date, and querying
attendance records for a specific employee over a date range.
"""

from django import forms

from .models import ThresholdProfile


class usernameForm(forms.Form):
    """
    A simple form to capture a username.

    This is used in the 'Add Photos' feature to specify which user's
    face dataset is being created.
    """

    username = forms.CharField(label="Username", max_length=150)


class DateForm(forms.Form):
    """
    A form to select a single date.

    Used by admins to view attendance records for all employees on a specific day.
    The widget provides an HTML5 date picker for easy selection.
    """

    date = forms.DateField(
        widget=forms.DateInput(attrs={"type": "date"}),
        input_formats=["%Y-%m-%d"],
    )


class UsernameAndDateForm(forms.Form):
    """
    A form to select a username and a date range.

    This form is used by admins to query the attendance history for a specific
    employee between a start date ('date_from') and an end date ('date_to').
    """

    username = forms.CharField(label="Username", max_length=150)
    date_from = forms.DateField(
        label="From",
        widget=forms.DateInput(attrs={"type": "date"}),
        input_formats=["%Y-%m-%d"],
    )
    date_to = forms.DateField(
        label="To",
        widget=forms.DateInput(attrs={"type": "date"}),
        input_formats=["%Y-%m-%d"],
    )


class DateForm_2(forms.Form):
    """
    A form for selecting a date range, intended for non-admin users.

    This form is used by employees to view their own attendance records.
    It is functionally similar to UsernameAndDateForm but omits the username field,
    as the user is inferred from the current session.
    """

    date_from = forms.DateField(
        label="From",
        widget=forms.DateInput(attrs={"type": "date"}),
        input_formats=["%Y-%m-%d"],
    )
    date_to = forms.DateField(
        label="To",
        widget=forms.DateInput(attrs={"type": "date"}),
        input_formats=["%Y-%m-%d"],
    )


class AttendanceSessionFilterForm(forms.Form):
    """
    A form for filtering attendance sessions and recognition outcomes.

    This form is used by admins to filter sessions by date range, employee,
    and recognition outcome (success, liveness fail, low confidence).
    """

    OUTCOME_CHOICES = [
        ("", "All outcomes"),
        ("success", "Success"),
        ("liveness_fail", "Liveness failed"),
        ("low_confidence", "Low confidence / rejected"),
    ]

    date_from = forms.DateField(
        label="From",
        required=False,
        widget=forms.DateInput(attrs={"type": "date", "class": "form-control"}),
        input_formats=["%Y-%m-%d"],
    )
    date_to = forms.DateField(
        label="To",
        required=False,
        widget=forms.DateInput(attrs={"type": "date", "class": "form-control"}),
        input_formats=["%Y-%m-%d"],
    )
    employee = forms.CharField(
        label="Employee",
        required=False,
        max_length=150,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Username"}),
    )
    outcome = forms.ChoiceField(
        label="Outcome",
        required=False,
        choices=OUTCOME_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )


class ThresholdProfileForm(forms.ModelForm):
    """Form for creating and editing threshold profiles."""

    class Meta:
        model = ThresholdProfile
        fields = [
            "name",
            "description",
            "distance_threshold",
            "selection_method",
            "target_far",
            "target_frr",
            "sites",
            "is_default",
        ]
        widgets = {
            "name": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "e.g., strict_office",
            }),
            "description": forms.Textarea(attrs={
                "class": "form-control",
                "rows": 3,
                "placeholder": "Description of when/where this profile should be used",
            }),
            "distance_threshold": forms.NumberInput(attrs={
                "class": "form-control",
                "step": "0.01",
                "min": "0",
                "max": "2",
            }),
            "selection_method": forms.Select(attrs={"class": "form-select"}),
            "target_far": forms.NumberInput(attrs={
                "class": "form-control",
                "step": "0.001",
                "min": "0",
                "max": "1",
                "placeholder": "e.g., 0.01",
            }),
            "target_frr": forms.NumberInput(attrs={
                "class": "form-control",
                "step": "0.001",
                "min": "0",
                "max": "1",
                "placeholder": "e.g., 0.05",
            }),
            "sites": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "e.g., site1, site2, lab_a",
            }),
            "is_default": forms.CheckboxInput(attrs={"class": "form-check-input"}),
        }
        help_texts = {
            "distance_threshold": "Lower values mean stricter matching (0.3-0.5 typical)",
            "sites": "Comma-separated list of site codes",
        }


class ThresholdImportForm(forms.Form):
    """Form for importing a threshold from evaluation artifacts."""

    IMPORT_CHOICES = [
        ("eer", "Equal Error Rate (EER)"),
        ("f1", "Optimal F1 Score"),
        ("far", "Target False Accept Rate"),
    ]

    import_method = forms.ChoiceField(
        label="Import Method",
        choices=IMPORT_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    profile_name = forms.CharField(
        label="Profile Name",
        max_length=100,
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "e.g., strict_office",
        }),
    )
    sites = forms.CharField(
        label="Sites",
        required=False,
        max_length=500,
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "e.g., site1, site2",
        }),
    )
    set_as_default = forms.BooleanField(
        label="Set as default profile",
        required=False,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )
