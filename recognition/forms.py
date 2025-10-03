"""
Forms for the recognition app.

This module defines the forms used to capture user input for various features,
such as adding photos for a user, viewing attendance by date, and querying
attendance records for a specific employee over a date range.
"""

from django import forms


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
    The widget provides dropdowns for year, month, and day for easy selection.
    """

    date = forms.DateField(
        widget=forms.SelectDateWidget(
            empty_label=("Choose Year", "Choose Month", "Choose Day")
        )
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
        widget=forms.SelectDateWidget(
            empty_label=("Choose Year", "Choose Month", "Choose Day")
        ),
    )
    date_to = forms.DateField(
        label="To",
        widget=forms.SelectDateWidget(
            empty_label=("Choose Year", "Choose Month", "Choose Day")
        ),
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
        widget=forms.SelectDateWidget(
            empty_label=("Choose Year", "Choose Month", "Choose Day")
        ),
    )
    date_to = forms.DateField(
        label="To",
        widget=forms.SelectDateWidget(
            empty_label=("Choose Year", "Choose Month", "Choose Day")
        ),
    )
