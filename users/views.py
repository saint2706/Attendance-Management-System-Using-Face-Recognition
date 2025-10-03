"""
Views for the users app.

This module contains views related to user management, such as employee registration.
Access to these views is typically restricted to administrators.
"""

from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required


@login_required
def register(request):
    """
    Handle the employee registration process.

    This view allows staff members or superusers to register new employee accounts.
    It uses Django's built-in UserCreationForm to handle user creation.

    - On GET request, it displays the registration form.
    - On POST request, it validates the form data and, if valid, saves the
      new user to the database, displaying a success message.

    If a non-admin user attempts to access this page, they are redirected.
    """
    # Restrict access to staff and superusers only
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()  # Add user to the database
            messages.success(request, "Employee registered successfully!")
            return redirect("dashboard")
    else:
        # For GET requests, create a new, empty form
        form = UserCreationForm()

    # Render the registration page with the form
    return render(request, "users/register.html", {"form": form})





