"""
Views for the users app.

This module contains views related to user management, such as employee registration
and the setup wizard for onboarding new administrators.
"""

import logging
from typing import Any

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from .forms import (
    AddEmployeeForm,
    CameraTestForm,
    OrgDetailsForm,
    PhotoCaptureConfirmForm,
    StartSessionForm,
    TrainingConfirmForm,
)
from .models import SetupWizardProgress

logger = logging.getLogger(__name__)


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
    # Restrict access to staff and superusers only.
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    if request.method == "POST":
        # If the form has been submitted, process the data.
        form = UserCreationForm(request.POST)
        if form.is_valid():
            # If the form is valid, save the new user to the database.
            form.save()
            messages.success(request, "Employee registered successfully!")
            # Redirect to the dashboard after successful registration.
            return redirect("dashboard")
    else:
        # For GET requests, create a new, empty form.
        form = UserCreationForm()

    # Render the registration page with the form.
    return render(request, "users/register.html", {"form": form})


def _get_or_create_wizard_progress(user) -> SetupWizardProgress:
    """Get or create wizard progress for the given user."""
    progress, _ = SetupWizardProgress.objects.get_or_create(user=user)
    return progress


def _build_wizard_context(progress: SetupWizardProgress) -> dict[str, Any]:
    """Build common context for wizard templates."""
    steps = [
        {
            "number": step.value,
            "name": step.label,
            "status": progress.get_step_status(step.value),
        }
        for step in SetupWizardProgress.Step
    ]
    return {
        "progress": progress,
        "steps": steps,
        "current_step": progress.current_step,
        "total_steps": len(SetupWizardProgress.Step),
    }


@login_required
def setup_wizard(request):
    """
    Main entry point for the setup wizard.

    Redirects to the appropriate step based on the user's progress.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    progress = _get_or_create_wizard_progress(request.user)

    if progress.completed:
        messages.info(request, "Setup wizard already completed. Redirecting to dashboard.")
        return redirect("dashboard")

    # Redirect to the current step
    step_urls = {
        SetupWizardProgress.Step.ORG_DETAILS: "setup-wizard-step1",
        SetupWizardProgress.Step.CAMERA_TEST: "setup-wizard-step2",
        SetupWizardProgress.Step.ADD_EMPLOYEE: "setup-wizard-step3",
        SetupWizardProgress.Step.TRAIN_MODEL: "setup-wizard-step4",
        SetupWizardProgress.Step.START_SESSION: "setup-wizard-step5",
    }

    return redirect(step_urls.get(progress.current_step, "setup-wizard-step1"))


@login_required
def setup_wizard_step1(request):
    """
    Step 1: Organization Details & Time Zone.

    Collects basic organization information.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    progress = _get_or_create_wizard_progress(request.user)

    if progress.completed:
        return redirect("dashboard")

    if request.method == "POST":
        form = OrgDetailsForm(request.POST)
        if form.is_valid():
            progress.org_name = form.cleaned_data["org_name"]
            progress.org_timezone = form.cleaned_data["org_timezone"]
            progress.current_step = SetupWizardProgress.Step.CAMERA_TEST
            progress.save()
            messages.success(request, "Organization details saved!")
            return redirect("setup-wizard-step2")
    else:
        form = OrgDetailsForm(
            initial={
                "org_name": progress.org_name,
                "org_timezone": progress.org_timezone or "UTC",
            }
        )

    context = _build_wizard_context(progress)
    context["form"] = form
    context["step_title"] = "Organization Details"
    context["step_description"] = (
        "Let's start by setting up your organization's basic information."
    )

    return render(request, "users/setup_wizard/step1_org_details.html", context)


@login_required
def setup_wizard_step2(request):
    """
    Step 2: Camera & Liveness Test.

    Verifies that the camera and liveness detection work correctly.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    progress = _get_or_create_wizard_progress(request.user)

    if progress.completed:
        return redirect("dashboard")

    if not progress.can_proceed_to_step(SetupWizardProgress.Step.CAMERA_TEST):
        messages.warning(request, "Please complete the previous step first.")
        return redirect("setup-wizard-step1")

    if request.method == "POST":
        form = CameraTestForm(request.POST)
        if form.is_valid():
            progress.camera_tested = form.cleaned_data["camera_tested"]
            progress.liveness_tested = form.cleaned_data["liveness_tested"]
            progress.current_step = SetupWizardProgress.Step.ADD_EMPLOYEE
            progress.save()
            messages.success(request, "Camera and liveness test completed!")
            return redirect("setup-wizard-step3")
    else:
        form = CameraTestForm(
            initial={
                "camera_tested": progress.camera_tested,
                "liveness_tested": progress.liveness_tested,
            }
        )

    context = _build_wizard_context(progress)
    context["form"] = form
    context["step_title"] = "Camera & Liveness Test"
    context["step_description"] = (
        "Test your camera and verify the liveness detection is working properly."
    )

    return render(request, "users/setup_wizard/step2_camera_test.html", context)


@login_required
def setup_wizard_step3(request):
    """
    Step 3: Add First Employee & Capture Photos.

    Creates the first employee and captures their photos for recognition.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    progress = _get_or_create_wizard_progress(request.user)

    if progress.completed:
        return redirect("dashboard")

    if not progress.can_proceed_to_step(SetupWizardProgress.Step.ADD_EMPLOYEE):
        messages.warning(request, "Please complete the camera test first.")
        return redirect("setup-wizard-step2")

    # Check if employee already created but photos not captured
    employee_created = bool(progress.first_employee_username)

    if request.method == "POST":
        if "create_employee" in request.POST:
            form = AddEmployeeForm(request.POST)
            if form.is_valid():
                user = form.save()
                progress.first_employee_username = user.username
                progress.save()
                messages.success(
                    request, f"Employee '{user.username}' created successfully!"
                )
                return redirect("setup-wizard-step3")
        elif "confirm_photos" in request.POST:
            photo_form = PhotoCaptureConfirmForm(request.POST)
            if photo_form.is_valid() and photo_form.cleaned_data["photos_captured"]:
                progress.first_employee_photos_captured = True
                progress.current_step = SetupWizardProgress.Step.TRAIN_MODEL
                progress.save()
                messages.success(request, "Photos captured and saved!")
                return redirect("setup-wizard-step4")
    else:
        form = AddEmployeeForm()

    context = _build_wizard_context(progress)
    context["form"] = form
    context["employee_created"] = employee_created
    context["first_employee_username"] = progress.first_employee_username
    context["photos_captured"] = progress.first_employee_photos_captured
    context["photo_form"] = PhotoCaptureConfirmForm()
    context["step_title"] = "Add First Employee"
    context["step_description"] = (
        "Create your first employee account and capture their face photos."
    )

    return render(request, "users/setup_wizard/step3_add_employee.html", context)


@login_required
def setup_wizard_step4(request):
    """
    Step 4: Run First Training Job.

    Trains the face recognition model with the captured photos.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    progress = _get_or_create_wizard_progress(request.user)

    if progress.completed:
        return redirect("dashboard")

    if not progress.can_proceed_to_step(SetupWizardProgress.Step.TRAIN_MODEL):
        messages.warning(request, "Please add an employee and capture photos first.")
        return redirect("setup-wizard-step3")

    task_context = None
    if progress.training_task_id:
        try:
            from celery.result import AsyncResult

            result = AsyncResult(progress.training_task_id)
            task_context = {
                "task_id": progress.training_task_id,
                "status": result.status,
                "ready": result.ready(),
                "successful": result.successful(),
            }
            if result.successful():
                progress.model_trained = True
                progress.save()
        except ImportError:
            logger.warning("Celery is not available; task status cannot be retrieved.")
        except Exception:
            logger.debug("Could not fetch task status", exc_info=True)

    if request.method == "POST":
        if "start_training" in request.POST:
            try:
                from recognition.tasks import train_recognition_model

                async_result = train_recognition_model.delay(
                    initiated_by=request.user.username
                )
                progress.training_task_id = async_result.id
                progress.save()
                messages.info(
                    request,
                    f"Training started! Task ID: {async_result.id}",
                )
                return redirect("setup-wizard-step4")
            except ImportError:
                logger.exception("Celery tasks not available")
                messages.error(
                    request, "Training service not available. Please check configuration."
                )
            except Exception:
                logger.exception("Failed to start training")
                messages.error(request, "Failed to start training. Please try again.")
        elif "continue" in request.POST and progress.model_trained:
            progress.current_step = SetupWizardProgress.Step.START_SESSION
            progress.save()
            return redirect("setup-wizard-step5")

    context = _build_wizard_context(progress)
    context["step_title"] = "Train Recognition Model"
    context["step_description"] = (
        "Train the AI model to recognize faces from the captured photos."
    )
    context["task"] = task_context
    context["model_trained"] = progress.model_trained
    context["form"] = TrainingConfirmForm()

    return render(request, "users/setup_wizard/step4_train_model.html", context)


@login_required
def setup_wizard_step5(request):
    """
    Step 5: Start First Attendance Session.

    Starts the first attendance session to complete the wizard.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    progress = _get_or_create_wizard_progress(request.user)

    if progress.completed:
        return redirect("dashboard")

    if not progress.can_proceed_to_step(SetupWizardProgress.Step.START_SESSION):
        messages.warning(request, "Please train the model first.")
        return redirect("setup-wizard-step4")

    if request.method == "POST":
        form = StartSessionForm(request.POST)
        if form.is_valid():
            progress.first_session_started = True
            progress.completed = True
            progress.save()
            messages.success(
                request,
                "Congratulations! Setup wizard completed. You can now use the attendance system.",
            )

            # Redirect to the appropriate attendance marking page
            session_type = form.cleaned_data["session_type"]
            if session_type == "check_in":
                return redirect("mark-your-attendance")
            return redirect("mark-your-attendance-out")
    else:
        form = StartSessionForm()

    context = _build_wizard_context(progress)
    context["form"] = form
    context["step_title"] = "Start Attendance Session"
    context["step_description"] = (
        "You're all set! Start your first attendance session to complete the setup."
    )

    return render(request, "users/setup_wizard/step5_start_session.html", context)


@login_required
def setup_wizard_skip(request):
    """
    Allow users to skip the wizard and mark it as complete.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    progress = _get_or_create_wizard_progress(request.user)
    progress.completed = True
    progress.save()
    messages.info(request, "Setup wizard skipped. You can configure settings manually.")
    return redirect("dashboard")


@login_required
@require_http_methods(["GET"])
def setup_wizard_status(request):
    """
    API endpoint to check wizard status.

    Returns JSON with current step and completion status.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return JsonResponse({"error": "Not authorized"}, status=403)

    progress = _get_or_create_wizard_progress(request.user)

    return JsonResponse(
        {
            "current_step": progress.current_step,
            "completed": progress.completed,
            "org_name": progress.org_name,
            "camera_tested": progress.camera_tested,
            "liveness_tested": progress.liveness_tested,
            "first_employee_username": progress.first_employee_username,
            "first_employee_photos_captured": progress.first_employee_photos_captured,
            "model_trained": progress.model_trained,
            "first_session_started": progress.first_session_started,
        }
    )
