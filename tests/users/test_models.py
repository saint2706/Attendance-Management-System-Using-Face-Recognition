from django.contrib.auth.models import User

import pytest

from users.models import Direction, RecognitionAttempt, SetupWizardProgress


@pytest.mark.django_db
def test_setup_wizard_progress_str():
    user = User.objects.create_user(username="testadmin")
    progress = SetupWizardProgress.objects.create(
        user=user, current_step=SetupWizardProgress.Step.ORG_DETAILS, completed=False
    )
    assert str(progress) == "testadmin - step 1 (Organization Details)"

    progress.completed = True
    assert str(progress) == "testadmin - completed (Organization Details)"


@pytest.mark.django_db
def test_setup_wizard_can_proceed_to_step():
    user = User.objects.create_user(username="testadmin")
    progress = SetupWizardProgress.objects.create(user=user)

    assert progress.can_proceed_to_step(SetupWizardProgress.Step.ORG_DETAILS) is True

    assert progress.can_proceed_to_step(SetupWizardProgress.Step.CAMERA_TEST) is False
    progress.org_name = "Test Org"
    progress.org_timezone = "UTC"
    assert progress.can_proceed_to_step(SetupWizardProgress.Step.CAMERA_TEST) is True

    assert progress.can_proceed_to_step(SetupWizardProgress.Step.ADD_EMPLOYEE) is False
    progress.camera_tested = True
    progress.liveness_tested = True
    assert progress.can_proceed_to_step(SetupWizardProgress.Step.ADD_EMPLOYEE) is True

    assert progress.can_proceed_to_step(SetupWizardProgress.Step.TRAIN_MODEL) is False
    progress.first_employee_username = "testemp"
    progress.first_employee_photos_captured = True
    assert progress.can_proceed_to_step(SetupWizardProgress.Step.TRAIN_MODEL) is True

    assert progress.can_proceed_to_step(SetupWizardProgress.Step.START_SESSION) is False
    progress.model_trained = True
    assert progress.can_proceed_to_step(SetupWizardProgress.Step.START_SESSION) is True

    assert progress.can_proceed_to_step(999) is False


@pytest.mark.django_db
def test_setup_wizard_get_step_status():
    user = User.objects.create_user(username="testadmin")
    progress = SetupWizardProgress.objects.create(
        user=user, current_step=SetupWizardProgress.Step.CAMERA_TEST
    )

    assert progress.get_step_status(SetupWizardProgress.Step.ORG_DETAILS) == "completed"
    assert progress.get_step_status(SetupWizardProgress.Step.CAMERA_TEST) == "current"

    assert progress.get_step_status(SetupWizardProgress.Step.ADD_EMPLOYEE) == "locked"
    progress.camera_tested = True
    progress.liveness_tested = True
    assert progress.get_step_status(SetupWizardProgress.Step.ADD_EMPLOYEE) == "available"


@pytest.mark.django_db
def test_recognition_attempt_str():
    user = User.objects.create_user(username="testemp")
    attempt1 = RecognitionAttempt.objects.create(
        user=user, username="testemp", direction=Direction.IN, successful=True
    )
    assert str(attempt1) == "testemp - Check-in - success"

    attempt2 = RecognitionAttempt.objects.create(
        user=None, username="", direction=Direction.OUT, successful=False
    )
    assert str(attempt2) == "unknown - Check-out - failure"

    attempt3 = RecognitionAttempt.objects.create(
        user=user, username="", direction=Direction.IN, successful=True
    )
    assert str(attempt3) == "testemp - Check-in - success"

    attempt4 = RecognitionAttempt.objects.create(
        user=None, username="explicitname", direction=Direction.OUT, successful=False
    )
    assert str(attempt4) == "explicitname - Check-out - failure"
