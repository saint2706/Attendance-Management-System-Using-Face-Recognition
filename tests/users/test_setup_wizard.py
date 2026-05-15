"""Tests for the setup wizard functionality."""

from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password
from django.test import Client
from django.urls import reverse

import pytest

from users.models import SetupWizardProgress

User = get_user_model()


@pytest.mark.django_db
class TestSetupWizardProgress:
    """Tests for the SetupWizardProgress model."""

    def test_create_wizard_progress(self):
        """Test creating a wizard progress instance."""
        user = User.objects.create(
            username="admin", password=make_password("password"), is_staff=True
        )
        progress = SetupWizardProgress.objects.create(user=user)

        assert progress.current_step == SetupWizardProgress.Step.ORG_DETAILS
        assert not progress.completed
        assert progress.org_name == ""
        assert progress.org_timezone == ""

    def test_can_proceed_to_step(self):
        """Test step progression logic."""
        user = User.objects.create(
            username="admin2", password=make_password("password"), is_staff=True
        )
        progress = SetupWizardProgress.objects.create(user=user)

        # Step 1 is always accessible
        assert progress.can_proceed_to_step(SetupWizardProgress.Step.ORG_DETAILS)

        # Step 2 requires org details
        assert not progress.can_proceed_to_step(SetupWizardProgress.Step.CAMERA_TEST)
        progress.org_name = "Test Org"
        progress.org_timezone = "UTC"
        progress.save()
        assert progress.can_proceed_to_step(SetupWizardProgress.Step.CAMERA_TEST)

        # Step 3 requires camera/liveness tested
        assert not progress.can_proceed_to_step(SetupWizardProgress.Step.ADD_EMPLOYEE)
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.save()
        assert progress.can_proceed_to_step(SetupWizardProgress.Step.ADD_EMPLOYEE)

        # Step 4 requires employee and photos
        assert not progress.can_proceed_to_step(SetupWizardProgress.Step.TRAIN_MODEL)
        progress.first_employee_username = "employee1"
        progress.first_employee_photos_captured = True
        progress.save()
        assert progress.can_proceed_to_step(SetupWizardProgress.Step.TRAIN_MODEL)

        # Step 5 requires model trained
        assert not progress.can_proceed_to_step(SetupWizardProgress.Step.START_SESSION)
        progress.model_trained = True
        progress.save()
        assert progress.can_proceed_to_step(SetupWizardProgress.Step.START_SESSION)

    def test_get_step_status(self):
        """Test getting step status."""
        user = User.objects.create(
            username="admin3", password=make_password("password"), is_staff=True
        )
        progress = SetupWizardProgress.objects.create(
            user=user,
            current_step=SetupWizardProgress.Step.CAMERA_TEST,
            org_name="Test",
            org_timezone="UTC",
        )

        assert progress.get_step_status(SetupWizardProgress.Step.ORG_DETAILS) == "completed"
        assert progress.get_step_status(SetupWizardProgress.Step.CAMERA_TEST) == "current"
        assert progress.get_step_status(SetupWizardProgress.Step.ADD_EMPLOYEE) == "locked"


@pytest.mark.django_db
class TestSetupWizardViews:
    """Tests for the setup wizard views."""

    @pytest.fixture
    def admin_user(self):
        """Create and return an admin user."""
        return User.objects.create(
            username="testadmin",
            password=make_password("testpass123"),
            is_staff=True,
        )

    @pytest.fixture
    def regular_user(self):
        """Create and return a regular user."""
        return User.objects.create(
            username="regularuser",
            password=make_password("testpass123"),
            is_staff=False,
        )

    @pytest.fixture
    def authenticated_admin_client(self, admin_user):
        """Return a client logged in as admin."""
        client = Client()
        client.login(username="testadmin", password="testpass123")
        return client

    @pytest.fixture
    def authenticated_regular_client(self, regular_user):
        """Return a client logged in as regular user."""
        client = Client()
        client.login(username="regularuser", password="testpass123")
        return client

    def test_wizard_requires_login(self, client):
        """Test that wizard requires authentication."""
        response = client.get(reverse("setup-wizard"))
        assert response.status_code == 302
        assert "/login/" in response.url

    def test_wizard_requires_staff(self, authenticated_regular_client):
        """Test that wizard requires staff status."""
        response = authenticated_regular_client.get(reverse("setup-wizard"))
        assert response.status_code == 302
        assert "not_authorised" in response.url or "not-authorised" in response.url

    def test_wizard_redirects_to_current_step(self, authenticated_admin_client, admin_user):
        """Test that wizard redirects to current step."""
        response = authenticated_admin_client.get(reverse("setup-wizard"))
        assert response.status_code == 302
        assert "/setup-wizard/step1/" in response.url

    def test_step1_renders(self, authenticated_admin_client):
        """Test that step 1 renders correctly."""
        response = authenticated_admin_client.get(reverse("setup-wizard-step1"))
        assert response.status_code == 200
        assert b"Organization Details" in response.content

    def test_step1_submission(self, authenticated_admin_client, admin_user):
        """Test step 1 form submission."""
        response = authenticated_admin_client.post(
            reverse("setup-wizard-step1"),
            {"org_name": "Test Corp", "org_timezone": "UTC"},
        )
        assert response.status_code == 302
        assert "/setup-wizard/step2/" in response.url

        # Verify progress was saved
        progress = SetupWizardProgress.objects.get(user=admin_user)
        assert progress.org_name == "Test Corp"
        assert progress.org_timezone == "UTC"
        assert progress.current_step == SetupWizardProgress.Step.CAMERA_TEST

    def test_step2_requires_step1_completion(self, authenticated_admin_client, admin_user):
        """Test that step 2 requires step 1 completion."""
        response = authenticated_admin_client.get(reverse("setup-wizard-step2"))
        # Should redirect back to step 1
        assert response.status_code == 302
        assert "/setup-wizard/step1/" in response.url

    def test_step2_renders_after_step1(self, authenticated_admin_client, admin_user):
        """Test step 2 renders after step 1 completion."""
        # Complete step 1
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.current_step = SetupWizardProgress.Step.CAMERA_TEST
        progress.save()

        response = authenticated_admin_client.get(reverse("setup-wizard-step2"))
        assert response.status_code == 200
        assert b"Camera" in response.content

    def test_wizard_skip(self, authenticated_admin_client, admin_user):
        """Test skipping the wizard."""
        response = authenticated_admin_client.get(reverse("setup-wizard-skip"))
        assert response.status_code == 302
        assert "/dashboard/" in response.url

        progress = SetupWizardProgress.objects.get(user=admin_user)
        assert progress.completed

    def test_wizard_status_api(self, authenticated_admin_client, admin_user):
        """Test wizard status API endpoint."""
        # Create some progress
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test Corp"
        progress.save()

        response = authenticated_admin_client.get(reverse("setup-wizard-status"))
        assert response.status_code == 200
        data = response.json()
        assert data["org_name"] == "Test Corp"
        assert data["current_step"] == SetupWizardProgress.Step.ORG_DETAILS
        assert not data["completed"]

    def test_completed_wizard_redirects_to_dashboard(self, authenticated_admin_client, admin_user):
        """Test that completed wizard redirects to dashboard."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.completed = True
        progress.save()

        response = authenticated_admin_client.get(reverse("setup-wizard"))
        assert response.status_code == 302
        assert "/dashboard/" in response.url

    def test_step3_requires_step2_completion(self, authenticated_admin_client, admin_user):
        """Test that step 3 requires step 2 completion."""
        response = authenticated_admin_client.get(reverse("setup-wizard-step3"))
        assert response.status_code == 302
        assert "/setup-wizard-step2/" in response.url or "/setup-wizard/step2/" in response.url

    def test_step3_renders_after_step2(self, authenticated_admin_client, admin_user):
        """Test step 3 renders after step 2 completion."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.current_step = SetupWizardProgress.Step.ADD_EMPLOYEE
        progress.save()

        response = authenticated_admin_client.get(reverse("setup-wizard-step3"))
        assert response.status_code == 200
        assert b"Add First Employee" in response.content

    def test_step3_create_employee_submission(self, authenticated_admin_client, admin_user):
        """Test step 3 employee creation submission."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.current_step = SetupWizardProgress.Step.ADD_EMPLOYEE
        progress.save()

        response = authenticated_admin_client.post(
            reverse("setup-wizard-step3"),
            {
                "create_employee": "1",
                "username": "newemployee",
                "password1": "complexPass123!",
                "password2": "complexPass123!",
            },
        )
        assert response.status_code == 302
        assert "/setup-wizard/step3/" in response.url

        progress.refresh_from_db()
        assert progress.first_employee_username == "newemployee"

    def test_step3_confirm_photos_submission(self, authenticated_admin_client, admin_user):
        """Test step 3 confirm photos submission."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.current_step = SetupWizardProgress.Step.ADD_EMPLOYEE
        progress.save()

        response = authenticated_admin_client.post(
            reverse("setup-wizard-step3"),
            {"confirm_photos": "1", "photos_captured": "True"},
        )
        assert response.status_code == 302
        assert "/setup-wizard/step4/" in response.url

        progress.refresh_from_db()
        assert progress.first_employee_photos_captured
        assert progress.current_step == SetupWizardProgress.Step.TRAIN_MODEL

    def test_step4_requires_step3_completion(self, authenticated_admin_client, admin_user):
        """Test that step 4 requires step 3 completion."""
        response = authenticated_admin_client.get(reverse("setup-wizard-step4"))
        assert response.status_code == 302
        assert "/setup-wizard/step3/" in response.url

    def test_step4_renders_after_step3(self, authenticated_admin_client, admin_user):
        """Test step 4 renders after step 3 completion."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.current_step = SetupWizardProgress.Step.TRAIN_MODEL
        progress.save()

        response = authenticated_admin_client.get(reverse("setup-wizard-step4"))
        assert response.status_code == 200
        assert b"Train Recognition Model" in response.content

    def test_step4_renders_with_task_status(
        self, authenticated_admin_client, admin_user, monkeypatch
    ):
        """Test step 4 renders correctly when a task is in progress."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.training_task_id = "mock-task-id"
        progress.current_step = SetupWizardProgress.Step.TRAIN_MODEL
        progress.save()

        class MockAsyncResult:
            def __init__(self, task_id):
                self.id = task_id
                self.status = "SUCCESS"

            def ready(self):
                return True

            def successful(self):
                return True

        monkeypatch.setattr("celery.result.AsyncResult", MockAsyncResult)

        response = authenticated_admin_client.get(reverse("setup-wizard-step4"))
        assert response.status_code == 200

        progress.refresh_from_db()
        assert progress.model_trained

    def test_step4_start_training_import_error(
        self, authenticated_admin_client, admin_user, monkeypatch
    ):
        """Test step 4 start training handles ImportError for celery tasks."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.current_step = SetupWizardProgress.Step.TRAIN_MODEL
        progress.save()

        def mock_import_error(*args, **kwargs):
            raise ImportError("Mocked import error")

        # By doing this hack, it raises inside the try-except
        monkeypatch.setattr("recognition.tasks.train_recognition_model.delay", mock_import_error)

        response = authenticated_admin_client.post(
            reverse("setup-wizard-step4"),
            {"start_training": "1"},
        )
        assert response.status_code == 200
        assert b"Training service not available. Please check configuration." in response.content

    def test_step4_start_training_general_error(
        self, authenticated_admin_client, admin_user, monkeypatch
    ):
        """Test step 4 start training handles Exception for celery tasks."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.current_step = SetupWizardProgress.Step.TRAIN_MODEL
        progress.save()

        def mock_error(*args, **kwargs):
            raise Exception("Mocked general error")

        monkeypatch.setattr("recognition.tasks.train_recognition_model.delay", mock_error)

        response = authenticated_admin_client.post(
            reverse("setup-wizard-step4"),
            {"start_training": "1"},
        )
        assert response.status_code == 200
        assert b"Failed to start training. Please try again." in response.content

    def test_step5_rate_limited(self, authenticated_admin_client, admin_user, monkeypatch):
        """Test step 5 handles rate limits properly."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.model_trained = True
        progress.current_step = SetupWizardProgress.Step.START_SESSION
        progress.save()

        # Mock getattr on request inside setup_wizard_step5
        def mock_limited(request, *args, **kwargs):
            request.limited = True
            return True

        monkeypatch.setattr("django_ratelimit.decorators.is_ratelimited", mock_limited)

        response = authenticated_admin_client.get(reverse("setup-wizard-step5"))
        assert response.status_code == 429
        assert b"Too many attempts. Please try again later." in response.content

    def test_step4_rate_limited(self, authenticated_admin_client, admin_user, monkeypatch):
        """Test step 4 handles rate limits properly."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.current_step = SetupWizardProgress.Step.TRAIN_MODEL
        progress.save()

        # Mock ratelimit decorator
        def mock_limited(request, *args, **kwargs):
            request.limited = True
            return True

        monkeypatch.setattr("django_ratelimit.decorators.is_ratelimited", mock_limited)

        response = authenticated_admin_client.get(reverse("setup-wizard-step4"))
        assert response.status_code == 200
        assert b"Too many attempts. Please try again later." in response.content

    def test_step4_task_does_not_exist(self, authenticated_admin_client, admin_user, monkeypatch):
        """Test step 4 task handling when Celery throws an Exception on AsyncResult."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.training_task_id = "mock-task-id"
        progress.current_step = SetupWizardProgress.Step.TRAIN_MODEL
        progress.save()

        def mock_async_result(task_id):
            raise Exception("Celery task exception")

        monkeypatch.setattr("celery.result.AsyncResult", mock_async_result)

        response = authenticated_admin_client.get(reverse("setup-wizard-step4"))
        assert response.status_code == 200

    def test_step4_start_training_submission(
        self, authenticated_admin_client, admin_user, monkeypatch
    ):
        """Test step 4 start training submission."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.current_step = SetupWizardProgress.Step.TRAIN_MODEL
        progress.save()

        class MockResult:
            id = "mock-task-id"

        class MockDelay:
            def __init__(self):
                self.called = False
                self.called_with = None

            def delay(self, **kwargs):
                self.called = True
                self.called_with = kwargs
                return MockResult()

        mock_task = MockDelay()
        monkeypatch.setattr("recognition.tasks.train_recognition_model.delay", mock_task.delay)

        response = authenticated_admin_client.post(
            reverse("setup-wizard-step4"),
            {"start_training": "1"},
        )
        assert response.status_code == 302
        assert "/setup-wizard/step4/" in response.url

        progress.refresh_from_db()
        assert progress.training_task_id == "mock-task-id"
        assert mock_task.called
        assert mock_task.called_with == {"initiated_by": admin_user.username}

    def test_step4_continue_submission(self, authenticated_admin_client, admin_user):
        """Test step 4 continue submission."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.training_task_id = "mock-task-id"
        progress.model_trained = True
        progress.current_step = SetupWizardProgress.Step.TRAIN_MODEL
        progress.save()

        response = authenticated_admin_client.post(
            reverse("setup-wizard-step4"),
            {"continue": "1"},
        )
        assert response.status_code == 302
        assert "/setup-wizard/step5/" in response.url

        progress.refresh_from_db()
        assert progress.current_step == SetupWizardProgress.Step.START_SESSION

    def test_step5_requires_step4_completion(self, authenticated_admin_client, admin_user):
        """Test that step 5 requires step 4 completion."""
        response = authenticated_admin_client.get(reverse("setup-wizard-step5"))
        assert response.status_code == 302
        assert "/setup-wizard/step4/" in response.url

    def test_step5_renders_after_step4(self, authenticated_admin_client, admin_user):
        """Test step 5 renders after step 4 completion."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.model_trained = True
        progress.current_step = SetupWizardProgress.Step.START_SESSION
        progress.save()

        response = authenticated_admin_client.get(reverse("setup-wizard-step5"))
        assert response.status_code == 200
        assert b"Start Attendance Session" in response.content

    def test_step5_start_session_check_in(self, authenticated_admin_client, admin_user):
        """Test step 5 start session submission for check-in."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.model_trained = True
        progress.current_step = SetupWizardProgress.Step.START_SESSION
        progress.save()

        response = authenticated_admin_client.post(
            reverse("setup-wizard-step5"),
            {"session_type": "check_in"},
        )
        assert response.status_code == 302
        assert reverse("mark-your-attendance") in response.url

        progress.refresh_from_db()
        assert progress.first_session_started
        assert progress.completed

    def test_step5_start_session_check_out(self, authenticated_admin_client, admin_user):
        """Test step 5 start session submission for check-out."""
        progress, _ = SetupWizardProgress.objects.get_or_create(user=admin_user)
        progress.org_name = "Test"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "newemployee"
        progress.first_employee_photos_captured = True
        progress.model_trained = True
        progress.current_step = SetupWizardProgress.Step.START_SESSION
        progress.save()

        response = authenticated_admin_client.post(
            reverse("setup-wizard-step5"),
            {"session_type": "check_out"},
        )
        assert response.status_code == 302
        assert reverse("mark-your-attendance-out") in response.url

        progress.refresh_from_db()
        assert progress.first_session_started
        assert progress.completed


@pytest.mark.django_db
class TestSetupWizardForms:
    """Tests for wizard forms."""

    def test_org_details_form_valid(self):
        """Test OrgDetailsForm with valid data."""
        from users.forms import OrgDetailsForm

        form = OrgDetailsForm(data={"org_name": "Test Corp", "org_timezone": "UTC"})
        assert form.is_valid()

    def test_org_details_form_invalid(self):
        """Test OrgDetailsForm with invalid data."""
        from users.forms import OrgDetailsForm

        form = OrgDetailsForm(data={"org_name": "", "org_timezone": "UTC"})
        assert not form.is_valid()
        assert "org_name" in form.errors

    def test_camera_test_form_valid(self):
        """Test CameraTestForm with valid data."""
        from users.forms import CameraTestForm

        form = CameraTestForm(data={"camera_tested": True, "liveness_tested": True})
        assert form.is_valid()

    def test_camera_test_form_requires_both(self):
        """Test CameraTestForm requires both checkboxes."""
        from users.forms import CameraTestForm

        form = CameraTestForm(data={"camera_tested": True, "liveness_tested": False})
        assert not form.is_valid()

    def test_add_employee_form_valid(self):
        """Test AddEmployeeForm with valid data."""
        from users.forms import AddEmployeeForm

        form = AddEmployeeForm(
            data={
                "username": "newemployee",
                "password1": "complexPass123!",
                "password2": "complexPass123!",
            }
        )
        assert form.is_valid(), form.errors

    def test_start_session_form_valid(self):
        """Test StartSessionForm with valid data."""
        from users.forms import StartSessionForm

        form = StartSessionForm(data={"session_type": "check_in"})
        assert form.is_valid()

        form = StartSessionForm(data={"session_type": "check_out"})
        assert form.is_valid()
