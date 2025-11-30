"""Tests for the setup wizard functionality."""

from django.contrib.auth import get_user_model
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
        user = User.objects.create_user(username="admin", password="password", is_staff=True)
        progress = SetupWizardProgress.objects.create(user=user)

        assert progress.current_step == SetupWizardProgress.Step.ORG_DETAILS
        assert not progress.completed
        assert progress.org_name == ""
        assert progress.org_timezone == ""

    def test_can_proceed_to_step(self):
        """Test step progression logic."""
        user = User.objects.create_user(username="admin2", password="password", is_staff=True)
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
        user = User.objects.create_user(username="admin3", password="password", is_staff=True)
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
        return User.objects.create_user(
            username="testadmin",
            password="testpass123",
            is_staff=True,
        )

    @pytest.fixture
    def regular_user(self):
        """Create and return a regular user."""
        return User.objects.create_user(
            username="regularuser",
            password="testpass123",
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
