"""
Tests for the users app.

This module contains test cases for the user registration view, ensuring that
access is correctly restricted based on user roles (staff, superuser, regular user)
and that the registration functionality works as expected for authorized users.
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password
from django.test import TestCase
from django.urls import reverse

from .models import Direction, Time


class CustomLoginViewTests(TestCase):
    """Test suite for the custom login view and rate limiting."""

    def setUp(self):
        self.login_url = reverse("login")
        User = get_user_model()
        hashed_password = make_password("Testpass123")
        self.user = User.objects.create(username="testuser_login", password=hashed_password)

    def test_login_success(self):
        """Verify successful login."""
        response = self.client.post(
            self.login_url, {"username": "testuser_login", "password": "Testpass123"}
        )
        self.assertRedirects(response, reverse("dashboard"))

    def test_login_rate_limit(self):
        """Verify that login is rate-limited after multiple attempts."""
        # 5 attempts allowed by rate limiter
        for _ in range(5):
            self.client.post(
                self.login_url, {"username": "ratelimit_user", "password": "wrongpassword"}
            )

        # 6th attempt should be rate limited
        response = self.client.post(
            self.login_url, {"username": "ratelimit_user", "password": "wrongpassword"}
        )
        self.assertEqual(response.status_code, 429)
        self.assertContains(response, "Too many login attempts", status_code=429)

        # Reset rate limit for next test by clearing the cache
        from django.core.cache import cache

        cache.clear()


class RegisterViewTests(TestCase):
    """Test suite for the user registration view."""

    def setUp(self):
        """Set up URLs and create users with different permission levels."""
        self.register_url = reverse("register")
        self.not_authorised_url = reverse("not-authorised")
        self.password = "Testpass123"
        User = get_user_model()
        hashed_password = make_password(self.password)

        # Create a staff user (e.g., a manager) who should have access
        self.staff_user = User.objects.create(
            username="staff_user", password=hashed_password, is_staff=True
        )
        # Create a superuser who should have access
        self.superuser = User.objects.create(
            username="super_user", email="super@example.com", password=hashed_password, is_superuser=True, is_staff=True
        )
        # Create a regular user who should NOT have access
        self.regular_user = User.objects.create(
            username="regular_user", password=hashed_password
        )

    def test_staff_user_can_access_register_page(self):
        """Verify that a logged-in staff user can access the registration page."""
        self.client.force_login(self.staff_user)
        response = self.client.get(self.register_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "users/register.html")

    def test_superuser_can_access_register_page(self):
        """Verify that a logged-in superuser can access the registration page."""
        self.client.force_login(self.superuser)
        response = self.client.get(self.register_url)
        self.assertEqual(response.status_code, 200)

    def test_non_staff_user_redirected_on_get(self):
        """Ensure a regular user is redirected when trying to GET the registration page."""
        self.client.force_login(self.regular_user)
        response = self.client.get(self.register_url)
        self.assertRedirects(response, self.not_authorised_url)

    def test_staff_user_can_register_new_user(self):
        """Test that a staff user can successfully register a new user via POST."""
        self.client.force_login(self.staff_user)
        new_user_data = {
            "username": "new_employee",
            "password1": "Newpass12345",
            "password2": "Newpass12345",
        }
        response = self.client.post(self.register_url, new_user_data)

        # Should redirect to the dashboard on success
        self.assertRedirects(response, reverse("dashboard"))
        # Verify the new user was created in the database
        self.assertTrue(get_user_model().objects.filter(username="new_employee").exists())

    def test_non_staff_user_cannot_register_via_post(self):
        """Ensure a regular user is redirected and cannot register a new user via POST."""
        self.client.force_login(self.regular_user)
        response = self.client.post(
            self.register_url,
            {
                "username": "should_not_create",
                "password1": "Password12345",
                "password2": "Password12345",
            },
        )

        # The user should be redirected, and no new user should be created
        self.assertRedirects(response, self.not_authorised_url)
        self.assertFalse(get_user_model().objects.filter(username="should_not_create").exists())


class SetupWizardTests(TestCase):
    """Test suite for the main setup wizard view."""

    def setUp(self):
        self.wizard_url = reverse("setup-wizard")
        self.not_authorised_url = reverse("not-authorised")
        User = get_user_model()
        hashed_password = make_password("Testpass123")
        self.staff_user = User.objects.create(
            username="wizard_staff", password=hashed_password, is_staff=True
        )
        self.regular_user = User.objects.create(
            username="wizard_regular", password=hashed_password
        )

    def test_non_staff_redirected(self):
        """Ensure non-staff users cannot access the wizard."""
        self.client.force_login(self.regular_user)
        response = self.client.get(self.wizard_url)
        self.assertRedirects(response, self.not_authorised_url)

    def test_completed_wizard_redirects_to_dashboard(self):
        """Ensure completed wizard redirects to dashboard."""
        from .models import SetupWizardProgress

        progress, _ = SetupWizardProgress.objects.get_or_create(user=self.staff_user)
        progress.completed = True
        progress.save()

        self.client.force_login(self.staff_user)
        response = self.client.get(self.wizard_url)
        self.assertRedirects(response, reverse("dashboard"))

    def test_wizard_redirects_to_current_step(self):
        """Ensure wizard redirects to the current step."""
        from .models import SetupWizardProgress

        progress, _ = SetupWizardProgress.objects.get_or_create(user=self.staff_user)
        progress.current_step = SetupWizardProgress.Step.CAMERA_TEST
        progress.save()

        self.client.force_login(self.staff_user)
        response = self.client.get(self.wizard_url)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("setup-wizard-step2"))


class SetupWizardStep1Tests(TestCase):
    """Test suite for setup wizard step 1."""

    def setUp(self):
        self.step1_url = reverse("setup-wizard-step1")
        User = get_user_model()
        hashed_password = make_password("Testpass123")
        self.staff_user = User.objects.create(
            username="wizard_staff", password=hashed_password, is_staff=True
        )

    def test_get_step1(self):
        """Ensure staff can access step 1."""
        self.client.force_login(self.staff_user)
        response = self.client.get(self.step1_url)
        self.assertEqual(response.status_code, 200)

    def test_post_step1_valid(self):
        """Ensure valid post saves progress and redirects."""
        self.client.force_login(self.staff_user)
        response = self.client.post(self.step1_url, {"org_name": "Test Org", "org_timezone": "UTC"})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("setup-wizard-step2"))

        from .models import SetupWizardProgress

        progress = SetupWizardProgress.objects.get(user=self.staff_user)
        self.assertEqual(progress.org_name, "Test Org")
        self.assertEqual(progress.current_step, SetupWizardProgress.Step.CAMERA_TEST)


class SetupWizardStep2Tests(TestCase):
    """Test suite for setup wizard step 2."""

    def setUp(self):
        self.step2_url = reverse("setup-wizard-step2")
        User = get_user_model()
        self.staff_user = User.objects.create_user(
            username="wizard_staff", password="Testpass123", is_staff=True
        )

    def test_get_step2_without_step1_redirects(self):
        """Ensure accessing step 2 without step 1 redirects."""
        self.client.force_login(self.staff_user)
        response = self.client.get(self.step2_url)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("setup-wizard-step1"))

    def test_get_step2_valid(self):
        """Ensure accessing step 2 works when step 1 is done."""
        from .models import SetupWizardProgress

        progress, _ = SetupWizardProgress.objects.get_or_create(user=self.staff_user)
        progress.org_name = "Test Org"
        progress.org_timezone = "UTC"
        progress.current_step = SetupWizardProgress.Step.CAMERA_TEST
        progress.save()

        self.client.force_login(self.staff_user)
        response = self.client.get(self.step2_url)
        self.assertEqual(response.status_code, 200)

    def test_post_step2_valid(self):
        """Ensure valid post saves progress and redirects."""
        from .models import SetupWizardProgress

        progress, _ = SetupWizardProgress.objects.get_or_create(user=self.staff_user)
        progress.org_name = "Test Org"
        progress.org_timezone = "UTC"
        progress.current_step = SetupWizardProgress.Step.CAMERA_TEST
        progress.save()

        self.client.force_login(self.staff_user)
        response = self.client.post(
            self.step2_url, {"camera_tested": True, "liveness_tested": True}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("setup-wizard-step3"))

        progress.refresh_from_db()
        self.assertTrue(progress.camera_tested)
        self.assertTrue(progress.liveness_tested)
        self.assertEqual(progress.current_step, SetupWizardProgress.Step.ADD_EMPLOYEE)


class SetupWizardStep3Tests(TestCase):
    """Test suite for setup wizard step 3."""

    def setUp(self):
        self.step3_url = reverse("setup-wizard-step3")
        User = get_user_model()
        hashed_password = make_password("Testpass123")
        self.staff_user = User.objects.create(
            username="wizard_staff", password=hashed_password, is_staff=True
        )

    def test_get_step3_without_step2_redirects(self):
        """Ensure accessing step 3 without step 2 redirects."""
        self.client.force_login(self.staff_user)
        response = self.client.get(self.step3_url)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("setup-wizard-step2"))

    def test_post_step3_create_employee(self):
        """Ensure creating employee in step 3 works."""
        from .models import SetupWizardProgress

        progress, _ = SetupWizardProgress.objects.get_or_create(user=self.staff_user)
        progress.org_name = "Test Org"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.current_step = SetupWizardProgress.Step.ADD_EMPLOYEE
        progress.save()

        self.client.force_login(self.staff_user)
        response = self.client.post(
            self.step3_url,
            {
                "create_employee": "true",
                "username": "wizard_new_user",
                "password": "NewUserPass123",
                # The form expects password1 and password2
                "password1": "NewUserPass123",
                "password2": "NewUserPass123",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, self.step3_url)

        progress.refresh_from_db()
        self.assertEqual(progress.first_employee_username, "wizard_new_user")

    def test_post_step3_confirm_photos(self):
        """Ensure confirming photos works and redirects to step 4."""
        from .models import SetupWizardProgress

        progress, _ = SetupWizardProgress.objects.get_or_create(user=self.staff_user)
        progress.org_name = "Test Org"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "wizard_new_user"
        progress.current_step = SetupWizardProgress.Step.ADD_EMPLOYEE
        progress.save()

        self.client.force_login(self.staff_user)
        response = self.client.post(
            self.step3_url, {"confirm_photos": "true", "photos_captured": True}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("setup-wizard-step4"))

        progress.refresh_from_db()
        self.assertTrue(progress.first_employee_photos_captured)
        self.assertEqual(progress.current_step, SetupWizardProgress.Step.TRAIN_MODEL)


class SetupWizardStep4Tests(TestCase):
    """Test suite for setup wizard step 4."""

    def setUp(self):
        self.step4_url = reverse("setup-wizard-step4")
        User = get_user_model()
        hashed_password = make_password("Testpass123")
        self.staff_user = User.objects.create(
            username="wizard_staff", password=hashed_password, is_staff=True
        )

    def test_get_step4_without_step3_redirects(self):
        """Ensure accessing step 4 without step 3 redirects."""
        self.client.force_login(self.staff_user)
        response = self.client.get(self.step4_url)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("setup-wizard-step3"))

    def test_post_step4_continue(self):
        """Ensure continue from step 4 works."""
        from .models import SetupWizardProgress

        progress, _ = SetupWizardProgress.objects.get_or_create(user=self.staff_user)
        progress.org_name = "Test Org"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "wizard_new_user"
        progress.first_employee_photos_captured = True
        progress.model_trained = True
        progress.current_step = SetupWizardProgress.Step.TRAIN_MODEL
        progress.save()

        self.client.force_login(self.staff_user)
        response = self.client.post(
            self.step4_url,
            {
                "continue": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("setup-wizard-step5"))

        progress.refresh_from_db()
        self.assertEqual(progress.current_step, SetupWizardProgress.Step.START_SESSION)


class SetupWizardStep5Tests(TestCase):
    """Test suite for setup wizard step 5."""

    def setUp(self):
        self.step5_url = reverse("setup-wizard-step5")
        User = get_user_model()
        hashed_password = make_password("Testpass123")
        self.staff_user = User.objects.create(
            username="wizard_staff", password=hashed_password, is_staff=True
        )

    def test_get_step5_without_step4_redirects(self):
        """Ensure accessing step 5 without step 4 redirects."""
        self.client.force_login(self.staff_user)
        response = self.client.get(self.step5_url)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("setup-wizard-step4"))

    def test_post_step5_start_checkin_session(self):
        """Ensure finishing step 5 check_in works."""
        from .models import SetupWizardProgress

        progress, _ = SetupWizardProgress.objects.get_or_create(user=self.staff_user)
        progress.org_name = "Test Org"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "wizard_new_user"
        progress.first_employee_photos_captured = True
        progress.model_trained = True
        progress.current_step = SetupWizardProgress.Step.START_SESSION
        progress.save()

        self.client.force_login(self.staff_user)
        response = self.client.post(
            self.step5_url,
            {
                "session_type": "check_in",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("mark-your-attendance"))

        progress.refresh_from_db()
        self.assertTrue(progress.first_session_started)
        self.assertTrue(progress.completed)

    def test_post_step5_start_checkout_session(self):
        """Ensure finishing step 5 check_out works."""
        from .models import SetupWizardProgress

        progress, _ = SetupWizardProgress.objects.get_or_create(user=self.staff_user)
        progress.org_name = "Test Org"
        progress.org_timezone = "UTC"
        progress.camera_tested = True
        progress.liveness_tested = True
        progress.first_employee_username = "wizard_new_user"
        progress.first_employee_photos_captured = True
        progress.model_trained = True
        progress.current_step = SetupWizardProgress.Step.START_SESSION
        progress.save()

        self.client.force_login(self.staff_user)
        response = self.client.post(
            self.step5_url,
            {
                "session_type": "check_out",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("mark-your-attendance-out"))

        progress.refresh_from_db()
        self.assertTrue(progress.first_session_started)
        self.assertTrue(progress.completed)


class SetupWizardSkipAndStatusTests(TestCase):
    """Test suite for setup wizard skip and status views."""

    def setUp(self):
        self.skip_url = reverse("setup-wizard-skip")
        self.status_url = reverse("setup-wizard-status")
        User = get_user_model()
        hashed_password = make_password("Testpass123")
        self.staff_user = User.objects.create(
            username="wizard_staff", password=hashed_password, is_staff=True
        )
        self.regular_user = User.objects.create(
            username="wizard_regular", password=hashed_password
        )

    def test_skip_wizard(self):
        """Ensure skipping wizard marks it complete and redirects."""
        self.client.force_login(self.staff_user)
        response = self.client.get(self.skip_url)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("dashboard"))

        from .models import SetupWizardProgress

        progress = SetupWizardProgress.objects.get(user=self.staff_user)
        self.assertTrue(progress.completed)

    def test_wizard_status_staff(self):
        """Ensure status returns JSON for staff."""
        self.client.force_login(self.staff_user)
        response = self.client.get(self.status_url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["completed"], False)

    def test_wizard_status_non_staff(self):
        """Ensure status returns 403 JSON for non-staff."""
        self.client.force_login(self.regular_user)
        response = self.client.get(self.status_url)

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["status"], 403)


class TimeModelTests(TestCase):
    """Test suite for the Time model."""

    def test_time_str_handles_missing_timestamp(self):
        """The string representation should handle a missing timestamp gracefully."""

        hashed_password = make_password("Testpass123")
        user = get_user_model().objects.create(username="time_user", password=hashed_password)
        time_entry = Time(user=user, time=None, direction=Direction.IN)

        self.assertEqual(
            str(time_entry),
            "time_user - No timestamp recorded - Check-in",
        )
