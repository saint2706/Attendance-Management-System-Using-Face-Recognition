"""
Tests for the users app.

This module contains test cases for the user registration view, ensuring that
access is correctly restricted based on user roles (staff, superuser, regular user)
and that the registration functionality works as expected for authorized users.
"""
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.test import TestCase


class RegisterViewTests(TestCase):
    """Test suite for the user registration view."""

    def setUp(self):
        """Set up URLs and create users with different permission levels."""
        self.register_url = reverse("register")
        self.not_authorised_url = reverse("not-authorised")
        self.password = "Testpass123"
        User = get_user_model()

        # Create a staff user (e.g., a manager) who should have access
        self.staff_user = User.objects.create_user(
            username="staff_user", password=self.password, is_staff=True
        )
        # Create a superuser who should have access
        self.superuser = User.objects.create_superuser(
            username="super_user", email="super@example.com", password=self.password
        )
        # Create a regular user who should NOT have access
        self.regular_user = User.objects.create_user(
            username="regular_user", password=self.password
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
        self.assertTrue(
            get_user_model().objects.filter(username="new_employee").exists()
        )

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
        self.assertFalse(
            get_user_model().objects.filter(username="should_not_create").exists()
        )
