from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse


class LoginSecurityTests(TestCase):
    def setUp(self):
        cache.clear()
        self.login_url = reverse("login")
        User = get_user_model()
        self.user = User.objects.create_user(username="testuser", password="password")

    def test_login_rate_limit_check(self):
        """
        Check if rate limiting is applied.
        After fix: this should confirm rate limit (response 429).
        """
        # We try 6 times. If limit is 5/m, the 6th should fail.

        responses = []
        for _ in range(6):
            response = self.client.post(
                self.login_url, {"username": "testuser", "password": "wrongpassword"}
            )
            responses.append(response.status_code)

        # The last one should be 429
        self.assertEqual(
            responses[-1],
            429,
            f"Expected 429 rate limit, got {responses[-1]}. Responses: {responses}",
        )

    def test_login_success_under_limit(self):
        """Verify login works if under limit."""
        cache.clear()
        response = self.client.post(
            self.login_url, {"username": "testuser", "password": "password"}
        )
        self.assertEqual(response.status_code, 302)  # Redirects on success
