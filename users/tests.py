from django.contrib.auth import get_user_model
from django.urls import reverse
from django.test import TestCase


class RegisterViewTests(TestCase):

        def setUp(self):
                self.register_url = reverse('register')
                self.not_authorised_url = reverse('not-authorised')
                self.password = 'Testpass123'
                User = get_user_model()
                self.staff_user = User.objects.create_user(
                        username='staff_user',
                        password=self.password,
                        is_staff=True,
                )
                self.superuser = User.objects.create_superuser(
                        username='super_user',
                        email='super@example.com',
                        password=self.password,
                )
                self.regular_user = User.objects.create_user(
                        username='regular_user',
                        password=self.password,
                )

        def test_staff_user_can_access_register_page(self):
                self.client.force_login(self.staff_user)
                response = self.client.get(self.register_url)
                self.assertEqual(response.status_code, 200)

        def test_superuser_can_access_register_page(self):
                self.client.force_login(self.superuser)
                response = self.client.get(self.register_url)
                self.assertEqual(response.status_code, 200)

        def test_non_staff_user_redirected(self):
                self.client.force_login(self.regular_user)
                response = self.client.get(self.register_url)
                self.assertRedirects(response, self.not_authorised_url)

        def test_staff_user_can_register_new_user(self):
                self.client.force_login(self.staff_user)
                response = self.client.post(
                        self.register_url,
                        {
                                'username': 'new_employee',
                                'password1': 'Newpass12345',
                                'password2': 'Newpass12345',
                        },
                )
                self.assertRedirects(response, reverse('dashboard'))
                self.assertTrue(
                        get_user_model().objects.filter(username='new_employee').exists()
                )

        def test_non_staff_post_redirected(self):
                self.client.force_login(self.regular_user)
                response = self.client.post(
                        self.register_url,
                        {
                                'username': 'should_not_create',
                                'password1': 'Password12345',
                                'password2': 'Password12345',
                        },
                )
                self.assertRedirects(response, self.not_authorised_url)
                self.assertFalse(
                        get_user_model()
                        .objects.filter(username='should_not_create')
                        .exists()
                )
