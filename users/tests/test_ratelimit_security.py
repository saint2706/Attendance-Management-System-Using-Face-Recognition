import pytest
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.urls import reverse
from users.models import SetupWizardProgress

@pytest.fixture(autouse=True)
def clear_rate_limit_cache():
    """Clear rate limit cache before each test."""
    cache.clear()
    yield
    cache.clear()

@pytest.mark.django_db
def test_register_rate_limit_fix(client):
    User = get_user_model()
    staff_user = User.objects.create_user(username='staff_limit_2', password='password', is_staff=True)
    client.force_login(staff_user)
    url = reverse('register')

    # Limit is 10/m
    for i in range(10):
        response = client.post(url, {'username': f'user{i}', 'password': 'password'})
        assert response.status_code != 429

    # 11th attempt
    response = client.post(url, {'username': 'user11', 'password': 'password'})
    assert response.status_code == 429
    assert b"Too many registration attempts" in response.content

@pytest.mark.django_db
def test_wizard_step3_rate_limit(client):
    User = get_user_model()
    staff_user = User.objects.create_user(username='staff_wiz3', password='password', is_staff=True)
    # Setup prerequisite state
    SetupWizardProgress.objects.create(
        user=staff_user,
        current_step=SetupWizardProgress.Step.ADD_EMPLOYEE,
        camera_tested=True,
        liveness_tested=True
    )

    client.force_login(staff_user)
    url = reverse('setup-wizard-step3')

    # Limit is 10/m - send simple POSTs to trigger rate limit
    for i in range(10):
        # Send POST with minimal data to avoid successful form submission
        response = client.post(url, {})
        assert response.status_code != 429

    # 11th attempt - should be rate limited
    response = client.post(url, {})
    assert response.status_code == 200

    messages = list(response.context['messages'])
    assert len(messages) > 0
    assert "Too many attempts" in str(messages[0])

@pytest.mark.django_db
def test_wizard_step4_rate_limit(client):
    User = get_user_model()
    staff_user = User.objects.create_user(username='staff_wiz', password='password', is_staff=True)
    # Setup prerequisite state
    SetupWizardProgress.objects.create(
        user=staff_user,
        current_step=SetupWizardProgress.Step.TRAIN_MODEL,
        first_employee_username="testemployee",
        first_employee_photos_captured=True,
        camera_tested=True,
        liveness_tested=True
    )

    client.force_login(staff_user)
    url = reverse('setup-wizard-step4')

    # Limit is 5/m
    for i in range(5):
        # We don't send 'start_training' to avoid mocking celery here, just POSTing is enough to trigger ratelimit
        response = client.post(url, {})
        assert response.status_code != 429

    # 6th attempt
    response = client.post(url, {})
    # Note: setup_wizard steps return 200 (re-render) even if limited.
    assert response.status_code == 200

    messages = list(response.context['messages'])
    assert len(messages) > 0
    assert "Too many attempts" in str(messages[0])
