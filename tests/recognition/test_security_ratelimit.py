
import pytest
from django.contrib.auth.models import User
from django.test import RequestFactory
from django.urls import reverse
from recognition.views_legacy import train
from django.contrib.sessions.middleware import SessionMiddleware
from django.contrib.messages.middleware import MessageMiddleware
from unittest.mock import patch, MagicMock

@pytest.mark.django_db
def test_train_view_rate_limit():
    """Test that the train view is rate limited."""
    # Create a staff user
    user = User.objects.create_user(username='staff_user', password='password', is_staff=True)

    factory = RequestFactory()
    url = reverse('train')

    # We need to mock the task delay to avoid starting actual tasks
    with patch('recognition.tasks.train_recognition_model.delay') as mock_task:
        mock_task.return_value.id = 'fake-task-id'

        for i in range(10):
            request = factory.post(url)
            request.user = user

            # Use middleware to set up session and messages properly
            middleware = SessionMiddleware(lambda x: None)
            middleware.process_request(request)
            request.session.save()

            msg_middleware = MessageMiddleware(lambda x: None)
            msg_middleware.process_request(request)

            response = train(request)

            # Check if rate limit was triggered
            if getattr(request, 'limited', False):
                return  # Test passed!

    pytest.fail("Rate limit was not triggered after 10 requests")
