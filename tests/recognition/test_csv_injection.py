from django.test import TestCase, RequestFactory
from django.contrib.auth.models import User
from users.models import RecognitionAttempt
from recognition.models import RecognitionOutcome
from recognition.admin_views import export_attendance_csv
import csv
import io

class TestCSVInjection(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_superuser('admin', 'admin@example.com', 'password')

    def test_csv_injection_fix(self):
        # Create malicious data
        malicious_username = "=1+1"
        RecognitionOutcome.objects.create(
            username=malicious_username,
            accepted=True,
            confidence=0.9,
            distance=0.1,
            source="test"
        )

        request = self.factory.get('/admin/attendance-dashboard/export/')
        request.user = self.user

        response = export_attendance_csv(request)
        content = response.content.decode('utf-8')

        # Check if the malicious payload is sanitized (prefixed with ')
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        # Row 0 is header. Row 1 is data.
        # Timestamp, Username, Direction, ...
        username_cell = rows[1][1]

        # This confirms fix
        self.assertTrue(username_cell.startswith("'="), f"Value not sanitized: {username_cell}")
