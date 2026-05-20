from recognition.api.serializers import AttendanceRecordSerializer
from users.models import RecognitionAttempt


def test_get_username_no_user():
    attempt = RecognitionAttempt(user=None, username="")
    serializer = AttendanceRecordSerializer(attempt)
    assert serializer.get_username(attempt) == "Unknown"
