from django.contrib.auth import get_user_model
from django.utils import timezone

from rest_framework import permissions, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from recognition.api.serializers import (
    AttendanceRecordSerializer,
    RegisterEmployeeSerializer,
    StatsSerializer,
    UserSerializer,
)
from users.models import RecognitionAttempt

User = get_user_model()


class UserViewSet(viewsets.ModelViewSet):
    """
    ViewSet for viewing and editing user instances.
    """

    queryset = User.objects.all().order_by("-date_joined")
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        if self.action == "create":
            return RegisterEmployeeSerializer
        return UserSerializer

    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            return User.objects.all().order_by("-date_joined")
        return User.objects.filter(id=user.id)

    @action(detail=False, methods=["get"])
    def me(self, request):
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)


class AttendanceViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing attendance records.
    """

    serializer_class = AttendanceRecordSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        queryset = RecognitionAttempt.objects.all().order_by("-created_at")

        if not user.is_staff:
            queryset = queryset.filter(user=user)

        # Filter by date range
        start_date = self.request.query_params.get("start_date")
        end_date = self.request.query_params.get("end_date")

        if start_date:
            queryset = queryset.filter(created_at__date__gte=start_date)
        if end_date:
            queryset = queryset.filter(created_at__date__lte=end_date)

        return queryset

    @action(detail=False, methods=["get"])
    def stats(self, request):
        """
        Get dashboard statistics.
        """
        today = timezone.now().date()
        total_employees = User.objects.filter(is_active=True).count()

        # This is a simplified logic - in a real app this would be more complex
        # involving grouping by user for the day
        present_today = (
            RecognitionAttempt.objects.filter(created_at__date=today, successful=True)
            .values("user")
            .distinct()
            .count()
        )

        # Mocking check-out data for now as strictly speaking we need to pair check-ins/outs
        checked_out_today = 0
        pending_checkout = present_today - checked_out_today

        data = {
            "total_employees": total_employees,
            "present_today": present_today,
            "checked_out_today": checked_out_today,
            "pending_checkout": pending_checkout,
        }

        serializer = StatsSerializer(data)
        return Response(serializer.data)

    @action(detail=False, methods=["post"])
    def mark(self, request):
        """
        Endpoint for marking attendance via API.
        This would integrate with the core recognition logic.
        For now, this is a placeholder that returns a mocked success response
        to unblock frontend development.
        """
        # TODO: Integrate with actual recognition.views logic
        # For now, we return a success response to simulate a working backend

        return Response(
            {
                "status": "success",
                "message": "Attendance marked successfully",
                "recognition": {
                    "detected": True,
                    "confidence": 0.98,
                    "user": request.user.username,
                    "time": timezone.now().isoformat(),
                },
            }
        )
