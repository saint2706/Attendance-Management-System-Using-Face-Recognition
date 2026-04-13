import datetime

from django.contrib.auth import get_user_model
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.views.decorators.vary import vary_on_headers

from drf_spectacular.utils import OpenApiResponse, extend_schema
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle

from recognition.api.serializers import (
    AttendanceFilterSerializer,
    AttendanceRecordSerializer,
    EmployeeSerializer,
    RecognitionRequestSerializer,
    RegisterEmployeeSerializer,
    StatsSerializer,
    UserSerializer,
)
from users.models import Direction, RecognitionAttempt

User = get_user_model()


class AttendanceRateThrottle(UserRateThrottle):
    scope = "attendance"

    def get_rate(self):
        from django.conf import settings

        return getattr(settings, "RECOGNITION_ATTENDANCE_RATE_LIMIT", "5/m")


class UserViewSet(viewsets.ModelViewSet):
    """
    ViewSet for viewing and editing user instances.
    """

    queryset = User.objects.prefetch_related("groups", "user_permissions").order_by("-date_joined")
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = PageNumberPagination

    def get_permissions(self):
        if self.action == "create":
            return [permissions.IsAdminUser()]
        return super().get_permissions()

    def get_serializer_class(self):
        if self.action == "create":
            return RegisterEmployeeSerializer
        if hasattr(self, "request") and self.request.user.is_staff:
            return EmployeeSerializer
        return UserSerializer

    def get_queryset(self):
        # ⚡ Bolt: Added prefetch_related for groups and user_permissions to
        # prevent N+1 queries when serializing the User model.
        user = self.request.user
        if user.is_staff:
            return (
                User.objects.all()
                .prefetch_related("groups", "user_permissions")
                .order_by("-date_joined")
            )
        return User.objects.filter(id=user.id).prefetch_related("groups", "user_permissions")

    @extend_schema(responses={200: UserSerializer})
    @action(detail=False, methods=["get"])
    def me(self, request):
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)


class AttendanceViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing attendance records.
    """

    queryset = RecognitionAttempt.objects.none()
    serializer_class = AttendanceRecordSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = PageNumberPagination

    def get_queryset(self):
        user = self.request.user
        queryset = (
            RecognitionAttempt.objects.all()
            .select_related("user", "present_record", "time_record")
            .prefetch_related("user__groups", "user__user_permissions")
            .order_by("-created_at")
        )

        if not user.is_staff:
            queryset = queryset.filter(user=user)

        # Filter by date range
        filter_serializer = AttendanceFilterSerializer(data=self.request.query_params)
        filter_serializer.is_valid(raise_exception=True)

        start_date = filter_serializer.validated_data.get("start_date")
        end_date = filter_serializer.validated_data.get("end_date")

        if start_date:
            start_datetime = timezone.make_aware(
                datetime.datetime.combine(start_date, datetime.time.min)
            )
            queryset = queryset.filter(created_at__gte=start_datetime)

        if end_date:
            end_datetime = timezone.make_aware(
                datetime.datetime.combine(end_date, datetime.time.max)
            )
            queryset = queryset.filter(created_at__lte=end_datetime)

        return queryset

    @extend_schema(responses={200: StatsSerializer})
    @action(detail=False, methods=["get"])
    @method_decorator(cache_page(60 * 5))
    @method_decorator(vary_on_headers("Authorization", "Cookie"))
    def stats(self, request):
        """
        Get dashboard statistics with real check-in/check-out tracking.

        Uses the direction field on RecognitionAttempt to properly calculate:
        - present_today: Users with successful check-IN today
        - checked_out_today: Users with successful check-OUT today
        - pending_checkout: Users checked in but not yet checked out
        """
        today = timezone.localdate()
        total_employees = User.objects.filter(is_active=True).count()

        today_start = timezone.make_aware(datetime.datetime.combine(today, datetime.time.min))
        today_end = timezone.make_aware(datetime.datetime.combine(today, datetime.time.max))

        # Single query to get all successful attempts for today
        attempts_today = (
            RecognitionAttempt.objects.filter(
                created_at__range=(today_start, today_end), successful=True
            )
            .values_list("user_id", "direction")
            .distinct()
        )

        checked_in_user_ids = set()
        checked_out_user_ids = set()

        for user_id, direction in attempts_today:
            if direction == Direction.IN:
                checked_in_user_ids.add(user_id)
            elif direction == Direction.OUT:
                checked_out_user_ids.add(user_id)

        present_today = len(checked_in_user_ids)
        checked_out_today = len(checked_out_user_ids)
        pending_checkout = len(checked_in_user_ids - checked_out_user_ids)

        data = {
            "total_employees": total_employees,
            "present_today": present_today,
            "checked_out_today": checked_out_today,
            "pending_checkout": pending_checkout,
        }

        serializer = StatsSerializer(data)
        return Response(serializer.data)

    @extend_schema(
        request=RecognitionRequestSerializer,
        responses={200: OpenApiResponse(description="Successful recognition")},
    )
    @action(detail=False, methods=["post"], throttle_classes=[AttendanceRateThrottle])
    def mark(self, request):
        """
        Endpoint for marking attendance via API using face recognition.

        Accepts either:
        - image: Base64-encoded face image for recognition
        - direction: 'in' or 'out' (defaults to 'in')

        Returns recognition result with matched user and confidence score.
        """
        import base64
        import logging

        import cv2
        import numpy as np
        from deepface import DeepFace
        from rest_framework.exceptions import ValidationError

        from recognition.api.exceptions import RecognitionAPIException, RecognitionException
        from recognition.pipeline import extract_embedding, find_closest_dataset_match
        from recognition.views import (
            _get_face_detection_backend,
            _get_face_recognition_model,
            _load_dataset_embeddings_for_matching,
            _should_enforce_detection,
            update_attendance_in_db_in,
            update_attendance_in_db_out,
        )

        logger = logging.getLogger(__name__)

        serializer = RecognitionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Get direction (default to check-in)
        direction = serializer.validated_data.get("direction", "in")

        # Extract image from request
        image_data = serializer.validated_data.get("image")

        # Decode base64 image
        try:
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if image_data.startswith("data:"):
                    _, _, image_data = image_data.partition(",")
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data

            # Convert to numpy array for OpenCV/DeepFace
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValidationError({"detail": "Unable to decode image"})
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            logger.warning(f"Image decode error: {e}")
            raise ValidationError({"detail": "Invalid image format"})

        # Get recognition configuration
        model_name = _get_face_recognition_model()
        detector_backend = _get_face_detection_backend()
        enforce_detection = _should_enforce_detection()

        # Extract face embedding
        try:
            representations = DeepFace.represent(
                img_path=frame,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=False,
            )
            embedding_vector, facial_area = extract_embedding(representations)

            if embedding_vector is None:
                raise RecognitionException(
                    {"detail": "No face detected in the image"},
                    recognition_data={"detected": False},
                )
        except ValueError as e:
            logger.warning(f"Face detection error: {e}")
            raise RecognitionException(
                {"detail": "No face detected in the image"},
                recognition_data={"detected": False},
            )
        except Exception as e:
            if isinstance(e, RecognitionException):
                raise
            logger.error(f"Recognition error: {e}")
            raise RecognitionAPIException({"detail": "Face recognition failed"})

        # Find matching identity in dataset
        dataset_index = _load_dataset_embeddings_for_matching(
            model_name, detector_backend, enforce_detection
        )

        if not dataset_index:
            raise RecognitionException(
                {"detail": "No enrolled employees in the system"},
                recognition_data={"detected": True, "matched": False},
            )

        # Convert dataset embeddings to proper format
        normalized_index = []
        for entry in dataset_index:
            candidate = entry.get("embedding") if isinstance(entry, dict) else None
            if candidate is not None:
                if not isinstance(candidate, np.ndarray):
                    try:
                        candidate = np.array(candidate, dtype=float)
                    except Exception:
                        continue
                normalized_entry = dict(entry)
                normalized_entry["embedding"] = candidate
                normalized_index.append(normalized_entry)

        if not normalized_index:
            raise ValidationError({"detail": "No valid embeddings available for matching"})

        # Find closest match
        from django.conf import settings

        distance_metric = getattr(settings, "DEEPFACE_DISTANCE_METRIC", "cosine")
        match_result = find_closest_dataset_match(
            embedding_vector, normalized_index, distance_metric
        )

        if match_result is None:
            response = Response(
                {
                    "type": "about:blank",
                    "title": "Match Failed",
                    "status": status.HTTP_200_OK,
                    "detail": "Face recognized but no match found in database",
                    "instance": request.path,
                    "recognition": {"detected": True, "matched": False},
                },
                status=status.HTTP_200_OK,
            )
            response.content_type = "application/problem+json"
            return response

        matched_username, distance, _ = match_result

        # Check if match is within threshold
        from recognition.pipeline import is_within_distance_threshold

        threshold = getattr(settings, "DEEPFACE_DISTANCE_THRESHOLD", 0.6)
        if not is_within_distance_threshold(distance, threshold):
            confidence = max(0, 1 - distance) if distance else 0
            raise RecognitionException(
                {"detail": "Face not recognized with sufficient confidence"},
                recognition_data={
                    "detected": True,
                    "matched": False,
                    "confidence": confidence,
                },
            )

        # Calculate confidence (inverse of distance for cosine)
        confidence = max(0, min(1, 1 - distance)) if distance is not None else 0.0

        # Get the matched user
        try:
            matched_user = User.objects.get(username=matched_username)
        except User.DoesNotExist:
            raise ValidationError({"detail": f"User '{matched_username}' not found in database"})

        # Record attendance
        now = timezone.now()

        # Create recognition attempt record
        attempt = RecognitionAttempt.objects.create(
            user=matched_user,
            username=matched_username,
            direction=Direction.IN if direction == "in" else Direction.OUT,
            successful=True,
            spoof_detected=False,
            source="api",
        )

        # Update attendance records using existing helper functions
        if direction == "in":
            update_attendance_in_db_in({matched_username: True})
        else:
            update_attendance_in_db_out({matched_username: True})

        return Response(
            {
                "status": "success",
                "message": f"Attendance marked successfully ({direction})",
                "recognition": {
                    "detected": True,
                    "matched": True,
                    "confidence": round(confidence, 4),
                    "user": matched_username,
                    "user_id": matched_user.id,
                    "direction": direction,
                    "time": now.isoformat(),
                    "attempt_id": attempt.id,
                },
            }
        )
