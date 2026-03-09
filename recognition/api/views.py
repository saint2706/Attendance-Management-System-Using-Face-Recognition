from django.contrib.auth import get_user_model
from django.utils import timezone

from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

from recognition.api.serializers import (
    AttendanceRecordSerializer,
    EmployeeSerializer,
    RegisterEmployeeSerializer,
    StatsSerializer,
    UserSerializer,
)
from users.models import Direction, RecognitionAttempt

User = get_user_model()


class UserViewSet(viewsets.ModelViewSet):
    """
    ViewSet for viewing and editing user instances.
    """

    queryset = User.objects.all().order_by("-date_joined")
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = PageNumberPagination

    def get_serializer_class(self):
        if self.action == "create":
            return RegisterEmployeeSerializer
        if hasattr(self, "request") and self.request.user.is_staff:
            return EmployeeSerializer
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
    pagination_class = PageNumberPagination

    def get_queryset(self):
        user = self.request.user
        queryset = RecognitionAttempt.objects.all().select_related("user").order_by("-created_at")

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
        Get dashboard statistics with real check-in/check-out tracking.

        Uses the direction field on RecognitionAttempt to properly calculate:
        - present_today: Users with successful check-IN today
        - checked_out_today: Users with successful check-OUT today
        - pending_checkout: Users checked in but not yet checked out
        """
        today = timezone.now().date()
        total_employees = User.objects.filter(is_active=True).count()

        # Single query to get all successful attempts for today
        attempts_today = (
            RecognitionAttempt.objects.filter(created_at__date=today, successful=True)
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

    @action(detail=False, methods=["post"])
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

        # Get direction (default to check-in)
        direction = request.data.get("direction", "in").lower()
        if direction not in ["in", "out"]:
            direction = "in"

        # Extract image from request
        image_data = request.data.get("image")
        if not image_data:
            return Response(
                {
                    "type": "about:blank",
                    "title": "Validation Error",
                    "status": status.HTTP_400_BAD_REQUEST,
                    "detail": "No image provided",
                    "instance": request.path,
                },
                status=status.HTTP_400_BAD_REQUEST,
                content_type="application/problem+json",
            )

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
                return Response(
                    {
                        "type": "about:blank",
                        "title": "Validation Error",
                        "status": status.HTTP_400_BAD_REQUEST,
                        "detail": "Unable to decode image",
                        "instance": request.path,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                    content_type="application/problem+json",
                )
        except Exception as e:
            logger.warning(f"Image decode error: {e}")
            return Response(
                {
                    "type": "about:blank",
                    "title": "Validation Error",
                    "status": status.HTTP_400_BAD_REQUEST,
                    "detail": "Invalid image format",
                    "instance": request.path,
                },
                status=status.HTTP_400_BAD_REQUEST,
                content_type="application/problem+json",
            )

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
            )
            embedding_vector, facial_area = extract_embedding(representations)

            if embedding_vector is None:
                return Response(
                    {
                        "type": "about:blank",
                        "title": "Validation Error",
                        "status": status.HTTP_400_BAD_REQUEST,
                        "detail": "No face detected in the image",
                        "instance": request.path,
                        **{"recognition": {"detected": False}},
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                    content_type="application/problem+json",
                )
        except ValueError as e:
            logger.warning(f"Face detection error: {e}")
            return Response(
                {
                    "type": "about:blank",
                    "title": "Validation Error",
                    "status": status.HTTP_400_BAD_REQUEST,
                    "detail": "No face detected in the image",
                    "instance": request.path,
                    **{"recognition": {"detected": False}},
                },
                status=status.HTTP_400_BAD_REQUEST,
                content_type="application/problem+json",
            )
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return Response(
                {
                    "type": "about:blank",
                    "title": "Internal Server Error",
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "detail": "Face recognition failed",
                    "instance": request.path,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content_type="application/problem+json",
            )

        # Find matching identity in dataset
        dataset_index = _load_dataset_embeddings_for_matching(
            model_name, detector_backend, enforce_detection
        )

        if not dataset_index:
            return Response(
                {
                    "type": "about:blank",
                    "title": "Validation Error",
                    "status": status.HTTP_400_BAD_REQUEST,
                    "detail": "No enrolled employees in the system",
                    "instance": request.path,
                    **{"recognition": {"detected": True, "matched": False}},
                },
                status=status.HTTP_400_BAD_REQUEST,
                content_type="application/problem+json",
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
            return Response(
                {
                    "type": "about:blank",
                    "title": "Validation Error",
                    "status": status.HTTP_400_BAD_REQUEST,
                    "detail": "No valid embeddings available for matching",
                    "instance": request.path,
                },
                status=status.HTTP_400_BAD_REQUEST,
                content_type="application/problem+json",
            )

        # Find closest match
        from django.conf import settings

        distance_metric = getattr(settings, "DEEPFACE_DISTANCE_METRIC", "cosine")
        match_result = find_closest_dataset_match(
            embedding_vector, normalized_index, distance_metric
        )

        if match_result is None:
            return Response(
                {
                    "type": "about:blank",
                    "title": "Match Failed",
                    "status": status.HTTP_200_OK,
                    "detail": "Face recognized but no match found in database",
                    "instance": request.path,
                    **{"recognition": {"detected": True, "matched": False}},
                },
                status=status.HTTP_200_OK,
                content_type="application/problem+json",
            )

        matched_username, distance, _ = match_result

        # Check if match is within threshold
        from recognition.pipeline import is_within_distance_threshold

        threshold = getattr(settings, "DEEPFACE_DISTANCE_THRESHOLD", 0.6)
        if not is_within_distance_threshold(distance, threshold):
            return Response(
                {
                    "type": "about:blank",
                    "title": "Validation Error",
                    "status": status.HTTP_400_BAD_REQUEST,
                    "detail": "Face not recognized with sufficient confidence",
                    "instance": request.path,
                    **{
                        "recognition": {
                            "detected": True,
                            "matched": False,
                            "confidence": (max(0, 1 - distance) if distance else 0),
                        }
                    },
                },
                status=status.HTTP_400_BAD_REQUEST,
                content_type="application/problem+json",
            )

        # Calculate confidence (inverse of distance for cosine)
        confidence = max(0, min(1, 1 - distance)) if distance is not None else 0.0

        # Get the matched user
        try:
            matched_user = User.objects.get(username=matched_username)
        except User.DoesNotExist:
            return Response(
                {
                    "type": "about:blank",
                    "title": "Validation Error",
                    "status": status.HTTP_400_BAD_REQUEST,
                    "detail": f"User '{matched_username}' not found in database",
                    "instance": request.path,
                },
                status=status.HTTP_400_BAD_REQUEST,
                content_type="application/problem+json",
            )

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
