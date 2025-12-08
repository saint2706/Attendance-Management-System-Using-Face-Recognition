from django.urls import include, path

from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView, TokenVerifyView

from recognition.views import FaceRecognitionAPI

from .views import AttendanceViewSet, UserViewSet

router = DefaultRouter()
router.register(r"users", UserViewSet, basename="user")
router.register(r"attendance", AttendanceViewSet, basename="attendance")

urlpatterns = [
    # Auth endpoints
    path("auth/login/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/verify/", TokenVerifyView.as_view(), name="token_verify"),
    # Custom endpoints that don't fit into ViewSets
    path("recognition/", FaceRecognitionAPI.as_view(), name="face-recognition"),
    # Router endpoints
    path("", include(router.urls)),
]
