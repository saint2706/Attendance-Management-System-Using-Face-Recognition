"""
Main URL configuration for the Smart Attendance System project.

This module defines the primary URL routing for the entire application. It maps
URL paths to their corresponding view functions from the `recognition` and `users`
apps, as well as to Django's built-in authentication views.
"""

from pathlib import Path

from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from django.http import FileResponse, Http404, HttpRequest
from django.urls import include, path

from recognition import admin_views as recog_admin_views
from recognition import views as recog_views
from users import views as users_views


def _serve_static_asset(relative_path: str, *, content_type: str) -> FileResponse:
    """Stream a collected static asset so it can be cached by the browser."""

    file_path = Path(settings.BASE_DIR, relative_path)
    if not file_path.exists():
        raise Http404(f"Static asset not found: {relative_path}")

    return FileResponse(file_path.open("rb"), content_type=content_type)


def progressive_web_app_manifest(request: HttpRequest) -> FileResponse:
    """Expose the web manifest with the correct MIME type."""

    return _serve_static_asset(
        "recognition/static/manifest.json",
        content_type="application/manifest+json",
    )


def progressive_web_app_service_worker(request: HttpRequest) -> FileResponse:
    """Serve the service worker from the site root so it can control all pages."""

    response = _serve_static_asset(
        "recognition/static/js/sw.js",
        content_type="application/javascript",
    )
    response["Service-Worker-Allowed"] = "/"
    return response


urlpatterns = [
    path("manifest.json", progressive_web_app_manifest, name="pwa-manifest"),
    path("sw.js", progressive_web_app_service_worker, name="service-worker"),
    # Custom Admin Views
    path(
        "admin/evaluation/",
        recog_admin_views.evaluation_dashboard,
        name="admin_evaluation_dashboard",
    ),
    path(
        "admin/ablation/",
        recog_admin_views.ablation_results,
        name="admin_ablation_results",
    ),
    path(
        "admin/failures/",
        recog_admin_views.failure_analysis,
        name="admin_failure_analysis",
    ),
    path(
        "admin/health/",
        recog_admin_views.system_health_dashboard,
        name="admin_system_health",
    ),
    path(
        "admin/attendance-dashboard/",
        recog_admin_views.attendance_dashboard,
        name="admin_attendance_dashboard",
    ),
    path(
        "admin/attendance-dashboard/export/",
        recog_admin_views.export_attendance_csv,
        name="admin_attendance_export",
    ),
    # Core pages
    path("", recog_views.home, name="home"),
    path("dashboard/", recog_views.dashboard, name="dashboard"),
    # Authentication views
    path(
        "login/",
        auth_views.LoginView.as_view(template_name="users/login.html"),
        name="login",
    ),
    path(
        "logout/",
        auth_views.LogoutView.as_view(template_name="recognition/home.html"),
        name="logout",
    ),
    # User and Photo Management (Admin-only)
    path("register/", users_views.register, name="register"),
    path("add_photos/", recog_views.add_photos, name="add-photos"),
    path("train/", recog_views.train, name="train"),
    path("tasks/<str:task_id>/", recog_views.task_status, name="task-status"),
    # Face Recognition and Attendance Marking
    path(
        "mark_your_attendance",
        recog_views.mark_your_attendance,
        name="mark-your-attendance",
    ),
    path(
        "mark_your_attendance_out",
        recog_views.mark_your_attendance_out,
        name="mark-your-attendance-out",
    ),
    path(
        "attendance_session/",
        recog_views.attendance_session,
        name="attendance-session",
    ),
    path(
        "attendance_session/feed/",
        recog_views.attendance_session_feed,
        name="attendance-session-feed",
    ),
    path(
        "api/face-recognition/",
        recog_views.FaceRecognitionAPI.as_view(),
        name="face-recognition-api",
    ),
    path(
        "api/attendance/batch/",
        recog_views.enqueue_attendance_batch,
        name="attendance-batch",
    ),
    path("monitoring/metrics/", recog_views.monitoring_metrics, name="monitoring-metrics"),
    # Attendance Viewing
    path(
        "view_attendance_home",
        recog_views.view_attendance_home,
        name="view-attendance-home",
    ),
    path(
        "view_attendance_date",
        recog_views.view_attendance_date,
        name="view-attendance-date",
    ),
    path(
        "view_attendance_employee",
        recog_views.view_attendance_employee,
        name="view-attendance-employee",
    ),
    path(
        "view_my_attendance",
        recog_views.view_my_attendance_employee_login,
        name="view-my-attendance-employee-login",
    ),
    # Error/Status Pages
    path("not_authorised", recog_views.not_authorised, name="not-authorised"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG or getattr(settings, "SILKY_AUTHORISATION", False):
    urlpatterns += [path("silk/", include("silk.urls", namespace="silk"))]
