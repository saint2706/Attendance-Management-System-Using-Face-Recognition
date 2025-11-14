"""
URL configuration for the Smart Attendance System project.

This module defines the URL patterns for the entire application, delegating specific
app-related URLs to their respective `urls.py` files.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include, path

from users import views as user_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("recognition.urls")),
    path("register/", user_views.register, name="register"),
    path("login/", auth_views.LoginView.as_view(template_name="users/login.html"), name="login"),
    path(
        "logout/", auth_views.LogoutView.as_view(template_name="users/logout.html"), name="logout"
    ),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
