from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("mark_attendance/", views.mark_your_attendance, name="mark-your-attendance"),
    path("mark_attendance_out/", views.mark_your_attendance_out, name="mark-your-attendance-out"),
    path("task_status/<str:task_id>/", views.task_status, name="task-status"),
]
