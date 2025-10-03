"""
Views for the recognition app.

This module contains the view functions for the face recognition-based attendance system.
It handles requests for rendering pages, processing forms, capturing images,
marking attendance, and displaying attendance data.
"""

from __future__ import annotations

import datetime
import logging
import math
import time
from pathlib import Path
from typing import Dict

import cv2
import imutils
import matplotlib as mpl

# Use 'Agg' backend for Matplotlib to avoid GUI-related issues in a web server environment
mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, QuerySet
from django.shortcuts import redirect, render
from django.utils import timezone
from django_pandas.io import read_frame
from imutils.video import VideoStream
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
from deepface import DeepFace

from .forms import DateForm, DateForm_2, UsernameAndDateForm, usernameForm
from users.models import Present, Time

# Initialize logger for the module
logger = logging.getLogger(__name__)


# Define root directories for data storage
DATA_ROOT = Path(settings.BASE_DIR) / "face_recognition_data"
TRAINING_DATASET_ROOT = DATA_ROOT / "training_dataset"

# Define paths for saving generated attendance graphs
APP_STATIC_ROOT = Path(__file__).resolve().parent / "static" / "recognition" / "img"
HOURS_VS_DATE_PATH = APP_STATIC_ROOT / "attendance_graphs" / "hours_vs_date" / "1.png"
EMPLOYEE_LOGIN_PATH = APP_STATIC_ROOT / "attendance_graphs" / "employee_login" / "1.png"
HOURS_VS_EMPLOYEE_PATH = (
    APP_STATIC_ROOT / "attendance_graphs" / "hours_vs_employee" / "1.png"
)
THIS_WEEK_PATH = APP_STATIC_ROOT / "attendance_graphs" / "this_week" / "1.png"
LAST_WEEK_PATH = APP_STATIC_ROOT / "attendance_graphs" / "last_week" / "1.png"


def _ensure_directory(path: Path) -> None:
    """
    Ensure the parent directory for the given path exists.

    If the directory does not exist, it is created. This is useful for
    ensuring that file paths for saving graphs are valid.

    Args:
        path: The file path whose parent directory needs to exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def username_present(username: str) -> bool:
    """
    Return whether the given username exists in the system.

    Args:
        username: The username to check.

    Returns:
        True if the user exists, False otherwise.
    """
    return User.objects.filter(username=username).exists()


def create_dataset(username: str) -> None:
    """
    Capture and store face images for the provided username.

    This function initializes a video stream from the webcam, captures 50 frames,
    and saves them as JPEG images in a directory named after the user.

    Args:
        username: The username for whom the dataset is being created.
    """
    dataset_directory = TRAINING_DATASET_ROOT / username
    dataset_directory.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing video stream to capture images for %s", username)
    video_stream = VideoStream(src=0).start()

    sample_number = 0
    while True:
        # Read a frame from the video stream
        frame = video_stream.read()
        frame = imutils.resize(frame, width=800)

        sample_number += 1
        output_path = dataset_directory / f"{sample_number}.jpg"
        cv2.imwrite(str(output_path), frame)

        # Display the frame to the user
        cv2.imshow("Add Images - Press 'q' to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q") or sample_number > 50:  # Capture 50 frames
            break

    logger.info("Finished capturing images for %s.", username)
    video_stream.stop()
    cv2.destroyAllWindows()


def update_attendance_in_db_in(present: Dict[str, bool]) -> None:
    """
    Persist check-in attendance information for the provided users.

    This function records the time-in for recognized users. It creates a `Present`
    record if one doesn't exist for the day, and adds a `Time` entry for the check-in.

    Args:
        present: A dictionary mapping usernames to their presence status (True).
    """
    today = timezone.localdate()
    current_time = timezone.now()
    for person, is_present in present.items():
        user = User.objects.filter(username=person).first()
        if user is None:
            logger.warning(
                "Skipping check-in attendance update for unknown user '%s'. "
                "Training data may be stale.",
                person,
            )
            continue

        # Find or create the attendance record for the day
        qs, created = Present.objects.get_or_create(
            user=user, date=today, defaults={"present": is_present}
        )

        if not created and is_present and not qs.present:
            # Mark as present if they were previously marked absent
            qs.present = True
            qs.save(update_fields=["present"])

        if is_present:
            # Record the check-in time
            Time.objects.create(user=user, date=today, time=current_time, out=False)


def update_attendance_in_db_out(present: Dict[str, bool]) -> None:
    """
    Persist check-out attendance information for the provided users.

    This function records the time-out for recognized users by creating a `Time`
    entry with the `out` flag set to True.

    Args:
        present: A dictionary mapping usernames to their presence status (True).
    """
    today = timezone.localdate()
    current_time = timezone.now()
    for person, is_present in present.items():
        if not is_present:
            continue

        user = User.objects.filter(username=person).first()
        if user is None:
            logger.warning(
                "Skipping check-out attendance update for unknown user '%s'. "
                "Training data may be stale.",
                person,
            )
            continue
        # Record the check-out time
        Time.objects.create(user=user, date=today, time=current_time, out=True)


def check_validity_times(times_all: QuerySet[Time]) -> tuple[bool, float]:
    """
    Validate and calculate break hours from a sequence of time entries.

    This function checks if the time entries follow a valid in-out sequence and
    calculates the total break time in hours.

    Args:
        times_all: A queryset of `Time` objects for a user on a given day,
                   ordered by time.

    Returns:
        A tuple containing:
        - A boolean indicating if the time sequence is valid.
        - The total break hours as a float.
    """
    if not times_all:
        return True, 0  # No records, so no invalidity

    # The first entry must be a check-in
    if times_all.first().out:
        return False, 0

    # The number of check-ins must equal the number of check-outs
    if times_all.filter(out=False).count() != times_all.filter(out=True).count():
        return False, 0

    break_hours = 0
    prev_time = None
    is_break = False

    for entry in times_all:
        if not entry.out:  # This is a check-in
            if is_break:
                # Calculate time since the last check-out
                break_duration = (entry.time - prev_time).total_seconds() / 3600
                break_hours += break_duration
            is_break = False
        else:  # This is a check-out
            is_break = True
        prev_time = entry.time

    return True, break_hours


def convert_hours_to_hours_mins(hours: float) -> str:
    """
    Convert a float representing hours into a 'h hrs m mins' format.

    Args:
        hours: The total hours as a float.

    Returns:
        A formatted string (e.g., "8 hrs 30 mins").
    """
    h = int(hours)
    minutes = (hours - h) * 60
    m = math.ceil(minutes)
    return f"{h} hrs {m} mins"


def hours_vs_date_given_employee(
    present_qs: QuerySet[Present], time_qs: QuerySet[Time], admin: bool = True
) -> QuerySet[Present]:
    """
    Calculate work and break hours for an employee over a date range and generate a plot.

    Args:
        present_qs: A queryset of `Present` objects for the employee.
        time_qs: A queryset of `Time` objects for the employee.
        admin: A boolean indicating if the view is for an admin (affects save path).

    Returns:
        The `Present` queryset annotated with work and break hours.
    """
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []

    for obj in present_qs:
        date = obj.date
        times_all = time_qs.filter(date=date).order_by("time")
        times_in = times_all.filter(out=False)
        times_out = times_all.filter(out=True)

        obj.time_in = times_in.first().time if times_in else None
        obj.time_out = times_out.last().time if times_out else None

        if obj.time_in and obj.time_out:
            obj.hours = (obj.time_out - obj.time_in).total_seconds() / 3600
        else:
            obj.hours = 0

        is_valid, break_hours = check_validity_times(times_all)
        obj.break_hours = break_hours if is_valid else 0

        df_hours.append(obj.hours)
        df_break_hours.append(obj.break_hours)

        # Format for display
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)

    # Generate and save the plot
    df = read_frame(present_qs, fieldnames=["date"])
    df["hours"] = df_hours
    df["break_hours"] = df_break_hours
    logger.debug("Attendance dataframe for employee: %s", df)

    sns.barplot(data=df, x="date", y="hours")
    plt.xticks(rotation="vertical")
    rcParams.update({"figure.autolayout": True})
    plt.tight_layout()

    target_path = HOURS_VS_DATE_PATH if admin else EMPLOYEE_LOGIN_PATH
    _ensure_directory(target_path)
    plt.savefig(target_path)
    plt.close()

    return present_qs


def hours_vs_employee_given_date(
    present_qs: QuerySet[Present], time_qs: QuerySet[Time]
) -> QuerySet[Present]:
    """
    Calculate work and break hours for all employees on a given date and generate a plot.

    Args:
        present_qs: A queryset of `Present` objects for the date.
        time_qs: A queryset of `Time` objects for the date.

    Returns:
        The `Present` queryset annotated with work and break hours.
    """
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    df_username = []

    for obj in present_qs:
        user = obj.user
        times_all = time_qs.filter(user=user).order_by("time")
        times_in = times_all.filter(out=False)
        times_out = times_all.filter(out=True)

        obj.time_in = times_in.first().time if times_in else None
        obj.time_out = times_out.last().time if times_out else None

        if obj.time_in and obj.time_out:
            obj.hours = (obj.time_out - obj.time_in).total_seconds() / 3600
        else:
            obj.hours = 0

        is_valid, break_hours = check_validity_times(times_all)
        obj.break_hours = break_hours if is_valid else 0

        df_hours.append(obj.hours)
        df_username.append(user.username)
        df_break_hours.append(obj.break_hours)

        # Format for display
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)

    # Generate and save the plot
    df = read_frame(present_qs, fieldnames=["user"])
    df["hours"] = df_hours
    df["username"] = df_username
    df["break_hours"] = df_break_hours

    sns.barplot(data=df, x="username", y="hours")
    plt.xticks(rotation="vertical")
    rcParams.update({"figure.autolayout": True})
    plt.tight_layout()
    _ensure_directory(HOURS_VS_EMPLOYEE_PATH)
    plt.savefig(HOURS_VS_EMPLOYEE_PATH)
    plt.close()

    return present_qs


def total_number_employees() -> int:
    """
    Return the total count of non-staff, non-superuser employees.
    """
    return User.objects.filter(is_staff=False, is_superuser=False).count()


def employees_present_today() -> int:
    """
    Return the count of employees marked as present today.
    """
    today = timezone.localdate()
    return Present.objects.filter(date=today, present=True).count()


def this_week_emp_count_vs_date() -> None:
    """
    Generate and save a line plot of employee presence for the current week.
    """
    today = timezone.localdate()
    start_of_week = today - datetime.timedelta(days=today.weekday())

    # Get attendance data for the current week
    qs = (
        Present.objects.filter(date__gte=start_of_week, date__lte=today, present=True)
        .values("date")
        .annotate(emp_count=Count("user"))
        .order_by("date")
    )

    # Create a dictionary for quick lookup
    attendance_by_date = {item["date"]: item["emp_count"] for item in qs}

    # Prepare data for the plot (Monday to Friday)
    str_dates_all = []
    emp_cnt_all = []
    for i in range(5):
        current_date = start_of_week + datetime.timedelta(days=i)
        if current_date > today:
            break
        str_dates_all.append(current_date.strftime("%Y-%m-%d"))
        emp_cnt_all.append(attendance_by_date.get(current_date, 0))

    if not str_dates_all:
        return # Avoid plotting if there's no data

    df = pd.DataFrame({"date": str_dates_all, "Number of employees": emp_cnt_all})

    sns.lineplot(data=df, x="date", y="Number of employees")
    _ensure_directory(THIS_WEEK_PATH)
    plt.savefig(THIS_WEEK_PATH)
    plt.close()


def last_week_emp_count_vs_date() -> None:
    """
    Generate and save a line plot of employee presence for the last week.
    """
    today = timezone.localdate()
    start_of_last_week = today - datetime.timedelta(days=today.weekday() + 7)
    end_of_last_week = start_of_last_week + datetime.timedelta(days=4)

    # Get attendance data for the last week
    qs = (
        Present.objects.filter(
            date__gte=start_of_last_week, date__lte=end_of_last_week, present=True
        )
        .values("date")
        .annotate(emp_count=Count("user"))
        .order_by("date")
    )

    attendance_by_date = {item["date"]: item["emp_count"] for item in qs}

    # Prepare data for the plot (Monday to Friday of last week)
    str_dates_all = []
    emp_cnt_all = []
    for i in range(5):
        current_date = start_of_last_week + datetime.timedelta(days=i)
        str_dates_all.append(current_date.strftime("%Y-%m-%d"))
        emp_cnt_all.append(attendance_by_date.get(current_date, 0))

    df = pd.DataFrame({"date": str_dates_all, "emp_count": emp_cnt_all})

    sns.lineplot(data=df, x="date", y="emp_count")
    _ensure_directory(LAST_WEEK_PATH)
    plt.savefig(LAST_WEEK_PATH)
    plt.close()


# ========== Main Views ==========


def home(request):
    """
    Render the home page.
    """
    return render(request, "recognition/home.html")


@login_required
def dashboard(request):
    """
    Render the dashboard, which differs for admins and regular employees.
    """
    if request.user.is_staff or request.user.is_superuser:
        logger.debug("Rendering admin dashboard for %s", request.user)
        return render(request, "recognition/admin_dashboard.html")

    logger.debug("Rendering employee dashboard for %s", request.user)
    return render(request, "recognition/employee_dashboard.html")


@login_required
def add_photos(request):
    """
    Handle the 'Add Photos' functionality for admins to create face datasets for users.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    if request.method == "POST":
        form = usernameForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            if username_present(username):
                create_dataset(username)
                messages.success(request, f"Dataset Created for {username}")
                return redirect("add-photos")

            messages.warning(request, "No such username found. Please register employee first.")
            return redirect("dashboard")
    else:
        form = usernameForm()

    return render(request, "recognition/add_photos.html", {"form": form})


# Default distance threshold for face recognition confidence
DEFAULT_DISTANCE_THRESHOLD = 0.4


def _mark_attendance(request, check_in: bool):
    """
    Core logic for marking attendance (both in and out) using face recognition.

    Args:
        request: The Django HttpRequest object.
        check_in: True for marking time-in, False for time-out.
    """
    vs = VideoStream(src=0).start()
    present = {}
    db_path = str(TRAINING_DATASET_ROOT)

    # Configure DeepFace settings
    model_name = "Facenet"
    detector_backend = "ssd"
    distance_threshold = getattr(
        settings, "RECOGNITION_DISTANCE_THRESHOLD", DEFAULT_DISTANCE_THRESHOLD
    )

    window_title = (
        "Mark Attendance - In - Press q to exit"
        if check_in
        else "Mark Attendance- Out - Press q to exit"
    )

    try:
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=800)

            try:
                # Use DeepFace to find matching faces in the database
                dfs = DeepFace.find(
                    img_path=frame,
                    db_path=db_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                    silent=True,
                )

                # Process the first valid dataframe of results
                if dfs and not dfs[0].empty:
                    df = dfs[0]
                    # Sort by distance to get the best match
                    if "distance" in df.columns:
                        df = df.sort_values(by="distance")

                    match = df.iloc[0]
                    distance = match.get("distance", 0.0)

                    # Check if the match is within the confidence threshold
                    if distance <= distance_threshold:
                        identity_path = Path(match["identity"])
                        username = identity_path.parent.name
                        present[username] = True
                        logger.info("Recognized %s with distance %.4f", username, distance)

                        # Draw a bounding box and name on the frame
                        x, y, w, h = (
                            int(match["source_x"]),
                            int(match["source_y"]),
                            int(match["source_w"]),
                            int(match["source_h"]),
                        )
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            frame, username, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                        )
                    else:
                        logger.info(
                            "Ignoring potential match for '%s' due to high distance %.4f",
                            Path(match["identity"]).parent.name, distance
                        )

            except Exception as e:
                logger.error("Error during face recognition loop: %s", e)

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        vs.stop()
        cv2.destroyAllWindows()

    # Update the database based on whether it's a check-in or check-out
    if check_in:
        update_attendance_in_db_in(present)
    else:
        update_attendance_in_db_out(present)

    return redirect("home")


@login_required
def mark_your_attendance(request):
    """View to handle marking time-in."""
    return _mark_attendance(request, check_in=True)


@login_required
def mark_your_attendance_out(request):
    """View to handle marking time-out."""
    return _mark_attendance(request, check_in=False)


@login_required
def train(request):
    """
    This view is now obsolete, as DeepFace handles recognition implicitly
    by searching through image folders. It redirects to the dashboard with an
    informational message.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    messages.info(
        request,
        "The training process is now automatic. Just add photos for new users.",
    )
    return redirect("dashboard")


@login_required
def not_authorised(request):
    """Render a page for users trying to access unauthorized areas."""
    return render(request, "recognition/not_authorised.html")


@login_required
def view_attendance_home(request):
    """
    Render the main attendance viewing page for admins.

    This view displays summary statistics and generates weekly attendance graphs.
    """
    total_num_of_emp = total_number_employees()
    emp_present_today = employees_present_today()
    this_week_emp_count_vs_date()
    last_week_emp_count_vs_date()
    context = {
        "total_num_of_emp": total_num_of_emp,
        "emp_present_today": emp_present_today,
    }
    return render(request, "recognition/view_attendance_home.html", context)


@login_required
def view_attendance_date(request):
    """
    Admin view to see attendance for all employees on a specific date.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    qs = None
    if request.method == "POST":
        form = DateForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data["date"]
            logger.debug("Admin %s viewing attendance for date %s", request.user, date)

            time_qs = Time.objects.filter(date=date)
            present_qs = Present.objects.filter(date=date)

            if present_qs.exists():
                qs = hours_vs_employee_given_date(present_qs, time_qs)
            else:
                messages.warning(request, "No records for the selected date.")

            return render(request, "recognition/view_attendance_date.html", {"form": form, "qs": qs})
    else:
        form = DateForm()

    return render(request, "recognition/view_attendance_date.html", {"form": form, "qs": qs})


@login_required
def view_attendance_employee(request):
    """
    Admin view to see attendance for a specific employee over a date range.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    qs = None
    if request.method == "POST":
        form = UsernameAndDateForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            date_from = form.cleaned_data["date_from"]
            date_to = form.cleaned_data["date_to"]

            if date_to < date_from:
                messages.warning(request, "Invalid date selection: 'To' date cannot be before 'From' date.")
                return redirect("view-attendance-employee")

            user = User.objects.filter(username=username).first()
            if user:
                time_qs = Time.objects.filter(user=user, date__gte=date_from, date__lte=date_to).order_by("-date")
                present_qs = Present.objects.filter(user=user, date__gte=date_from, date__lte=date_to).order_by("-date")

                if present_qs.exists():
                    qs = hours_vs_date_given_employee(present_qs, time_qs, admin=True)
                else:
                    messages.warning(request, "No records for the selected duration.")
            else:
                messages.warning(request, "Username not found.")

            return render(request, "recognition/view_attendance_employee.html", {"form": form, "qs": qs})
    else:
        form = UsernameAndDateForm()

    return render(request, "recognition/view_attendance_employee.html", {"form": form, "qs": qs})


@login_required
def view_my_attendance_employee_login(request):
    """
    Employee-specific view to see their own attendance over a date range.
    """
    if request.user.is_staff or request.user.is_superuser:
        return redirect("not-authorised")

    qs = None
    if request.method == "POST":
        form = DateForm_2(request.POST)
        if form.is_valid():
            user = request.user
            date_from = form.cleaned_data["date_from"]
            date_to = form.cleaned_data["date_to"]

            if date_to < date_from:
                messages.warning(request, "Invalid date selection.")
                return redirect("view-my-attendance-employee-login")

            time_qs = Time.objects.filter(user=user, date__gte=date_from, date__lte=date_to).order_by("-date")
            present_qs = Present.objects.filter(user=user, date__gte=date_from, date__lte=date_to).order_by("-date")

            if present_qs.exists():
                qs = hours_vs_date_given_employee(present_qs, time_qs, admin=False)
            else:
                messages.warning(request, "No records for the selected duration.")

            return render(request, "recognition/view_my_attendance_employee_login.html", {"form": form, "qs": qs})
    else:
        form = DateForm_2()

    return render(request, "recognition/view_my_attendance_employee_login.html", {"form": form, "qs": qs})
