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
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, QuerySet
from django.shortcuts import redirect, render
from django.utils import timezone

import cv2
import imutils
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deepface import DeepFace
from django_pandas.io import read_frame
from imutils.video import VideoStream
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from users.models import Present, Time

from .forms import DateForm, DateForm_2, UsernameAndDateForm, usernameForm

# Use 'Agg' backend for Matplotlib to avoid GUI-related issues in a web server environment
mpl.use("Agg")

# Initialize logger for the module
logger = logging.getLogger(__name__)


# Define root directories for data storage
DATA_ROOT = Path(settings.BASE_DIR) / "face_recognition_data"
TRAINING_DATASET_ROOT = DATA_ROOT / "training_dataset"

# Define paths for saving generated attendance graphs within MEDIA_ROOT
ATTENDANCE_GRAPHS_ROOT = Path(
    getattr(settings, "ATTENDANCE_GRAPHS_ROOT", Path(settings.MEDIA_ROOT) / "attendance_graphs")
)
HOURS_VS_DATE_PATH = ATTENDANCE_GRAPHS_ROOT / "hours_vs_date" / "1.png"
EMPLOYEE_LOGIN_PATH = ATTENDANCE_GRAPHS_ROOT / "employee_login" / "1.png"
HOURS_VS_EMPLOYEE_PATH = ATTENDANCE_GRAPHS_ROOT / "hours_vs_employee" / "1.png"
THIS_WEEK_PATH = ATTENDANCE_GRAPHS_ROOT / "this_week" / "1.png"
LAST_WEEK_PATH = ATTENDANCE_GRAPHS_ROOT / "last_week" / "1.png"


def _ensure_directory(path: Path) -> None:
    """
    Ensure the parent directory for the given path exists.

    If the directory does not exist, it is created. This is useful for
    ensuring that file paths for saving graphs are valid.

    Args:
        path: The file path whose parent directory needs to exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _media_url_for(path: Path) -> str:
    """Return the MEDIA_URL-relative URL for the provided file path."""

    media_root = Path(settings.MEDIA_ROOT)
    try:
        relative_path = path.resolve().relative_to(media_root.resolve())
    except ValueError:
        relative_path = path
    return urljoin(settings.MEDIA_URL, relative_path.as_posix())


def _save_plot_to_media(path: Path) -> str:
    """Persist the current Matplotlib plot and return its media URL."""

    _ensure_directory(path)
    try:
        plt.savefig(path)
    finally:
        plt.close()
    return _media_url_for(path)


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

    headless = _is_headless_environment()
    max_frames = int(getattr(settings, "RECOGNITION_HEADLESS_DATASET_FRAMES", 50))
    frame_pause = float(getattr(settings, "RECOGNITION_HEADLESS_FRAME_SLEEP", 0.01))

    sample_number = 0
    try:
        while True:
            # Read a frame from the video stream
            frame = video_stream.read()
            if frame is None:
                continue
            frame = imutils.resize(frame, width=800)

            sample_number += 1
            output_path = dataset_directory / f"{sample_number}.jpg"
            cv2.imwrite(str(output_path), frame)

            if not headless:
                # Display the frame to the user and allow them to quit manually
                cv2.imshow("Add Images - Press 'q' to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if frame_pause:
                    time.sleep(frame_pause)

            if sample_number >= max_frames:
                break
    finally:
        logger.info("Finished capturing images for %s.", username)
        video_stream.stop()
        if not headless:
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

    first_entry = times_all.first()
    if first_entry is None or first_entry.time is None:
        return True, 0

    # The first entry must be a check-in
    if first_entry.out:
        return False, 0

    # The number of check-ins must equal the number of check-outs
    if times_all.filter(out=False).count() != times_all.filter(out=True).count():
        return False, 0

    break_hours = 0
    prev_time = None
    is_break = False

    for entry in times_all:
        if not entry.out:  # This is a check-in
            if is_break and prev_time is not None and entry.time is not None:
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
) -> Tuple[QuerySet[Present], str]:
    """
    Calculate work and break hours for an employee over a date range and generate a plot.

    Args:
        present_qs: A queryset of `Present` objects for the employee.
        time_qs: A queryset of `Time` objects for the employee.
        admin: A boolean indicating if the view is for an admin (affects save path).

    Returns:
        A tuple containing the annotated queryset and the media URL of the generated plot.
    """
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []

    for obj in present_qs:
        date = obj.date
        times_all = time_qs.filter(date=date).order_by("time")
        times_in = times_all.filter(out=False)
        times_out = times_all.filter(out=True)

        # Use intermediate variables to avoid calling .time on None
        first_in = times_in.first()
        last_out = times_out.last()
        obj.time_in = first_in.time if first_in else None
        obj.time_out = last_out.time if last_out else None

        hours_val = 0.0
        if obj.time_in and obj.time_out:
            hours_val = (obj.time_out - obj.time_in).total_seconds() / 3600

        is_valid, break_hours_val = check_validity_times(times_all)
        if not is_valid:
            break_hours_val = 0.0

        df_hours.append(hours_val)
        df_break_hours.append(break_hours_val)

        # Format for display
        obj.hours = convert_hours_to_hours_mins(hours_val)
        obj.break_hours = convert_hours_to_hours_mins(break_hours_val)

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
    chart_url = _save_plot_to_media(target_path)

    return present_qs, chart_url


def hours_vs_employee_given_date(
    present_qs: QuerySet[Present], time_qs: QuerySet[Time]
) -> Tuple[QuerySet[Present], str]:
    """
    Calculate work and break hours for all employees on a given date and generate a plot.

    Args:
        present_qs: A queryset of `Present` objects for the date.
        time_qs: A queryset of `Time` objects for the date.

    Returns:
        A tuple containing the annotated queryset and the media URL of the generated plot.
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

        # Use intermediate variables to avoid calling .time on None
        first_in = times_in.first()
        last_out = times_out.last()
        obj.time_in = first_in.time if first_in else None
        obj.time_out = last_out.time if last_out else None

        hours_val = 0.0
        if obj.time_in and obj.time_out:
            hours_val = (obj.time_out - obj.time_in).total_seconds() / 3600

        is_valid, break_hours_val = check_validity_times(times_all)
        if not is_valid:
            break_hours_val = 0.0

        df_hours.append(hours_val)
        df_username.append(user.username)
        df_break_hours.append(break_hours_val)

        # Format for display
        obj.hours = convert_hours_to_hours_mins(hours_val)
        obj.break_hours = convert_hours_to_hours_mins(break_hours_val)

    # Generate and save the plot
    df = read_frame(present_qs, fieldnames=["user"])
    df["hours"] = df_hours
    df["username"] = df_username
    df["break_hours"] = df_break_hours

    sns.barplot(data=df, x="username", y="hours")
    plt.xticks(rotation="vertical")
    rcParams.update({"figure.autolayout": True})
    plt.tight_layout()
    chart_url = _save_plot_to_media(HOURS_VS_EMPLOYEE_PATH)

    return present_qs, chart_url


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


def this_week_emp_count_vs_date() -> Optional[str]:
    """
    Generate and save a line plot of employee presence for the current week.

    Returns:
        The media URL of the generated plot, or ``None`` when no data is available.
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
        return None  # Avoid plotting if there's no data

    df = pd.DataFrame({"date": str_dates_all, "Number of employees": emp_cnt_all})

    sns.lineplot(data=df, x="date", y="Number of employees")
    return _save_plot_to_media(THIS_WEEK_PATH)


def last_week_emp_count_vs_date() -> Optional[str]:
    """
    Generate and save a line plot of employee presence for the last week.

    Returns:
        The media URL of the generated plot, or ``None`` when no data is available.
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
    return _save_plot_to_media(LAST_WEEK_PATH)


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


def _is_headless_environment() -> bool:
    """Return ``True`` when no graphical display is available."""

    override = getattr(settings, "RECOGNITION_HEADLESS", None)
    if override is not None:
        return bool(override)

    # Windows typically has a display available when running interactively
    if sys.platform.startswith("win"):
        return False

    display_vars = ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")
    if any(os.environ.get(var) for var in display_vars):
        return False

    return True


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
    headless = _is_headless_environment()
    max_frames = int(getattr(settings, "RECOGNITION_HEADLESS_ATTENDANCE_FRAMES", 30))
    frame_pause = float(getattr(settings, "RECOGNITION_HEADLESS_FRAME_SLEEP", 0.01))
    frames_processed = 0

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
            if frame is None:
                continue
            frame = imutils.resize(frame, width=800)
            frames_processed += 1

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

                # Ensure dfs is non-empty and the first element is a pandas DataFrame
                # before accessing the DataFrame `.empty` attribute. This avoids
                # type errors when DeepFace returns non-DataFrame values.
                if dfs and len(dfs) > 0 and isinstance(dfs[0], pd.DataFrame) and not dfs[0].empty:
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
                            frame,
                            username,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        logger.info(
                            "Ignoring potential match for '%s' due to high distance %.4f",
                            Path(match["identity"]).parent.name,
                            distance,
                        )

            except Exception as e:
                logger.error("Error during face recognition loop: %s", e)

            if not headless:
                cv2.imshow(window_title, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if frame_pause:
                    time.sleep(frame_pause)
                if frames_processed >= max_frames:
                    logger.info("Headless mode reached frame limit of %d frames", max_frames)
                    break
    finally:
        vs.stop()
        if not headless:
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
    this_week_graph_url = this_week_emp_count_vs_date()
    last_week_graph_url = last_week_emp_count_vs_date()
    context = {
        "total_num_of_emp": total_num_of_emp,
        "emp_present_today": emp_present_today,
        "this_week_graph_url": this_week_graph_url,
        "last_week_graph_url": last_week_graph_url,
    }
    return render(request, "recognition/view_attendance_home.html", context)


@login_required
def view_attendance_date(request):
    """
    Admin view to see attendance for all employees on a specific date.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    qs: Optional[QuerySet[Present]] = None
    chart_url: Optional[str] = None
    if request.method == "POST":
        form = DateForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data["date"]
            logger.debug("Admin %s viewing attendance for date %s", request.user, date)

            time_qs = Time.objects.filter(date=date)
            present_qs = Present.objects.filter(date=date)

            if present_qs.exists():
                qs, chart_url = hours_vs_employee_given_date(present_qs, time_qs)
            else:
                messages.warning(request, "No records for the selected date.")
    else:
        form = DateForm()

    context = {
        "form": form,
        "qs": qs,
        "hours_vs_employee_chart_url": chart_url,
    }
    return render(request, "recognition/view_attendance_date.html", context)


@login_required
def view_attendance_employee(request):
    """
    Admin view to see attendance for a specific employee over a date range.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect("not-authorised")

    qs: Optional[QuerySet[Present]] = None
    chart_url: Optional[str] = None
    if request.method == "POST":
        form = UsernameAndDateForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            date_from = form.cleaned_data["date_from"]
            date_to = form.cleaned_data["date_to"]

            if date_to < date_from:
                messages.warning(
                    request, "Invalid date selection: 'To' date cannot be before 'From' date."
                )
                return redirect("view-attendance-employee")

            user = User.objects.filter(username=username).first()
            if user:
                time_qs = Time.objects.filter(
                    user=user, date__gte=date_from, date__lte=date_to
                ).order_by("-date")
                present_qs = Present.objects.filter(
                    user=user, date__gte=date_from, date__lte=date_to
                ).order_by("-date")

                if present_qs.exists():
                    qs, chart_url = hours_vs_date_given_employee(
                        present_qs, time_qs, admin=True
                    )
                else:
                    messages.warning(request, "No records for the selected duration.")
            else:
                messages.warning(request, "Username not found.")
    else:
        form = UsernameAndDateForm()

    context = {
        "form": form,
        "qs": qs,
        "hours_vs_date_chart_url": chart_url,
    }
    return render(request, "recognition/view_attendance_employee.html", context)


@login_required
def view_my_attendance_employee_login(request):
    """
    Employee-specific view to see their own attendance over a date range.
    """
    if request.user.is_staff or request.user.is_superuser:
        return redirect("not-authorised")

    qs: Optional[QuerySet[Present]] = None
    chart_url: Optional[str] = None
    if request.method == "POST":
        form = DateForm_2(request.POST)
        if form.is_valid():
            user = request.user
            date_from = form.cleaned_data["date_from"]
            date_to = form.cleaned_data["date_to"]

            if date_to < date_from:
                messages.warning(request, "Invalid date selection.")
                return redirect("view-my-attendance-employee-login")

            time_qs = Time.objects.filter(
                user=user, date__gte=date_from, date__lte=date_to
            ).order_by("-date")
            present_qs = Present.objects.filter(
                user=user, date__gte=date_from, date__lte=date_to
            ).order_by("-date")

            if present_qs.exists():
                qs, chart_url = hours_vs_date_given_employee(
                    present_qs, time_qs, admin=False
                )
            else:
                messages.warning(request, "No records for the selected duration.")

    else:
        form = DateForm_2()

    context = {
        "form": form,
        "qs": qs,
        "hours_vs_date_chart_url": chart_url,
    }
    return render(
        request, "recognition/view_my_attendance_employee_login.html", context
    )


def _get_face_detection_backend() -> str:
    """
    Return the configured face detection backend for DeepFace.

    Defaults to 'opencv' if not specified in Django settings.

    Returns:
        The name of the backend (e.g., 'opencv', 'ssd', 'dlib').
    """
    return getattr(settings, "RECOGNITION_FACE_DETECTION_BACKEND", "opencv")


def _get_face_recognition_model() -> str:
    """
    Return the face recognition model name configured in Django settings.

    This value is used to select the pre-trained model for face recognition.

    Returns:
        The name of the face recognition model.
    """
    return getattr(settings, "RECOGNITION_FACE_MODEL", "VGG-Face")


def _get_recognition_training_test_split_ratio() -> float:
    """
    Return the train-test split ratio for model validation.

    Defaults to 0.25 if not specified in Django settings.

    Returns:
        The proportion of the dataset to be used for testing.
    """
    return float(getattr(settings, "RECOGNITION_TRAINING_TEST_SPLIT_RATIO", 0.25))


def _get_recognition_training_seed() -> int:
    """
    Return the fixed seed for reproducible train-test splits.

    Defaults to 42 if not specified in Django settings.

    Returns:
        An integer seed for random operations.
    """
    return int(getattr(settings, "RECOGNITION_TRAINING_SEED", 42))


# ========== Views ==========


@login_required
def train_view(request):
    """
    Train the face recognition model and evaluate its performance.

    This view triggers the training process, which involves:
    1. Finding all user-specific image directories.
    2. Extracting face embeddings and splitting data into train/test sets.
    3. Training a Support Vector Classifier (SVC) on the training data.
    4. Evaluating the model on the test data and generating performance metrics.
    5. Saving the trained model, class names, and evaluation report.
    """
    if not (request.user.is_staff or request.user.is_superuser):
        messages.error(request, "Not authorised to perform this action")
        return redirect("not_authorised")

    logger.info("Training of the model has been initiated by %s.", request.user)

    # --- 1. Data Preparation ---
    image_paths = list(TRAINING_DATASET_ROOT.glob("*/*.jpg"))
    # convert Path objects to strings for DeepFace.represent
    image_paths_str = [str(p) for p in image_paths]
    if not image_paths:
        messages.error(request, "No training data found. Please add photos for users.")
        return render(request, "recognition/train.html", {"trained": False})

    class_names = [p.parent.name for p in image_paths]
    unique_classes = sorted(list(set(class_names)))

    if len(unique_classes) < 2:
        messages.error(
            request,
            "Training requires at least two different users with photos. "
            f"Found only {len(unique_classes)}.",
        )
        return render(request, "recognition/train.html", {"trained": False})

    # --- 2. Feature Extraction ---
    try:
        logger.info("Extracting embeddings from %d images...", len(image_paths))
        embeddings = DeepFace.represent(
            img_path=image_paths_str,
            model_name=_get_face_recognition_model(),
            detector_backend=_get_face_detection_backend(),
            enforce_detection=False,
        )
        # Normalize embeddings into a list of vectors regardless of DeepFace return type.
        # DeepFace.represent may return a numpy.ndarray (n x emb_dim) or a list of dicts
        # where each dict contains 'embedding' and possibly 'facial_area'.
        embedding_vectors = []
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
            # ndarray of shape (n_samples, emb_dim)
            embedding_vectors = [list(vec) for vec in embeddings]
        elif isinstance(embeddings, list):
            for item in embeddings:
                if isinstance(item, dict) and "embedding" in item:
                    embedding_vectors.append(item["embedding"])
                elif isinstance(item, (list, tuple, np.ndarray)):
                    embedding_vectors.append(list(item))
                else:
                    # Try to coerce unknown iterable-like items
                    try:
                        embedding_vectors.append(list(item))
                    except Exception:
                        logger.debug(
                            "Skipping unrecognized embedding item during training: %r", item
                        )
        else:
            logger.error("Unexpected embeddings type from DeepFace.represent: %s", type(embeddings))
            raise TypeError("Unexpected embeddings format from DeepFace.represent")
        logger.info("Successfully extracted embeddings.")
    except Exception as e:
        logger.error("Failed to extract embeddings during training: %s", e)
        messages.error(
            request,
            "An error occurred during face embedding extraction. " "Check logs for details.",
        )
        return render(request, "recognition/train.html", {"trained": False})

    # --- 3. Data Splitting ---
    test_split_ratio = _get_recognition_training_test_split_ratio()
    random_seed = _get_recognition_training_seed()

    X_train, X_test, y_train, y_test = train_test_split(
        embedding_vectors,
        class_names,
        test_size=test_split_ratio,
        random_state=random_seed,
        stratify=class_names,
    )
    logger.info("Data split: %d training samples, %d test samples.", len(X_train), len(X_test))

    # --- 4. Model Training ---
    logger.info("Training Support Vector Classifier...")
    model = SVC(gamma="auto", probability=True, random_state=random_seed)
    model.fit(X_train, y_train)
    logger.info("SVC training complete.")

    # --- 5. Model Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
    logger.info("Model Evaluation Report:\n%s", report)

    # --- 6. Save Artifacts ---
    try:
        # Save the trained model
        model_path = DATA_ROOT / "svc.sav"
        with model_path.open("wb") as f:
            pickle.dump(model, f)

        # Save the class names
        classes_path = DATA_ROOT / "classes.npy"
        np.save(classes_path, unique_classes)

        # Save the evaluation report
        report_path = DATA_ROOT / "classification_report.txt"
        with report_path.open("w") as f:
            f.write("Model Evaluation Report\n")
            f.write("=========================\n\n")
            f.write(f"Timestamp: {timezone.now()}\n")
            f.write(f"Test Split Ratio: {test_split_ratio}\n")
            f.write(f"Random Seed: {random_seed}\n\n")
            f.write(f"Accuracy: {accuracy:.2%}\n")
            f.write(f"Precision (weighted): {precision:.2f}\n")
            f.write(f"Recall (weighted): {recall:.2f}\n")
            f.write(f"F1-Score (weighted): {f1:.2f}\n\n")
            f.write("Classification Report:\n")
            # Ensure we write a string (classification_report may be str or dict-like)
            f.write(str(report))

        logger.info("Successfully saved model, classes, and evaluation report.")
        messages.success(request, "Model trained and evaluated successfully!")

    except Exception as e:
        logger.error("Failed to save training artifacts: %s", e)
        messages.error(request, "Failed to save the trained model. Check permissions.")
        return render(request, "recognition/train.html", {"trained": False})

    context = {
        "trained": True,
        "accuracy": f"{accuracy:.2%}",
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "f1_score": f"{f1:.2f}",
        "class_report": str(report).replace("\n", "<br>"),
    }
    return render(request, "recognition/train.html", context)


@login_required
def mark_attendance_view(request, attendance_type):
    """
    View to handle marking attendance (check-in or check-out) using the trained model.

    Args:
        request: The Django HttpRequest object.
        attendance_type: A string indicating the attendance type ('in' or 'out').
    """
    if not (request.user.is_staff or request.user.is_superuser):
        messages.error(request, "Not authorised to perform this action")
        return redirect("not_authorised")

    logger.info(
        "Attendance marking process ('%s') initiated by %s.",
        attendance_type,
        request.user,
    )

    # --- Load Model ---
    try:
        model_path = DATA_ROOT / "svc.sav"
        classes_path = DATA_ROOT / "classes.npy"
        with model_path.open("rb") as f:
            model = pickle.load(f)  # noqa: F841
        class_names = np.load(classes_path)
    except FileNotFoundError:
        messages.error(
            request, "Model not found. Please train the model before marking attendance."
        )
        return redirect("train")
    except Exception as e:
        logger.error("Failed to load the recognition model: %s", e)
        messages.error(request, "Failed to load the model. Check logs for details.")
        return redirect("train")

    # --- Video Stream ---
    video_stream = VideoStream(src=0).start()
    time.sleep(2.0)  # Allow camera to warm up

    present = {name: False for name in class_names}
    headless = _is_headless_environment()
    max_frames = int(getattr(settings, "RECOGNITION_HEADLESS_ATTENDANCE_FRAMES", 100))
    frame_pause = float(getattr(settings, "RECOGNITION_HEADLESS_FRAME_SLEEP", 0.01))

    frame_count = 0
    try:
        while True:
            frame = video_stream.read()
            if frame is None:
                continue
            frame = imutils.resize(frame, width=800)
            frame_count += 1

            try:
                # --- Face Recognition ---
                embeddings = DeepFace.represent(
                    img_path=frame,
                    model_name=_get_face_recognition_model(),
                    detector_backend=_get_face_detection_backend(),
                    enforce_detection=False,
                )

                # Normalize and safely extract the first embedding + facial_area
                embedding_vector = None
                facial_area = None
                if (
                    isinstance(embeddings, np.ndarray)
                    and embeddings.ndim == 2
                    and len(embeddings) > 0
                ):
                    embedding_vector = list(embeddings[0])
                elif isinstance(embeddings, list) and len(embeddings) > 0:
                    first = embeddings[0]
                    if isinstance(first, dict):
                        embedding_vector = first.get("embedding")
                        facial_area = first.get("facial_area")  # noqa: F841
                    elif isinstance(first, (list, tuple, np.ndarray)):
                        embedding_vector = list(first)
                    else:
                        try:
                            embedding_vector = list(first)
                        except Exception:
                            embedding_vector = None

                if embedding_vector is not None:
                    # proceed with prediction using the extracted vector
                    # facial_area may be None (skip drawing box in that case)
                    pass

            except Exception as e:
                logger.warning("Could not process frame for recognition: %s", e)

            if not headless:
                cv2.imshow(f"Marking Attendance ({attendance_type})", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if frame_pause:
                    time.sleep(frame_pause)

            if frame_count >= max_frames:
                break
    finally:
        video_stream.stop()
        if not headless:
            cv2.destroyAllWindows()

    # --- Update Database ---
    if attendance_type == "in":
        update_attendance_in_db_in(present)
        messages.success(request, "Checked-in users have been marked present.")
    elif attendance_type == "out":
        update_attendance_in_db_out(present)
        messages.success(request, "Checked-out users have been marked.")

    return redirect("home")
