"""Views for the recognition app."""

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
logger = logging.getLogger(__name__)


DATA_ROOT = Path(settings.BASE_DIR) / "face_recognition_data"
TRAINING_DATASET_ROOT = DATA_ROOT / "training_dataset"

APP_STATIC_ROOT = Path(__file__).resolve().parent / "static" / "recognition" / "img"
HOURS_VS_DATE_PATH = APP_STATIC_ROOT / "attendance_graphs" / "hours_vs_date" / "1.png"
EMPLOYEE_LOGIN_PATH = APP_STATIC_ROOT / "attendance_graphs" / "employee_login" / "1.png"
HOURS_VS_EMPLOYEE_PATH = (
    APP_STATIC_ROOT / "attendance_graphs" / "hours_vs_employee" / "1.png"
)
THIS_WEEK_PATH = APP_STATIC_ROOT / "attendance_graphs" / "this_week" / "1.png"
LAST_WEEK_PATH = APP_STATIC_ROOT / "attendance_graphs" / "last_week" / "1.png"


def _ensure_directory(path: Path) -> None:
    """Ensure the parent directory for the given path exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


def username_present(username: str) -> bool:
    """Return whether the given username exists in the system."""

    return User.objects.filter(username=username).exists()


def create_dataset(username: str) -> None:
    """Capture and store face images for the provided username."""

    dataset_directory = TRAINING_DATASET_ROOT / username
    dataset_directory.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing video stream")
    video_stream = VideoStream(src=0).start()

    sample_number = 0
    while True:
        frame = video_stream.read()
        frame = imutils.resize(frame, width=800)

        sample_number += 1
        output_path = dataset_directory / f"{sample_number}.jpg"
        cv2.imwrite(str(output_path), frame)

        cv2.imshow("Add Images", frame)
        cv2.waitKey(1)
        if sample_number > 50:  # Capture 50 frames
            break

    video_stream.stop()
    cv2.destroyAllWindows()


def update_attendance_in_db_in(present: Dict[str, bool]) -> None:
    """Persist check-in attendance information for the provided users."""

    today = timezone.localdate()
    current_time = timezone.now()
    for person, is_present in present.items():
        user = User.objects.get(username=person)
        qs = Present.objects.filter(user=user, date=today).first()

        if qs is None:
            attendance = Present(user=user, date=today, present=is_present)
            attendance.save()
        elif is_present and not qs.present:
            qs.present = True
            qs.save(update_fields=["present"])

        if is_present:
            Time.objects.create(user=user, date=today, time=current_time, out=False)


def update_attendance_in_db_out(present: Dict[str, bool]) -> None:
    """Persist check-out attendance information for the provided users."""

    today = timezone.localdate()
    current_time = timezone.now()
    for person, is_present in present.items():
        if not is_present:
            continue

        user = User.objects.get(username=person)
        Time.objects.create(user=user, date=today, time=current_time, out=True)


def check_validity_times(times_all):

    if len(times_all) > 0:
        sign = times_all.first().out
    else:
        sign = True
    times_in = times_all.filter(out=False)
    times_out = times_all.filter(out=True)
    if len(times_in) != len(times_out):
        sign = True
    break_hourss = 0
    if sign == True:
        check = False
        break_hourss = 0
        return (check, break_hourss)
    prev = True
    prev_time = times_all.first().time

    for obj in times_all:
        curr = obj.out
        if curr == prev:
            check = False
            break_hourss = 0
            return (check, break_hourss)
        if curr == False:
            curr_time = obj.time
            to = curr_time
            ti = prev_time
            break_time = ((to - ti).total_seconds()) / 3600
            break_hourss += break_time

        else:
            prev_time = obj.time

        prev = curr

    return (True, break_hourss)


def convert_hours_to_hours_mins(hours):

    h = int(hours)
    hours -= h
    m = hours * 60
    m = math.ceil(m)
    return str(str(h) + " hrs " + str(m) + "  mins")


# used
def hours_vs_date_given_employee(present_qs, time_qs, admin=True):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    qs = present_qs

    for obj in qs:
        date = obj.date
        times_in = time_qs.filter(date=date, out=False).order_by("time")
        times_out = time_qs.filter(date=date, out=True).order_by("time")
        times_all = time_qs.filter(date=date).order_by("time")
        obj.time_in = times_in.first().time if times_in else None
        obj.time_out = times_out.last().time if times_out else None
        if obj.time_in and obj.time_out:
            hours = (obj.time_out - obj.time_in).total_seconds() / 3600
            obj.hours = hours
        else:
            obj.hours = 0
        check, break_hours = check_validity_times(times_all)
        obj.break_hours = break_hours if check else 0
        df_hours.append(obj.hours)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)

    df = read_frame(qs)
    df["hours"] = df_hours
    df["break_hours"] = df_break_hours
    logger.debug("Attendance dataframe: %s", df)

    sns.barplot(data=df, x="date", y="hours")
    plt.xticks(rotation="vertical")
    rcParams.update({"figure.autolayout": True})
    plt.tight_layout()
    target_path = HOURS_VS_DATE_PATH if admin else EMPLOYEE_LOGIN_PATH
    _ensure_directory(target_path)
    plt.savefig(target_path)
    plt.close()
    return qs


# used
def hours_vs_employee_given_date(present_qs, time_qs):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    df_username = []
    qs = present_qs

    for obj in qs:
        user = obj.user
        times_in = time_qs.filter(user=user, out=False)
        times_out = time_qs.filter(user=user, out=True)
        times_all = time_qs.filter(user=user)
        obj.time_in = times_in.first().time if times_in else None
        obj.time_out = times_out.last().time if times_out else None
        if obj.time_in and obj.time_out:
            obj.hours = (obj.time_out - obj.time_in).total_seconds() / 3600
        else:
            obj.hours = 0
        check, break_hours = check_validity_times(times_all)
        obj.break_hours = break_hours if check else 0
        df_hours.append(obj.hours)
        df_username.append(user.username)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)

    df = read_frame(qs)
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
    return qs


def total_number_employees() -> int:
    qs = User.objects.all()
    return len(qs) - 1


def employees_present_today() -> int:
    today = timezone.localdate()
    qs = Present.objects.filter(date=today, present=True)
    return len(qs)


# used
def this_week_emp_count_vs_date() -> None:
    today = timezone.localdate()
    some_day_last_week = today - datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - datetime.timedelta(
        days=(some_day_last_week.isocalendar()[2] - 1)
    )
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(date__gte=monday_of_this_week, date__lte=today)
    str_dates = []
    emp_count = []
    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs = Present.objects.filter(date=date, present=True)
        emp_count.append(len(qs))

    while cnt < 5:
        date = str(monday_of_this_week + datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if str_dates.count(date) > 0:
            idx = str_dates.index(date)

            emp_cnt_all.append(emp_count[idx])
        else:
            emp_cnt_all.append(0)

    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["Number of employees"] = emp_cnt_all

    sns.lineplot(data=df, x="date", y="Number of employees")
    _ensure_directory(THIS_WEEK_PATH)
    plt.savefig(THIS_WEEK_PATH)
    plt.close()


# used
def last_week_emp_count_vs_date() -> None:
    today = timezone.localdate()
    some_day_last_week = today - datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - datetime.timedelta(
        days=(some_day_last_week.isocalendar()[2] - 1)
    )
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(
        date__gte=monday_of_last_week, date__lt=monday_of_this_week
    )
    str_dates = []
    emp_count = []

    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs = Present.objects.filter(date=date, present=True)
        emp_count.append(len(qs))

    while cnt < 5:
        date = str(monday_of_last_week + datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if str_dates.count(date) > 0:
            idx = str_dates.index(date)

            emp_cnt_all.append(emp_count[idx])

        else:
            emp_cnt_all.append(0)

    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["emp_count"] = emp_cnt_all

    sns.lineplot(data=df, x="date", y="emp_count")
    _ensure_directory(LAST_WEEK_PATH)
    plt.savefig(LAST_WEEK_PATH)
    plt.close()


# Create your views here.
def home(request):

    return render(request, "recognition/home.html")


@login_required
def dashboard(request):
    if request.user.username == "admin":
        logger.debug("Rendering admin dashboard for %s", request.user)
        return render(request, "recognition/admin_dashboard.html")
    else:
        logger.debug("Rendering employee dashboard for %s", request.user)

        return render(request, "recognition/employee_dashboard.html")


@login_required
def add_photos(request):
    if request.user.username != "admin":
        return redirect("not-authorised")
    if request.method == "POST":
        form = usernameForm(request.POST)
        data = request.POST.copy()
        username = data.get("username")
        if username_present(username):
            create_dataset(username)
            messages.success(request, f"Dataset Created for {username}")
            return redirect("add-photos")
        else:
            messages.warning(
                request, f"No such username found. Please register employee first."
            )
            return redirect("dashboard")

    else:

        form = usernameForm()
        return render(request, "recognition/add_photos.html", {"form": form})


def _mark_attendance(request, check_in):
    vs = VideoStream(src=0).start()
    present = {}
    db_path = str(TRAINING_DATASET_ROOT)
    # Switched to a faster model and detector for better performance
    model_name = "Facenet"
    detector_backend = "ssd"

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
                dfs = DeepFace.find(
                    img_path=frame,
                    db_path=db_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                    silent=True,
                )
                if dfs and not dfs[0].empty:
                    df = dfs[0]
                    if not df.empty:
                        # Get the path of the matched image
                        identity_path = df.iloc[0]["identity"]
                        # Extract username from the path
                        username = Path(identity_path).parent.name
                        present[username] = True
                        logger.info(f"Recognized {username}")
                        # Draw rectangle and put text
                        x = int(df.iloc[0]["source_x"])
                        y = int(df.iloc[0]["source_y"])
                        w = int(df.iloc[0]["source_w"])
                        h = int(df.iloc[0]["source_h"])
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

            except Exception as e:
                logger.error(f"Error in face recognition: {e}")

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        vs.stop()
        cv2.destroyAllWindows()

    if check_in:
        update_attendance_in_db_in(present)
    else:
        update_attendance_in_db_out(present)
    return redirect("home")


def mark_your_attendance(request):
    return _mark_attendance(request, check_in=True)


def mark_your_attendance_out(request):
    return _mark_attendance(request, check_in=False)


@login_required
def train(request):
    """
    This view is now obsolete as DeepFace does not require a separate training step.
    The 'training' is implicitly handled by organizing images in folders.
    """
    if request.user.username != "admin":
        return redirect("not-authorised")

    messages.info(
        request,
        "The training process is now automatic. Just add photos for new users.",
    )
    return redirect("dashboard")


@login_required
def not_authorised(request):
    return render(request, "recognition/not_authorised.html")


@login_required
def view_attendance_home(request):
    total_num_of_emp = total_number_employees()
    emp_present_today = employees_present_today()
    this_week_emp_count_vs_date()
    last_week_emp_count_vs_date()
    return render(
        request,
        "recognition/view_attendance_home.html",
        {"total_num_of_emp": total_num_of_emp, "emp_present_today": emp_present_today},
    )


@login_required
def view_attendance_date(request):
    if request.user.username != "admin":
        return redirect("not-authorised")
    qs = None
    time_qs = None
    present_qs = None

    if request.method == "POST":
        form = DateForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data.get("date")
            logger.debug("Viewing attendance for date %s", date)
            time_qs = Time.objects.filter(date=date)
            present_qs = Present.objects.filter(date=date)
            if time_qs or present_qs:
                qs = hours_vs_employee_given_date(present_qs, time_qs)
                return render(
                    request,
                    "recognition/view_attendance_date.html",
                    {"form": form, "qs": qs},
                )
            messages.warning(request, "No records for selected date.")
            return redirect("view-attendance-date")
    else:
        form = DateForm()
    return render(
        request, "recognition/view_attendance_date.html", {"form": form, "qs": qs}
    )


@login_required
def view_attendance_employee(request):
    if request.user.username != "admin":
        return redirect("not-authorised")
    time_qs = None
    present_qs = None
    qs = None

    if request.method == "POST":
        form = UsernameAndDateForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            if username_present(username):
                user = User.objects.get(username=username)
                time_qs = Time.objects.filter(user=user)
                present_qs = Present.objects.filter(user=user)
                date_from = form.cleaned_data.get("date_from")
                date_to = form.cleaned_data.get("date_to")
                if date_to < date_from:
                    messages.warning(request, "Invalid date selection.")
                    return redirect("view-attendance-employee")
                time_qs = time_qs.filter(
                    date__gte=date_from, date__lte=date_to
                ).order_by("-date")
                present_qs = present_qs.filter(
                    date__gte=date_from, date__lte=date_to
                ).order_by("-date")
                if time_qs or present_qs:
                    qs = hours_vs_date_given_employee(present_qs, time_qs, admin=True)
                    return render(
                        request,
                        "recognition/view_attendance_employee.html",
                        {"form": form, "qs": qs},
                    )
                messages.warning(request, "No records for selected duration.")
                return redirect("view-attendance-employee")
            messages.warning(request, "No such username found.")
            return redirect("view-attendance-employee")
    else:
        form = UsernameAndDateForm()
    return render(
        request, "recognition/view_attendance_employee.html", {"form": form, "qs": qs}
    )


def view_my_attendance_employee_login(request):
    if request.user.username == "admin":
        return redirect("not-authorised")
    qs = None
    time_qs = None
    present_qs = None
    if request.method == "POST":
        form = DateForm_2(request.POST)
        if form.is_valid():
            u = request.user
            time_qs = Time.objects.filter(user=u)
            present_qs = Present.objects.filter(user=u)
            date_from = form.cleaned_data.get("date_from")
            date_to = form.cleaned_data.get("date_to")
            if date_to < date_from:
                messages.warning(request, f"Invalid date selection.")
                return redirect("view-my-attendance-employee-login")
            else:

                time_qs = (
                    time_qs.filter(date__gte=date_from)
                    .filter(date__lte=date_to)
                    .order_by("-date")
                )
                present_qs = (
                    present_qs.filter(date__gte=date_from)
                    .filter(date__lte=date_to)
                    .order_by("-date")
                )

                if len(time_qs) > 0 or len(present_qs) > 0:
                    qs = hours_vs_date_given_employee(present_qs, time_qs, admin=False)
                    return render(
                        request,
                        "recognition/view_my_attendance_employee_login.html",
                        {"form": form, "qs": qs},
                    )
                else:

                    messages.warning(request, f"No records for selected duration.")
                    return redirect("view-my-attendance-employee-login")
    else:

        form = DateForm_2()
        return render(
            request,
            "recognition/view_my_attendance_employee_login.html",
            {"form": form, "qs": qs},
        )
