import datetime
import os
import random
import shutil
import sys
from pathlib import Path

import django

# Set up Django environment
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")
django.setup()

from django.contrib.auth import get_user_model  # noqa: E402
from django.utils import timezone  # noqa: E402

from users.models import Direction, Present, RecognitionAttempt, Time  # noqa: E402

# Setup paths
FACES_DIR = project_root / "faces"
# Try to find where the app expects the dataset
# Defaulting to 'dataset' in project root
DATASET_ROOT = project_root / "dataset"


def setup_users_and_faces():
    """Create admin and employees, and set up face dataset."""
    print("Setting up users and faces...")

    User = get_user_model()

    # 1. Create Admin
    admin_username = "admin"
    if not User.objects.filter(username=admin_username).exists():
        User.objects.create_superuser(admin_username, "admin@example.com", "adminpass")
        print(f"Created superuser: {admin_username}")
    else:
        print(f"Superuser {admin_username} already exists")

    # 2. Get face images
    face_images = sorted(list(FACES_DIR.glob("*.jpg")))
    if not face_images:
        print("No images found in faces directory!")
        return []

    print(f"Found {len(face_images)} face images.")

    # 3. Create Employees and Dataset folders
    employees = []

    # Use first image for admin (optional, or just keep admin separate)
    # Let's use all images for employees for simplicity, or 1 for admin

    # Let's say image 0 is admin
    admin_image = face_images[0]
    _setup_single_user_face(admin_username, admin_image)

    # Remaining images for employees
    for i, img_path in enumerate(face_images[1:], 1):
        username = f"employee_{i:03d}"
        email = f"{username}@example.com"

        # Create user
        user, created = User.objects.get_or_create(username=username, defaults={"email": email})
        if created:
            user.set_password("password123")
            user.save()

        employees.append(user)

        # Setup face ref
        _setup_single_user_face(username, img_path)

    print(f"Setup complete. {len(employees)} employees prepared.")
    return employees


def _setup_single_user_face(username, source_image_path):
    """Copy image to dataset folder for the user."""
    user_dir = DATASET_ROOT / username
    user_dir.mkdir(parents=True, exist_ok=True)

    dest_path = user_dir / "reference.jpg"
    shutil.copy2(source_image_path, dest_path)


def generate_attendance_history(employees):
    """Generate 1 year of attendance data."""
    print("Generating attendance history (this may take a moment)...")

    end_date = timezone.now().date()
    start_date = end_date - datetime.timedelta(days=365)

    # Bulk create lists
    attempts_to_create = []
    times_to_create = []
    presents_to_create = []

    # Get admin user for completeness
    User = get_user_model()
    admin = User.objects.get(username="admin")
    all_users = [admin] + employees

    current_date = start_date
    while current_date <= end_date:
        # standard work week Mon-Fri
        is_weekend = current_date.weekday() >= 5

        if is_weekend:
            # Maybe 5% chance of weekend work
            if random.random() > 0.05:
                current_date += datetime.timedelta(days=1)
                continue

        for user in all_users:
            # 5% absent rate
            if random.random() < 0.05:
                # Mark absent ? Usually we just don't create records
                # But Present model might need 'present=False'
                presents_to_create.append(Present(user=user, date=current_date, present=False))
                continue

            # Present
            # 9:00 AM start, +/- 30 mins
            # Late logic: if > 9:15 considered late maybe?

            base_time = datetime.datetime.combine(current_date, datetime.time(9, 0))
            offset_minutes = random.randint(-20, 45)  # 8:40 to 9:45
            check_in_dt = base_time + datetime.timedelta(minutes=offset_minutes)
            check_in_dt = timezone.make_aware(check_in_dt)

            # Check Out
            # 5:00 PM end, +/- 60 mins
            base_out = datetime.datetime.combine(current_date, datetime.time(17, 0))
            out_offset = random.randint(-30, 90)  # 4:30 to 6:30
            check_out_dt = base_out + datetime.timedelta(minutes=out_offset)
            check_out_dt = timezone.make_aware(check_out_dt)

            # 5% chance user forgets to check out
            forgot_checkout = random.random() < 0.05

            # Create Records

            # 1. Recognition Attempts (The source of truth for stats endpoint)
            attempts_to_create.append(
                RecognitionAttempt(
                    user=user,
                    username=user.username,
                    direction=Direction.IN,
                    site="hq",
                    source="synthetic",
                    successful=True,
                    created_at=check_in_dt,
                    latency_ms=random.randint(200, 800),
                )
            )

            if not forgot_checkout:
                attempts_to_create.append(
                    RecognitionAttempt(
                        user=user,
                        username=user.username,
                        direction=Direction.OUT,
                        site="hq",
                        source="synthetic",
                        successful=True,
                        created_at=check_out_dt,
                        latency_ms=random.randint(200, 800),
                    )
                )

            # 2. Time & Present Models

            # Time model tracks individual events
            # Check-in event
            times_to_create.append(
                Time(user=user, date=current_date, time=check_in_dt, direction=Direction.IN)
            )

            # Check-out event (if applicable)
            if not forgot_checkout:
                times_to_create.append(
                    Time(user=user, date=current_date, time=check_out_dt, direction=Direction.OUT)
                )

            # Present Model (Daily status)
            p = Present(user=user, date=current_date, present=True)
            presents_to_create.append(p)

        current_date += datetime.timedelta(days=1)

    print(f"Creating {len(attempts_to_create)} recognition attempts...")
    RecognitionAttempt.objects.bulk_create(attempts_to_create, batch_size=1000)

    print(f"Creating {len(times_to_create)} time records...")
    Time.objects.bulk_create(times_to_create, batch_size=1000)

    print(f"Creating {len(presents_to_create)} present records...")
    Present.objects.bulk_create(presents_to_create, batch_size=1000)

    print("Data generation complete!")


def test_full_flow():
    """Test the stats endpoint to verify data visibility."""
    print("\nVerifying Dashboard Stats...")

    from django.test import RequestFactory

    from recognition.api.views import AttendanceViewSet

    factory = RequestFactory()
    view = AttendanceViewSet.as_view({"get": "stats"})

    request = factory.get("/api/attendance/stats/")
    User = get_user_model()
    request.user = User.objects.get(username="admin")  # Authenticate as admin

    response = view(request)
    if response.status_code == 200:
        data = response.data
        print("\nDashboard Stats Response:")
        print(f"Total Employees:   {data.get('total_employees')}")
        print(f"Present Today:     {data.get('present_today')}")
        print(f"Checked Out Today: {data.get('checked_out_today')}")
        print(f"Pending Checkout:  {data.get('pending_checkout')}")
        print("\n✅ Verification Successful: Endpoint returned valid data.")
    else:
        print(f"\n❌ Verification Failed: Status {response.status_code}")
        print(response.data)


if __name__ == "__main__":
    # Ensure dataset root exists because we will copy images there
    if not DATASET_ROOT.exists():
        DATASET_ROOT.mkdir(parents=True)

    employees = setup_users_and_faces()
    if employees:
        # Clear old data to avoid duplicates over multiple runs
        print("Clearing old synthetic data timestamps...")
        # Optional: strictly speaking we should just append, but checking stats is easier with fresh daily data
        # For simplicity, we just generate.

        generate_attendance_history(employees)
        test_full_flow()
