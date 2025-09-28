# Modern Smart Attendance System

This project is a fully refactored and modernized smart attendance system that leverages deep learning for face recognition. It provides a seamless and automated way to track employee attendance, eliminating the need for manual record-keeping. The system is built with a responsive web interface for a great user experience on any device.
## Attendance Management System Using Face Recognition 💻

## Features

- **Automated Attendance:** Mark time-in and time-out effortlessly using real-time face recognition.
- **Responsive Web Interface:** A clean, modern, and intuitive UI that works beautifully on desktops, tablets, and mobile devices.
- **Admin Dashboard:** A powerful dashboard for administrators to manage employees, add user photos, and view comprehensive attendance reports.
- **Employee Dashboard:** A personalized dashboard for employees to view their own attendance records.
- **Automatic Training:** The face recognition model updates automatically when new employee photos are added—no manual training required.
- **Performance Optimized:** Utilizes the efficient "Facenet" model and "SSD" detector for a fast and responsive recognition experience.
- **Continuous Integration:** Includes a GitHub Actions workflow to automatically run tests, ensuring code quality and stability.

## Technical Stack

- **Backend:** Django 5+
- **Face Recognition:** DeepFace (wrapping Facenet)
- **Frontend:** HTML5, CSS3, Bootstrap 5
- **Database:** SQLite (default, configurable in Django)
- **Testing:** Django's built-in test framework

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.12 or higher
- A webcam for face recognition

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/smart-attendance-system.git
    cd smart-attendance-system
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run database migrations:**
    ```bash
    python manage.py migrate
    ```
## Presentation 🎓

5.  **Create a superuser (admin account):**
    ```bash
    python manage.py createsuperuser
    ```
    Follow the prompts to create your admin username, email, and password.

6.  **Run the development server:**
    ```bash
    python manage.py runserver
    ```
    The application will be available at `http://127.0.0.1:8000/`.

## Usage

### 1. Admin: Register a New Employee

- Log in to the admin dashboard (`/login`) with your superuser credentials.
- From the dashboard, navigate to **Register Employee**.
- Fill out the registration form to create a new user account for the employee.

### 2. Admin: Add Photos for the Employee

- After registering the employee, go back to the admin dashboard and select **Add Photos**.
- Enter the username of the employee you just registered and click **Start Camera**.
- The system will automatically capture a set of photos for face recognition. No manual training is needed!

### 3. Employee: Mark Attendance

- On the home page, click **Mark Time-In** to clock in or **Mark Time-Out** to clock out.
- The system will activate the webcam and recognize the employee's face to record the time.

### 4. View Attendance

- **Admins** can view comprehensive attendance reports for all employees from their dashboard.
- **Employees** can log in to view their personal attendance history from their own dashboard.

---

This modernized Smart Attendance System is now easier to set up, more efficient, and more user-friendly than ever before. Enjoy a seamless attendance tracking experience!

