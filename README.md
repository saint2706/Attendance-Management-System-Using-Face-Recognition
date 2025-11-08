# Modern Smart Attendance System

This project is a fully refactored and modernized smart attendance system that leverages deep learning for face recognition. It provides a seamless and automated way to track employee attendance, eliminating the need for manual record-keeping. The system is built with a responsive web interface for a great user experience on any device.

![Home Page Light Theme](docs/images/home-light.png)

## Features

- **Automated Attendance:** Mark time-in and time-out effortlessly using real-time face recognition.
- **Responsive Web Interface:** A clean, modern, and intuitive UI that works beautifully on desktops, tablets, and mobile devices.
- **Admin Dashboard:** A powerful dashboard for administrators to manage employees, add user photos, and view comprehensive attendance reports.
- **Employee Dashboard:** A personalized dashboard for employees to view their own attendance records.
- **Automatic Training:** The face recognition model updates automatically when new employee photos are added.
- **Performance Optimized:** Utilizes the efficient "Facenet" model and "SSD" detector for a fast and responsive recognition experience.
- **Continuous Integration:** Includes a GitHub Actions workflow to automatically run tests, ensuring code quality and stability.

## Technical Stack

- **Backend:** Django 5+
- **Face Recognition:** DeepFace (wrapping Facenet)
- **Frontend:** HTML5, CSS3, Bootstrap 5, Custom CSS Design System
- **JavaScript:** Vanilla JS (no framework dependencies)
- **Database:** SQLite (default, configurable in Django)
- **Testing:** Django's built-in test framework, Playwright (planned)

## Getting Started

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

## Documentation

For more detailed information, please refer to the full documentation:

- **[User Guide](USER_GUIDE.md)**: A comprehensive guide for non-programmers on using and understanding the system.
- **[Developer Guide](DEVELOPER_GUIDE.md)**: Information for developers on the system's architecture, evaluation pipeline, and management commands.
- **[Contributing Guide](CONTRIBUTING.md)**: Instructions for setting up the development environment and contributing to the project.
- **[API Reference](API_REFERENCE.md)**: Details on URL patterns, API endpoints, and command-line tools.
- **[Architecture Overview](ARCHITECTURE.md)**: A high-level overview of the system architecture and data flows.
- **[Data Card](DATA_CARD.md)**: Comprehensive documentation on the dataset, including privacy policies and data splits.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
