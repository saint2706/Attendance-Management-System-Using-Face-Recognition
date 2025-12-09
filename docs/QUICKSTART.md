# Quick Start Guide

This guide shows the **quickest way** to run a local demo of the system and explore its main screens. You do not need prior knowledge of Python, Django, Docker, or virtual environments—just follow the numbered steps below.

> **⚠️ Demo only:** This setup is designed for exploring features, not for production use. For real deployments, see the [Deployment Guide](DEPLOYMENT.md) and [Security & Compliance Guide](security.md).

## What you will be able to do

- Run a local demo with synthetic data and pre-created accounts.
- Log in and explore the dashboard, attendance session, and reports screens.
- Understand how the face recognition workflow operates.

## Step-by-step instructions

### 1. Install Python

Download and install Python 3.12 or newer from [python.org](https://www.python.org/downloads/). During installation on Windows, check "Add Python to PATH".

### 2. Download this repository

Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run:

```bash
git clone https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition.git
cd Attendance-Management-System-Using-Face-Recognition
```

> **Tip:** If you do not have Git installed, download the repository as a ZIP from GitHub and extract it.

### 3. Create a virtual environment (isolates project dependencies)

```bash
python -m venv venv
```

This creates a folder called `venv` that keeps project libraries separate from your system.

### 4. Activate the virtual environment

- **macOS / Linux:**

  ```bash
  source venv/bin/activate
  ```

- **Windows (Command Prompt):**

  ```cmd
  venv\Scripts\activate
  ```

- **Windows (PowerShell):**

  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

You should see `(venv)` at the start of your terminal prompt.

### 5. Install project dependencies

```bash
pip install -r requirements.txt
```

This downloads all the libraries the project needs. It may take a few minutes.

### 6. Set up environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

> On Windows Command Prompt, use `copy .env.example .env` instead.

The default values in `.env.example` are pre-configured for local demos.

### 7. Bootstrap the demo (creates database, sample data, and demo accounts)

```bash
make demo
```

> **Windows note:** If `make` is not available, run these commands instead:
>
> ```bash
> python manage.py migrate --noinput
> python scripts/bootstrap_demo.py
> ```

This command:

- Creates the database tables.
- Generates synthetic face images for three demo users.
- Creates a demo admin account and three demo employee accounts.

### 8. Start the development server

```bash
python manage.py runserver
```

You should see output like:

```text
Starting development server at http://127.0.0.1:8000/
```

### 9. Open the app in your browser

Navigate to <http://127.0.0.1:8000/> in your web browser.

## Demo login credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | `demo_admin` | `demo_admin_pass` |
| Employee 1 | `user_001` | `demo_user_pass` |
| Employee 2 | `user_002` | `demo_user_pass` |
| Employee 3 | `user_003` | `demo_user_pass` |

## What you should see

1. **Home page** – The landing page with "Mark Time-In", "Mark Time-Out", and "Dashboard Login" cards.
2. **Login page** – Click "Dashboard Login" and enter the demo admin credentials above.
3. **Admin dashboard** – A first-run checklist (if no employees are registered) and action cards for registering employees, adding photos, viewing reports, and more.
4. **Attendance session** – Navigate to "Attendance Session" to see the live recognition feed and recent attendance logs.
5. **Reports** – Click "View Attendance" to see attendance records by date or employee.

## Stopping the demo

- Press `Ctrl+C` in the terminal to stop the development server.
- To deactivate the virtual environment, run: `deactivate`

## Next steps

- **Explore further:** Try registering a new employee and adding photos via the webcam.
- **Read the User Guide:** See [USER_GUIDE.md](USER_GUIDE.md) for detailed instructions on all features.
- **Production deployment:** When ready to deploy for real use, follow the [Deployment Guide](DEPLOYMENT.md) and [Security & Compliance Guide](security.md).
