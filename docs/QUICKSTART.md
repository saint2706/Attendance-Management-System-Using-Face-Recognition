# Quick Start Guide

Get the Attendance Management System running locally in **5 minutes** with sample data and demo accounts. No prior experience with Python, Django, or Docker required—just follow the steps.

> **⚠️ For Evaluation Only**: This setup uses synthetic data and development settings. For production deployment, see the [Deployment Guide](DEPLOYMENT.md).

## What You'll Get

After completing this guide, you'll have:

- ✅ A fully functional local instance running at `http://localhost:8000`
- ✅ Pre-created admin and employee demo accounts
- ✅ Synthetic face data for testing recognition
- ✅ Access to all features: dashboard, attendance tracking, reports

**Time Required**: ~5 minutes (plus dependency install time)

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.12 or newer** ([Download here](https://www.python.org/downloads/))
- **Git** (optional, for cloning; alternatively download ZIP)
- **A webcam** (optional, for testing live face recognition)
- **10 GB free disk space** (for dependencies and sample data)

> **Windows Users**: During Python installation, check **"Add Python to PATH"** before clicking Install.

---

## Quick Start: 5 Steps

### Step 1: Get the Code

**Option A: Clone with Git** (recommended)
```bash
git clone https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition.git
cd Attendance-Management-System-Using-Face-Recognition
```

**Option B: Download ZIP**
1. Go to [https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition)
2. Click **Code** → **Download ZIP**
3. Extract the ZIP file
4. Open a terminal in the extracted folder

---

### Step 2: Create a Virtual Environment

A virtual environment isolates this project's dependencies from your system Python.

```bash
python -m venv venv
```

**Activate the environment:**

| Operating System | Command |
|------------------|---------|
| **macOS / Linux** | `source venv/bin/activate` |
| **Windows CMD** | `venv\Scripts\activate` |
| **Windows PowerShell** | `.\venv\Scripts\Activate.ps1` |

✅ **Success Indicator**: Your terminal prompt should now show `(venv)` at the beginning.

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**What's Happening?**
- Installs Django, DeepFace, TensorFlow, and 40+ other dependencies
- Downloads machine learning models (~500MB)
- Takes 3-5 minutes on a typical connection

⚠️ **Common Issues**:
- **"pip not found"**: Ensure Python is in PATH, or try `python -m pip install -r requirements.txt`
- **TensorFlow installation fails**: You may need to install Visual C++ Redistributable (Windows) or build-essential (Linux)

---

### Step 4: Configure Environment Variables

Copy the example configuration file:

```bash
# macOS / Linux / Git Bash
cp .env.example .env

# Windows CMD
copy .env.example .env
```

The `.env.example` file contains safe defaults for local development. **No changes needed** for quick start.

> **For Production**: See [Configuration Guide](CONFIGURATION.md) for required environment variables.

---

### Step 5: Bootstrap Demo Data

Run the automated demo setup:

```bash
make demo
```

**Alternative (if `make` is not installed):**
```bash
python manage.py migrate --noinput
python scripts/bootstrap_demo.py
```

**What This Does:**
1. Creates SQLite database and runs migrations
2. Generates synthetic face images for demo users
3. Creates these accounts:
   - 1 admin account (`demo_admin`)
   - 3 employee accounts (`user_001`, `user_002`, `user_003`)

**Output**: You'll see confirmation messages. The process takes ~30 seconds.

---

### Step 6: Start the Server

```bash
python manage.py runserver
```

**Expected Output:**
```
System check identified no issues (0 silenced).
December 22, 2025 - 12:00:00
Django version 6.0, using settings 'attendance_system_facial_recognition.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

---

### Step 7: Access the System

Open your web browser and navigate to:

```
http://127.0.0.1:8000/
```

or

```
http://localhost:8000/
```

✅ **Success!** You should see the home page with three main cards:
- **Mark Time-In** - Employee check-in
- **Mark Time-Out** - Employee check-out  
- **Dashboard Login** - Admin access

---

## Demo Credentials

Use these accounts to explore the system:

| Role | Username | Password | Access Level |
|------|----------|----------|--------------|
| **Admin** | `demo_admin` | `demo_admin_pass` | Full system access |
| **Employee** | `user_001` | `demo_user_pass` | Self-service only |
| **Employee** | `user_002` | `demo_user_pass` | Self-service only |
| **Employee** | `user_003` | `demo_user_pass` | Self-service only |

---

## Exploring the System

### As an Admin

1. **Click "Dashboard Login"** on the home page
2. **Enter admin credentials** (demo_admin / demo_admin_pass)
3. **Explore the dashboard:**
   - View real-time attendance status
   - Check system health metrics
   - Access reports and analytics
4. **Try these features:**
   - **Employees** → View the three demo employees
   - **Attendance Records** → See pre-populated check-ins
   - **Reports** → Generate attendance summaries
   - **System Settings** → Configure thresholds and features

### As an Employee

1. **Click "Mark Time-In"** on the home page
2. **Allow camera access** when prompted
3. **Position your face** in the camera frame
4. The system will attempt to match your face against demo accounts
   - ⚠️ Since you're not in the system, recognition will fail (expected behavior)
   - To test successful recognition, add your own photos via admin panel

**Testing with Demo Data:**
- The system includes synthetic face images for demo users
- These synthetic images are used for internal testing only
- For real face recognition testing, add actual photos via admin panel

### Testing Face Recognition

To test the recognition system with your own face:

1. **Log in as admin** (demo_admin)
2. **Navigate to "Employees"** → Select a demo employee
3. **Click "Add Photos"** and grant camera access
4. **Capture 5-10 photos** from different angles
5. **Wait for training** (happens automatically in background)
6. **Test recognition** by clicking "Mark Time-In" on home page

---

## Stopping the Demo

**Stop the server:**
```bash
# Press Ctrl+C in the terminal running the server
```

**Deactivate virtual environment:**
```bash
deactivate
```

**Clean up (optional):**
```bash
# Remove database and generated data
rm db.sqlite3
rm -rf face_recognition_data/
rm -rf media/
```

---

## What's Next?

### Learn More
- **[User Guide](USER_GUIDE.md)** - Complete feature walkthrough
- **[Admin Guide](ADMIN_GUIDE.md)** - User management and configuration
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

### Develop or Extend
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Local development setup
- **[Architecture](ARCHITECTURE.md)** - System design and components
- **[API Reference](API_REFERENCE.md)** - REST endpoints and CLI tools

### Deploy to Production
- **[Deployment Guide](DEPLOYMENT.md)** - Docker and Kubernetes setup
- **[Security Guide](SECURITY.md)** - Hardening and best practices
- **[Configuration](CONFIGURATION.md)** - Environment variables reference

---

## Troubleshooting Quick Start

### Issue: "pip: command not found"

**Solution:**
```bash
python -m pip install -r requirements.txt
```

### Issue: "No module named 'django'"

**Cause**: Virtual environment not activated or dependencies not installed.

**Solution:**
```bash
# Ensure virtual environment is activated (you should see "(venv)" in prompt)
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: TensorFlow installation fails

**Windows Solution:**
1. Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Retry: `pip install -r requirements.txt`

**Linux Solution:**
```bash
sudo apt-get update
sudo apt-get install -y python3-dev build-essential
pip install -r requirements.txt
```

### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Kill existing process using port 8000
# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use a different port:
python manage.py runserver 8001
```

### Issue: Camera not detected

**Solution:**
- Check browser permissions (should show camera icon in address bar)
- Try a different browser (Chrome/Edge recommended)
- Ensure no other application is using the camera
- Restart the browser and try again

### More Help

For additional troubleshooting, see:
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**
- **[FAQ](FAQ.md)**
- **[GitHub Issues](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/issues)**

---

## Quick Start Checklist

✅ Installed Python 3.12+  
✅ Cloned repository  
✅ Created and activated virtual environment  
✅ Installed dependencies  
✅ Copied .env.example to .env  
✅ Ran `make demo` to bootstrap  
✅ Started server with `python manage.py runserver`  
✅ Accessed http://localhost:8000  
✅ Logged in with demo credentials  
✅ Explored the dashboard and features  

**Congratulations!** You've successfully set up the Attendance Management System.

---

*Last Updated: December 2025 | Quick Start Version: 2.0*
