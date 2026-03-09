# The Ultimate "Idiot-Proof" Guide: Face Recognition Attendance System

Welcome! If you're reading this, you want to set up an automated facial recognition attendance system, but you might not be a computer expert. **That is completely okay!** This guide is designed for *anyone* to follow. We will hold your hand through every single step, from turning on your computer to having a fully working system.

We assume nothing. If a step says "open a terminal," we will tell you *how* to open a terminal.

---

## 📖 Part 1: What is this project? (The Idiot-Proof Summary)

Imagine a very smart digital bouncer standing at the door of your office or classroom.
1. **The Bouncer Learns Faces:** First, you show the bouncer pictures of everyone who is allowed inside (your employees or students).
2. **The Bouncer Watches the Door:** You place a camera at the entrance. The bouncer watches the camera feed.
3. **The Bouncer Takes Notes:** When someone walks in, the bouncer looks at their face, remembers who they are, and writes down the exact time they arrived. It also checks if they are a real person (and not someone holding up a photo on a phone).
4. **The Bouncer Gives You a Report:** At the end of the day, the bouncer hands you a clean, organized spreadsheet saying exactly who showed up and when.

**This software is that digital bouncer.**

It runs on a computer, watches a camera, and automatically marks attendance. It works on laptops, tablets, classroom cameras, or office entry kiosks.

---

## 🛠️ Part 2: Installing Prerequisites (The Boring But Necessary Stuff)

Before we build the house, we need tools. This software requires three main tools to run:
1. **Git:** A tool to download the project's code from the internet.
2. **Docker:** A tool that packages all the complicated software pieces into one neat box that "just works" on any computer.
3. **Python & Node.js:** (Only needed if you want to modify the code yourself. We will cover this later, but for now, Docker is all you need to just *use* it).

We highly recommend using **Docker** because it does 95% of the hard work for you.

### 🍎 For Mac Users (macOS)

**Step 1: Open the Terminal**
1. Press `Command (⌘) + Spacebar` on your keyboard to open Spotlight Search.
2. Type `Terminal` and press `Enter`. A small black or white window with text will open.

**Step 2: Install Homebrew (A tool that installs other tools)**
1. Copy this exact line of text:
   `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Paste it into your Terminal and press `Enter`.
3. It will ask for your Mac password. When you type your password, **nothing will show up on the screen** (no stars, no dots). This is normal! Just type it and press `Enter`.
4. Press `Enter` again when it asks to confirm. Wait until it finishes (it might take a few minutes).

**Step 3: Install Git and Docker**
1. Copy and paste this command into the Terminal, then press `Enter`:
   `brew install git`
2. Copy and paste this command, then press `Enter`:
   `brew install --cask docker`
3. Wait for it to finish.

**Step 4: Start Docker**
1. Press `Command (⌘) + Spacebar`, type `Docker`, and press `Enter`.
2. A whale icon will appear at the very top right of your screen (near the clock). Click it and wait until it says "Docker Desktop is running".

### 🪟 For Windows Users

**Step 1: Download Git**
1. Open your web browser (Chrome, Edge, Firefox) and go to: `https://git-scm.com/download/win`
2. Click "Click here to download" (or download the 64-bit Git for Windows Setup).
3. Open the downloaded file (`Git-XXX.exe`).
4. Click "Next" on every single screen until the installation finishes. You don't need to change any settings.

**Step 2: Download Docker Desktop**
1. Open your web browser and go to: `https://docs.docker.com/desktop/install/windows-install/`
2. Click the big blue "Docker Desktop for Windows" button.
3. Open the downloaded file (`Docker Desktop Installer.exe`).
4. Make sure "Use WSL 2 instead of Hyper-V" is checked (if it asks). Click "Ok" or "Next" until it finishes.
5. It might ask you to restart your computer. If it does, restart it.

**Step 3: Start Docker**
1. Click the Windows Start Menu button.
2. Type `Docker Desktop` and click it to open.
3. Accept the terms if a box pops up. Wait until the window says Docker is running (you might see a green bar or a whale icon in the bottom right corner of your screen).

### 🐧 For Linux (Ubuntu) Users

**Step 1: Open the Terminal**
1. Press `Ctrl + Alt + T` on your keyboard.

**Step 2: Install Git**
1. Type `sudo apt update` and press `Enter`. Type your password (it won't show on screen) and press `Enter`.
2. Type `sudo apt install git -y` and press `Enter`.

**Step 3: Install Docker**
1. Type `sudo apt install docker.io -y` and press `Enter`.
2. Type `sudo apt install docker-compose -y` and press `Enter`.
3. (Optional but recommended) Type `sudo usermod -aG docker $USER` and press `Enter`. This lets you use Docker without typing `sudo` every time. **You must restart your computer for this to take effect.**

---

## 🏗️ Part 3: Getting the Code and Setting Up the Environment

Now that your tools are ready, we will download the project.

### Step 1: Clone the Project (Copying the files to your computer)
1. Open your terminal (Mac/Linux) or Command Prompt/PowerShell (Windows).
2. Type this exact command and press `Enter`:
   `git clone https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition.git`
3. The computer will download the files. Wait until it says "done".
4. Type `cd Attendance-Management-System-Using-Face-Recognition` and press `Enter`. This moves you *inside* the project folder.

### Step 2: The Secret Keys (Environment Variables)

This project needs some secret passwords to protect the face data. We store these in a hidden file called `.env` (pronounced "dot-e-n-v"). **You must create this file.**

**How to create the `.env` file:**
1. In your terminal, type `cp .env.example .env` (Mac/Linux) or `copy .env.example .env` (Windows) and press `Enter`.
2. This makes a copy of the example settings and names it `.env`.

**What goes inside the `.env` file?**
You need to open this `.env` file in a text editor (like Notepad on Windows or TextEdit on Mac). Here is an *idiot-proof* example of what the file should look like for a basic setup:

```env
# ----------------------------------------------------------------------
# 🤫 THE IDIOT-PROOF .ENV FILE (COPY AND PASTE THIS IF YOU ARE STUCK)
# ----------------------------------------------------------------------

# 1. The main secret key for the website (Do not share this!)
DJANGO_SECRET_KEY='a-very-long-random-string-like-this-one-here-change-it-later'

# 2. These two keys MUST be exactly 32-bytes (44 characters) ending in '='.
# They lock up the pictures of faces so hackers can't see them.
# DO NOT CHANGE THESE unless you know how to generate a Fernet key!
DATA_ENCRYPTION_KEY='ufkljjgdbIMsc4N4-cVeRTtBk8sM6rDl6q-FMpepe8g='
FACE_DATA_ENCRYPTION_KEY='ufkljjgdbIMsc4N4-cVeRTtBk8sM6rDl6q-FMpepe8g='

# 3. Tells the system we are just testing it out, not running a big company server.
DEBUG=1

# 4. Where the database is. This connects everything together.
DATABASE_URL='sqlite:///db.sqlite3'
```

**CRITICAL NOTE FOR EVERYONE:** The `DATA_ENCRYPTION_KEY` and `FACE_DATA_ENCRYPTION_KEY` **must** be formatted perfectly (like the gibberish above ending in an `=` sign). If you just type "password123", the whole system will crash immediately. Just copy the example above.

### Step 3: Setting Up Native Local Development (Only if you are a programmer)

*If you are not a programmer and just want to run the app, SKIP TO PART 4 (Docker).*

If you want to edit the code, you need Python and Node.js installed.

1. **Install Python and Node:**
   - Download Python (3.12+) from `python.org/downloads`. (Windows users: Check the box "Add Python to PATH" during install!)
   - Download Node.js from `nodejs.org` (LTS version).
   - Download pnpm by running `npm install -g pnpm` in your terminal.

2. **Setup the Backend (Python):**
   - In your terminal, make sure you are in the project folder.
   - Type `python -m venv venv` and press `Enter`. (This creates an isolated "sandbox" for Python).
   - Mac/Linux: Type `source venv/bin/activate` and press `Enter`.
   - Windows: Type `venv\Scripts\activate` and press `Enter`.
   - Your terminal prompt will now have `(venv)` at the beginning. This means you are inside the sandbox.
   - Type `pip install -r requirements.txt` and press `Enter`. (This installs the required Python packages).
   - Type `python manage.py migrate` and press `Enter`. (This creates the database tables).
   - Type `python manage.py runserver` and press `Enter`. (This starts the backend API on port 8000).

3. **Setup the Frontend (React):**
   - Open a **second, new terminal window** (don't close the first one!).
   - Navigate back to the project folder: `cd Attendance-Management-System-Using-Face-Recognition`
   - Type `cd frontend` and press `Enter`.
   - Type `pnpm install` and press `Enter`. (This downloads all the web packages. Wait for it to finish).
   - Type `pnpm run dev` and press `Enter`. (This starts the website on port 5173).

4. Open your web browser and go to `http://localhost:5173`. You should see the login screen!

---

## 🚀 Part 4: Deployment (The easiest way to run everything)

Unless you are actively changing code, you should use Docker. Docker bundles everything (the Python backend, the React frontend, the database, the face recognition AI) into one simple command.

### 🐳 The "Just Make It Work" Docker Setup (Highly Recommended)

If you followed Part 2 and installed Docker, this takes 2 minutes.

1. Open your terminal (Mac/Linux) or Command Prompt (Windows).
2. Move into the project folder:
   `cd Attendance-Management-System-Using-Face-Recognition`
3. Make sure you created the `.env` file from Part 3! Docker needs this file to run.
4. Type this command and press `Enter`:
   `docker-compose up -d --build`
5. **WAIT.** The first time you run this, it will download gigabytes of data. It might take 10–20 minutes depending on your internet. Go get a coffee. ☕
6. When it says "Done" or "Started", open your web browser (Chrome or Edge).
7. Type `http://localhost:8000` in the address bar and press `Enter`.
8. You should see the login screen!

---

## 🖥️ Part 5: Hardware Scenarios (How to actually use it in real life)

You have the software running, but where do you put the camera? How do people use it? Here are 100% hand-holding guides for every scenario.

### Scenario A: The Laptop (The Easiest Way)
*Best for: Testing, small meetings, or pop-up events.*

**Setup Steps:**
1. Open your laptop and run the Docker command above (`docker-compose up -d`).
2. Open your web browser and go to `http://localhost:8000`.
3. Log in with your admin account (create one first if needed).
4. Click on the "Mark Attendance" or "Kiosk" page.
5. Your browser will ask: "Allow camera access?" **Click "Allow" or "Yes".**
6. Stand in front of your laptop. The screen will show your face and check you in. Done!

### Scenario B: The Classroom Camera Setup
*Best for: Teachers wanting to track students as they walk in.*

**Assumptions:** You are using a laptop/computer on the teacher's desk, connected to a USB webcam on a tripod.

**Setup Steps:**
1. Buy any generic 1080p USB webcam (like a Logitech C920) and a cheap camera tripod.
2. Place the tripod near the classroom door. Point the camera so it captures people's faces from the chest up as they walk in. Ensure the camera is *not* facing a bright window (backlighting will ruin the face recognition).
3. Plug the USB camera into your teacher laptop.
4. On your laptop, run the Docker command (`docker-compose up -d`).
5. Open your web browser and go to `http://localhost:8000`.
6. Open the "Kiosk" page. When it asks for camera access, make sure to select your **USB Webcam** (not the built-in laptop camera).
7. Students walk past the camera. The system will chime and mark them present.

### Scenario C: The Office Entry Kiosk (The Professional Setup)
*Best for: Permanent office entrances.*

**Assumptions:** You are using an iPad, Android Tablet, or a touchscreen monitor connected to a mini-PC (like an Intel NUC or Raspberry Pi) mounted on the wall.

**Setup Steps:**
1. **The Server:** You need one computer running the software (using the Docker command above). This computer can sit in a closet or under a desk. It must be connected to the office Wi-Fi. Let's say its IP address is `192.168.1.50`.
2. **The Kiosk:** Mount your tablet on the wall near the door. Connect it to the exact same Wi-Fi.
3. Open the web browser on the tablet (Safari or Chrome).
4. Type in the server's IP address and the port (e.g., `http://192.168.1.50:8000`).
5. Log in, open the "Kiosk" page.
6. **CRITICAL STEP FOR KIOSKS:** In our software, when the kiosk resets after checking someone in, *the focus is automatically returned to the "Capture" button*. This means if you plug a USB keyboard or a USB foot-pedal into the tablet, people can just press `Spacebar` or `Enter` to capture their face without touching the screen!
7. **Pro-Tip:** In Safari on iPad, tap the "Share" button and select "Add to Home Screen". This turns the website into a full-screen app, hiding the browser address bar so employees can't mess with it.

### Scenario D: The "Headless" Server (For IT Admins)
*Best for: Running the system in the cloud or on a server rack with no monitor attached.*

**Setup Steps:**
1. SSH into your server (e.g., `ssh user@your-server-ip`).
2. Install Docker using the Ubuntu instructions from Part 2.
3. Clone the repo and set up the `.env` file just like Part 3.
4. Run `docker-compose up -d --build`.
5. Because the server has no camera, you **must** use a client device (like a phone, tablet, or laptop) to actually capture faces.
6. Simply connect any device on the same network to `http://your-server-ip:8000` and open the Kiosk page. The *client device's camera* will be used, and the heavy lifting (face recognition) will be sent to your headless server!

---

## 🛑 Troubleshooting (Help, it broke!)

- **"The website won't load! It says localhost refused to connect."**
  - Did you run `docker-compose up -d`? Wait 2 minutes and try again. Docker might still be starting up.
- **"It says my face data is invalid or crashing!"**
  - Check your `.env` file. Does `DATA_ENCRYPTION_KEY` look EXACTLY like the gibberish we gave you? If not, fix it, then delete your database file (`db.sqlite3`) and restart.
- **"The camera feed is black!"**
  - Your web browser blocked the camera. Look for a tiny camera icon with a red 'X' in the address bar (top right). Click it and select "Always allow".
