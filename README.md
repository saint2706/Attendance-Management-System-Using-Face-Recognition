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

### Configuration

Before running the project you should configure the following environment variables:

- `DJANGO_DEBUG` (default: `False`): enables Django's debug mode for local development.
  Set it to `True` when you want verbose error pages and auto-reload behaviour.
- `DJANGO_ALLOWED_HOSTS`: a comma-separated list of hostnames that Django is allowed to
  serve. This value is required when `DJANGO_DEBUG` is `False`.
- `DJANGO_SECRET_KEY`: a cryptographically secure secret key. This must be set in
  production deployments. The built-in development key is only accepted when
  `DJANGO_DEBUG` is explicitly enabled.
- `RECOGNITION_DISTANCE_THRESHOLD` (default: `0.4`): sets the maximum
  allowable embedding distance returned by DeepFace when marking attendance.
  Lower values make recognition stricter, while higher values are more
  permissive. Configure it via an environment variable before starting the
  server if you need to tune recognition sensitivity.

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

---

## Complete Walkthrough for Non-Programmers 🧭

The following guide is written for readers with no coding background. It explains every concept that appears in the project and walks you through the entire journey—from preparing data to interpreting the results of face recognition. Take your time with each step; you can always refer back as you experiment with the system.

### 1. Understanding the Project at a High Level

- **Goal:** Automatically record who is present (and when) by using a webcam to recognize employee faces.
- **Why Face Recognition?** Every human face is unique. Computers can translate a face into a set of numbers that act like a fingerprint. When the same face reappears, those numbers look almost identical, allowing the system to identify the person.

### 2. Core Concepts Explained in Plain Language

| Concept | What It Means | How It Is Used Here |
| --- | --- | --- |
| **Pixel** | The smallest dot in a digital image. Faces in photos are made of thousands of pixels. | Photos captured from the webcam are just grids of pixels. |
| **Vector** | A list of numbers. Think of it like a spreadsheet row. | Each face is converted into a vector (often 128 numbers long). |
| **Embedding** | A vector that summarizes the important traits of something—in our case, a face. | The DeepFace library turns each face image into an embedding. |
| **Distance Between Vectors** | A measure of how different two vectors are. | If two face embeddings are close enough, we decide they belong to the same person. |
| **Cosine Similarity** | A score from -1 to 1 that tells how similar two vectors are. 1 means identical, 0 means unrelated. | DeepFace uses cosine similarity to compare embeddings. |
| **Threshold** | A boundary that separates “match” from “no match.” | If similarity is above the threshold, the face is considered recognized. |
| **Dataset** | A collection of data points. | Here, a dataset is a folder full of face photos grouped by person. |
| **Train/Test Split** | Dividing data to learn (train) and to check performance (test). | The system evaluates recognition quality by keeping some photos hidden during training. |

### 3. The Complete Data Pipeline (Loading → Prediction)

1. **Loading Data**
   - Photos are stored in `face_recognition_data/`. Each subfolder is named after an employee and contains their images.
   - When you add a new employee and capture photos, these files are saved automatically.

2. **Cleaning Data**
   - Ensure every folder only contains clear photos of the correct person.
   - Delete blurry images or photos with other people in the frame, because they can confuse the model.

3. **Preparing Data**
   - Resize images so they all share the same dimensions (DeepFace handles this for you).
   - Convert images to numerical arrays: each pixel’s color value becomes a number.
   - Normalize the pixel values so that lighting differences do not overwhelm facial features.

4. **Training (Learning What Each Face Looks Like)**
   - DeepFace passes each image through the **Facenet** neural network to produce an embedding.
   - Facenet is a deep learning model that has already been trained on millions of faces. It uses layers of math (matrix multiplications, activations like ReLU, and batch normalization) to extract facial features.
   - For every employee, the system stores one or more embeddings representing that person.

5. **Testing (Checking the System’s Memory)**
   - Hold back a portion of photos for each employee.
   - Compare the embeddings of these held-out photos with the known embeddings.
   - If the similarity exceeds the threshold, the prediction is considered correct.

6. **Predicting in Real Time**
   - When a user stands in front of the webcam, the current frame is sent to DeepFace.
   - The system detects the face, generates an embedding, and compares it to stored embeddings.
   - If a match is found, the attendance record is saved with the current timestamp.

### 4. Understanding the Math Without Jargon

- **Matrix Multiplication:** Imagine stacking multiple filters over a photo to highlight patterns like edges or curves. Neural networks multiply matrices to apply these filters across the image efficiently.
- **Activation Functions (e.g., ReLU):** After filtering, the network decides which signals to keep. ReLU simply says “if the number is negative, replace it with zero,” helping the model focus on important features.
- **Normalization:** Keeps numbers within similar ranges so the network does not get distracted by large values.
- **Cosine Similarity Formula:**

  \[
  \text{similarity} = \frac{A \cdot B}{\|A\| \times \|B\|}
  \]

  - \(A \cdot B\) is the dot product—multiply each pair of numbers and sum them.
  - \(\|A\|\) and \(\|B\|\) are lengths of the vectors.
  - Values close to 1 mean the faces are very similar.

### 5. Metrics and How to Read Them

When you evaluate the system (for example, by running Django tests or custom evaluation scripts), you might see the following metrics:

- **Accuracy:** The percentage of predictions the system got right overall.
- **Precision:** Of all faces the system claimed belonged to a person, what fraction were correct? High precision means very few false positives (wrong matches).
- **Recall:** Of all the faces that truly belonged to a person, what fraction did the system recognize? High recall means it rarely misses someone who is present.
- **F1 Score:** The harmonic mean of precision and recall—useful when you need a single score that balances both concerns.

To compute these metrics, the system compares predicted labels (who the model thinks is present) against actual labels (the real person in the photo). A confusion matrix—a small table with true vs. predicted labels—helps you see where mistakes happen.

### 6. Step-by-Step Usage Guide

1. **Prepare Your Environment**
   - Follow the installation steps above.
   - Plug in a webcam and ensure your operating system detects it.

2. **Collect Employee Photos**
   - Use good lighting and ask employees to face the camera directly.
   - Capture multiple angles: front, slight left, slight right.

3. **Run the Server**
   - `python manage.py runserver`
   - Visit `http://127.0.0.1:8000/` in your browser.

4. **Register Employees**
   - Log in with the admin account and use the dashboard to create users.

5. **Capture Training Photos**
   - Use the “Add Photos” feature to record embeddings automatically.
   - Review the saved photos in `face_recognition_data/<username>/` and delete any poor-quality images.

6. **Verify the Model**
   - Optionally, run automated tests: `python manage.py test`.
   - Review accuracy, precision, recall, and F1 score outputs in the terminal to confirm the system performs as expected.

7. **Daily Operation**
   - Employees use the “Mark Time-In” or “Mark Time-Out” buttons.
   - The system recognizes their face and records the timestamp.

8. **Review Attendance Reports**
   - Admins can view summaries and export attendance logs from the dashboard.

### 7. Interpreting and Acting on Results

- **Confidence Scores:** If the system displays similarity scores, values closer to 1 mean stronger confidence. If many scores are borderline, collect more training photos.
- **False Matches:** If the system mistakes one person for another, raise the similarity threshold or improve photo quality.
- **Missed Recognitions:** If someone is not recognized, add more varied photos of that person and ensure consistent lighting.
- **Continuous Improvement:** Periodically retrain (by re-running the photo capture) to keep up with appearance changes like hairstyles or glasses.

### 8. Troubleshooting Checklist

- **Webcam Not Working?** Check operating system permissions and try another USB port.
- **Poor Lighting?** Use a lamp facing the person to reduce shadows.
- **Performance Issues?** Close other applications using the webcam or upgrade to a machine with more RAM/CPU power.
- **Data Backup:** Regularly copy `db.sqlite3` and the `face_recognition_data/` directory to an external drive.

With these explanations, you should be able to operate, evaluate, and improve the attendance system without writing any code.

