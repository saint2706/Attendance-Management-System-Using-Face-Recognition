# Architecture Overview

This document provides a high-level overview of the application's architecture, explaining how the different components work together to provide a seamless and automated way to track employee attendance.

## 1. Core Components

The application is built on the following core components:

-   **Django**: The web framework that provides the application's structure, including the models, views, and templates.
-   **DeepFace**: A lightweight face recognition and facial attribute analysis (age, gender, emotion, and race) library for Python. It is used to power the face recognition functionality of the application.
-   **SQLite**: The default database for the application. It is used to store the application's data, including the user accounts and attendance records.
-   **Bootstrap**: The front-end framework that is used to create the application's responsive web interface.

## 2. Architecture Diagram

The following diagram provides a high-level overview of the application's architecture:

```
+-----------------+      +-----------------+      +-----------------+
|   Web Browser   | <--> |     Django      | <--> |     SQLite      |
+-----------------+      +-----------------+      +-----------------+
                           |
                           v
+-----------------+      +-----------------+
|   DeepFace      | <--> | Face Reco Data  |
+-----------------+      +-----------------+
```

## 3. Workflow

The application's workflow is as follows:

1.  **User Registration**: An admin registers a new employee by creating a new user account for them.
2.  **Add Photos**: The admin adds photos for the employee by using the "Add Photos" feature. The application uses the webcam to capture a set of photos for face recognition and stores them in the `face_recognition_data` directory.
3.  **Mark Attendance**: The employee marks their attendance by using the "Mark Time-In" or "Mark Time-Out" feature. The application uses the webcam to capture a frame and sends it to DeepFace for face recognition.
4.  **Face Recognition**: DeepFace compares the captured frame to the stored face images and returns the name of the recognized employee.
5.  **Store Attendance**: The application stores the attendance record in the database.
6.  **View Attendance**: The admin can view the attendance records for all employees, and the employee can view their own attendance records.
