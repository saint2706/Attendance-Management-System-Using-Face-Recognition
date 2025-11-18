# User Guide

This guide provides a comprehensive walkthrough of the Smart Attendance System, designed for non-technical users. It covers everything from logging in to interpreting attendance reports.

## 1. Core Concepts

- **Face Recognition:** The system uses a webcam to identify employees by their unique facial features. Each face is converted into a secure digital signature (an "embedding"). When an employee clocks in or out, the system compares the new embedding to the stored ones to find a match.
- **Attendance Record:** A timestamped entry that confirms when an employee has checked in or out.
- **Dashboard:** The main screen after logging in, providing access to all features. Admins and employees see different dashboards tailored to their roles.

## 2. Step-by-Step Instructions

### Logging In

1.  Navigate to the system's home page.
2.  Click on the **Dashboard Login** button.
3.  Enter your username and password.
4.  Click **Login**.

### For Employees

#### Marking Attendance

1.  From the home page, click **Mark Time-In** to clock in for the day.
2.  Position your face in front of the webcam. The system will automatically recognize you and record your check-in time.
3.  At the end of the day, click **Mark Time-Out** to clock out.

#### Viewing Your Attendance

1.  Log in to your dashboard.
2.  Select a date range to view your attendance history.
3.  The system will display a table with your check-in and check-out times, as well as the total hours worked for each day.

### For Admins

#### Registering a New Employee

1.  Log in to your admin dashboard.
2.  Click on **Register Employee**.
3.  Fill in the new employee's details and click **Register**.
    [NEW SCREENSHOT: Employee Registration Form]

#### Adding Employee Photos

1.  From the admin dashboard, click on **Add Photos**.
2.  Enter the username of the employee and click **Add Photos**.
3.  The system will automatically capture a set of images to create a face profile for the employee. Ensure the employee is in a well-lit area and facing the camera.

#### Viewing Attendance Reports

Admins have access to several attendance reports:

- **Attendance by Date:** View a list of all employees who were present on a specific date, along with their work hours.
- **Attendance by Employee:** View the complete attendance history for a single employee over a selected date range.

## 3. Interpreting the Results

The attendance reports provide the following information:

- **Time In:** The first recorded check-in time for the day.
- **Time Out:** The last recorded check-out time for the day.
- **Hours:** The total hours worked for the day.
- **Break Hours:** The total time spent on breaks.

The system also generates graphs to help you visualize attendance trends.

## 4. Troubleshooting

- **Recognition Issues:** If the system is having trouble recognizing an employee, ensure they are in a well-lit area and facing the camera directly. If the problem persists, an admin may need to recapture the employee's photos.
- **Incorrect Timestamps:** If you notice any incorrect timestamps in your attendance records, please contact an administrator to have them corrected.

## 5. Model Evaluation Reports

Administrators who need to verify recognition quality can generate a full evaluation report from the command line:

```bash
python manage.py eval --split-csv reports/splits.csv
```

The command reuses the live recognition engine to process the encrypted image dataset and stores its findings in `reports/evaluation/`:

- `metrics_summary.json` – accuracy, precision, recall, macro F1, False Acceptance Rate (FAR), and False Rejection Rate (FRR).
- `confusion_matrix.csv` / `.png` – shows which identities are confused most often and whether "Unknown" predictions occur frequently.
- `threshold_sweep.csv` / `.png` – plots how FAR, FRR, accuracy, and F1 change as you tighten or loosen the distance threshold.

Use these reports to fine-tune thresholds before rolling changes into production or to document the system's performance for compliance reviews.
