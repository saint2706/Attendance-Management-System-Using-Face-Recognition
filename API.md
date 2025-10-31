# API Documentation

This document outlines all the URL patterns for the project and explains the purpose and functionality of each corresponding view.

## Core Pages

| URL Path      | View Function        | Name                 | Description                                                                                                                                                             |
|---------------|----------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `/`           | `recog_views.home`   | `home`               | Renders the home page of the application.                                                                                                                               |
| `/dashboard/` | `recog_views.dashboard`| `dashboard`          | Renders the dashboard, which differs for admins and regular employees.                                                                                                    |

## Authentication

| URL Path   | View Function/Class             | Name    | Description                                                                 |
|------------|---------------------------------|---------|-----------------------------------------------------------------------------|
| `/login/`  | `auth_views.LoginView`          | `login` | Displays the login page and handles user authentication.                      |
| `/logout/` | `auth_views.LogoutView`         | `logout`| Logs the user out and redirects them to the home page.                        |

## User and Photo Management (Admin-only)

| URL Path        | View Function           | Name           | Description                                                                                                                                                    |
|-----------------|-------------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `/register/`    | `users_views.register`  | `register`     | Allows staff members or superusers to register new employee accounts.                                                                                           |
| `/add_photos/`  | `recog_views.add_photos`| `add-photos`   | Handles the 'Add Photos' functionality for admins to create face datasets for users.                                                                           |
| `/train/`       | `recog_views.train`     | `train`        | **Obsolete.** This view is no longer used, as the training process is now automatic. It redirects to the dashboard with an informational message.                  |

## Face Recognition and Attendance Marking

| URL Path                      | View Function                      | Name                         | Description                                            |
|-------------------------------|------------------------------------|------------------------------|--------------------------------------------------------|
| `/mark_your_attendance`       | `recog_views.mark_your_attendance` | `mark-your-attendance`       | Handles marking time-in using face recognition.        |
| `/mark_your_attendance_out`   | `recog_views.mark_your_attendance_out` | `mark-your-attendance-out`   | Handles marking time-out using face recognition.       |

## Attendance Viewing

| URL Path                     | View Function                                 | Name                                  | Description                                                                                                                                         |
|------------------------------|-----------------------------------------------|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `/view_attendance_home`      | `recog_views.view_attendance_home`            | `view-attendance-home`                | Renders the main attendance viewing page for admins.                                                                                                |
| `/view_attendance_date`      | `recog_views.view_attendance_date`            | `view-attendance-date`                | Admin view to see attendance for all employees on a specific date.                                                                                   |
| `/view_attendance_employee`  | `recog_views.view_attendance_employee`        | `view-attendance-employee`            | Admin view to see attendance for a specific employee over a date range.                                                                             |
| `/view_my_attendance`        | `recog_views.view_my_attendance_employee_login` | `view-my-attendance-employee-login`   | Employee-specific view to see their own attendance over a date range.                                                                               |

## Error/Status Pages

| URL Path           | View Function              | Name                 | Description                                                       |
|--------------------|----------------------------|----------------------|-------------------------------------------------------------------|
| `/not_authorised`  | `recog_views.not_authorised` | `not-authorised`     | Renders a page for users trying to access unauthorized areas.     |
