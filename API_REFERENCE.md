# API Reference

This document provides a reference for the API endpoints available in the Attendance Management System.

## Face Recognition API

The Face Recognition API is now fully asynchronous.

### Enqueue Face Recognition Task

-   **Endpoint:** `/mark_attendance/` (for check-in) or `/mark_attendance_out/` (for check-out)
-   **Method:** `POST`
-   **Description:** Enqueues a face recognition task.
-   **Request Body:**
    -   A multipart form data request with an `image` field containing the image file.
-   **Response:**
    -   **Status Code:** `202 Accepted`
    -   **Body:**
        ```json
        {
            "task_id": "your-task-id"
        }
        ```

### Get Task Status

-   **Endpoint:** `/task_status/<task_id>/`
-   **Method:** `GET`
-   **Description:** Retrieves the status of a face recognition task.
-   **URL Parameters:**
    -   `task_id`: The ID of the task returned by the enqueue endpoint.
-   **Response:**
    -   **Status Code:** `200 OK`
    -   **Body:**
        ```json
        {
            "task_id": "your-task-id",
            "status": "SUCCESS", // or "PENDING", "FAILURE"
            "result": {
                "status": "success",
                "username": "recognized_username"
            }
        }
        ```
