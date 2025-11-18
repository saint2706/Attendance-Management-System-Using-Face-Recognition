# Data Card: Project Data Models

This document describes the data models used in the Smart Attendance System, as defined in the Django application.

## 1. `users.Present`

Represents the daily attendance status of an Employee.

| Field     | Type        | Description                                                                                             |
|-----------|-------------|---------------------------------------------------------------------------------------------------------|
| `user`    | ForeignKey  | A reference to the `auth.User` model, linking the attendance record to a specific Employee.              |
| `date`    | DateField   | The date of the attendance record.                                                                      |
| `present` | BooleanField| `True` if the Employee was present on this date, `False` otherwise.                                     |

**Implicit Relationships:**

-   This model is the primary record for daily attendance and is created or updated when an Employee successfully checks in for the first time on a given day.

## 2. `users.Time`

Records a specific time-in or time-out event for an Employee.

| Field  | Type         | Description                                                                                             |
|--------|--------------|---------------------------------------------------------------------------------------------------------|
| `user` | ForeignKey   | A reference to the `auth.User` model.                                                                   |
| `date` | DateField    | The date of the time entry.                                                                             |
| `time` | DateTimeField| The exact time of the event.                                                                            |
| `out`  | BooleanField | `False` for a time-in event, `True` for a time-out event.                                                 |

**Implicit Relationships:**

-   This model has a conceptual relationship with `users.Present`. Multiple `Time` records can exist for a single `Present` record, representing multiple check-ins and check-outs on the same day.

## 3. `users.RecognitionAttempt`

Persists metadata for each face recognition attempt.

| Field            | Type         | Description                                                                                             |
|------------------|--------------|---------------------------------------------------------------------------------------------------------|
| `user`           | ForeignKey   | A reference to the `auth.User` model, if the user was successfully identified.                          |
| `username`       | CharField    | The username inferred at the time of recognition.                                                       |
| `direction`      | CharField    | `"in"` for a check-in attempt, `"out"` for a check-out attempt.                                         |
| `site`           | CharField    | The physical location or site identifier for the attempt.                                               |
| `source`         | CharField    | The system component that initiated the attempt (e.g., `webcam`, `api`).                                  |
| `successful`     | BooleanField | `True` if the attempt resulted in a successful recognition.                                             |
| `spoof_detected` | BooleanField | `True` if the attempt was blocked by the anti-spoofing mechanism.                                       |
| `latency_ms`     | FloatField   | The end-to-end latency of the recognition attempt in milliseconds.                                      |
| `present_record` | ForeignKey   | A reference to the `users.Present` record created or updated by this attempt.                           |
| `time_record`    | ForeignKey   | A reference to the `users.Time` record created by this attempt.                                         |
| `error_message`  | TextField    | A description of the error if the attempt was unsuccessful.                                             |

**Implicit Relationships:**

-   A successful `RecognitionAttempt` will typically result in the creation of a `Time` record and the creation or update of a `Present` record.

## 4. `recognition.RecognitionOutcome`

Persists a snapshot of a recognition decision made during attendance flows.

| Field       | Type         | Description                                                                                             |
|-------------|--------------|---------------------------------------------------------------------------------------------------------|
| `created_at`| DateTimeField| The timestamp of the recognition decision.                                                              |
| `username`  | CharField    | The username associated with the recognition attempt.                                                   |
| `direction` | CharField    | `"in"` or `"out"`.                                                                                      |
| `source`    | CharField    | The source of the recognition attempt (e.g., `webcam`, `api`).                                          |
| `accepted`  | BooleanField | `True` if the recognition was accepted, `False` otherwise.                                              |
| `confidence`| FloatField   | The confidence score of the recognition.                                                                |
| `distance`  | FloatField   | The distance between the face embedding and the matched embedding.                                      |
| `threshold` | FloatField   | The distance threshold used for the recognition.                                                        |

**Implicit Relationships:**

-   This model is used for analytics and is not directly linked to the other attendance models via foreign keys. It provides a historical record of recognition decisions.

## Liveness Signals and Limitations

-   The motion-based liveness buffer runs entirely in memory during each recognition attempt; no additional biometric data is stored beyond the existing encrypted training images.
-   The detector looks for subtle parallax (blinks, head turns, breathing) across a three-to-five-frame window. Extremely static lighting or perfectly stabilized video replays can therefore lower the score or slip through if the DeepFace anti-spoofing model also agrees.
-   Operators should document any high-risk deployments (e.g., unattended kiosks) and consider pairing this check with hardware sensors or on-device challenge/response if attackers can present high-quality screens within a few centimetres of the camera.

## Sample Dataset for Reproducibility

-   A `sample_data/` directory now ships with the repository. It mirrors the `face_recognition_data/training_dataset/` layout and contains three procedurally generated, non-identifiable JPEG avatars per identity.
-   The helper script `scripts/reproduce_sample_results.py` temporarily points the evaluation harness at this directory so reviewers can regenerate metrics with `make reproduce` without handling encrypted production photos.
-   The sample dataset is strictly for demos and smoke tests. Replace it with the encrypted `face_recognition_data/` tree before operating in production so the evaluation pipeline reflects the real enrollment set.
