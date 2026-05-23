## 2026-04-25 - [Add composite index on created_at and successful]
**Learning:** Adding a composite index on frequently filtered fields can significantly improve query performance for API endpoints.
**Action:** Added a composite index on `created_at` and `successful` in the `RecognitionAttempt` model to optimize the attendance stats endpoint query.
## Optimizations
* Replaced `if present_qs.exists():` with `if present_qs:` in `recognition/views_legacy.py` (`view_attendance_date`, `view_attendance_employee`, `view_my_attendance_employee_login`). This avoids a redundant database query since the queryset is evaluated right after in the charting functions.
