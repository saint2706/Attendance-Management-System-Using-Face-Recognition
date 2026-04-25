## 2026-04-25 - [Add composite index on created_at and successful]
**Learning:** Adding a composite index on frequently filtered fields can significantly improve query performance for API endpoints.
**Action:** Added a composite index on `created_at` and `successful` in the `RecognitionAttempt` model to optimize the attendance stats endpoint query.
