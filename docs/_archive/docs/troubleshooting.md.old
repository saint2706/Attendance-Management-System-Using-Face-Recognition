# Operations Troubleshooting Guide

This guide provides actionable steps for diagnosing and resolving operational issues
encountered while running the Attendance Management System. It focuses on the
asynchronous attendance pipeline (Celery + Redis), the batch attendance API, and
fallback behaviours implemented in `recognition/views.py`.

## Celery or Redis Outage

### Symptoms
- Batch attendance submissions remain in a `PENDING` state or never complete.
- `enqueue_attendance_batch` API returns HTTP 503 with the message `"Unable to enqueue attendance batch at this time."`
- Application logs contain connection errors to Redis (e.g., `ConnectionError`, `TimeoutError`).

### Diagnostics
1. **Check worker availability**
   ```bash
   celery -A attendance_system_facial_recognition worker inspect ping
   ```
   All registered workers should respond with `pong`.
2. **Inspect active tasks**
   ```bash
   celery -A attendance_system_facial_recognition inspect active
   ```
   Confirm whether tasks are stuck or if queues are empty.
3. **Validate Redis connectivity**
   ```bash
   redis-cli -h <redis-host> -p <redis-port> ping
   ```
   A healthy Redis instance responds with `PONG`.
4. Review Celery worker logs (`logs/celery-worker.log` if using the provided Docker
   setup) for traceback information.

### Recovery Steps
1. Restart Redis and Celery workers:
   ```bash
   docker compose restart redis celery
   ```
   or, if running locally:
   ```bash
   systemctl restart redis
   pkill -f "celery worker"
   celery -A attendance_system_facial_recognition worker -l info
   ```
2. After services are healthy, requeue affected batches using their `task_id`s (see
   [Handling failed batch API tasks](#handling-failed-batch-api-tasks)).
3. Consider scaling worker concurrency temporarily if there is a backlog:
   ```bash
   celery -A attendance_system_facial_recognition worker -l info --concurrency=4
   ```

> ℹ️ When monitoring integrations are available, link automated alerts for Celery
> queue depth, worker heartbeats, and Redis availability here.

## Batch Attendance API Failures

### Symptoms
- HTTP 4xx or 5xx responses from `POST /recognition/enqueue-attendance-batch/`.
- Clients observe rate limiting with HTTP 429 responses.
- API responses contain error details such as `"'records' must be a list."` or `"Record at index 2 must be a JSON object."`

### Diagnostics
1. **Validate payload format**: Ensure the request body is UTF-8 encoded JSON with a
   top-level `records` list.
2. **Check server logs** for JSON parsing or validation errors logged by
   `recognition.views.enqueue_attendance_batch`.
3. **Inspect the Celery task** associated with a submission:
   ```bash
   celery -A attendance_system_facial_recognition inspect query_task <task-id>
   ```
4. **Rate limiting (HTTP 429)**
   - Confirm the caller is not exceeding the configured limit in
     `settings.RECOGNITION_ATTENDANCE_RATE_LIMIT`.
   - Use server access logs to correlate frequent requests from the same user or IP.

### Recovery Steps
1. For malformed payloads, correct the data and retry the request.
2. If Redis/Celery were previously unavailable, follow the outage recovery steps and
   resubmit the batch.
3. When encountering HTTP 429:
   - Back off and retry after the window specified by the rate limit (default `5/m`).
   - Coordinate with the operations team to adjust the limit if legitimate bursts are required.
4. For HTTP 503 responses, verify worker availability and retry once the queue is
   healthy.

> ℹ️ Add references to API request rate dashboards or alert policies when available.

## Fallback Behaviours in `recognition/views.py`

Certain failure scenarios trigger graceful fallbacks in the recognition pipeline.
Understanding them helps interpret API responses and user-facing behaviour.

### Empty Embedding Dataset (HTTP 503)
If no enrolled embeddings are available, `_load_dataset_embeddings_for_matching`
returns an empty index, and the recognition API responds with HTTP 503 and
`"No enrolled face embeddings are available for comparison."`

**Action**
1. Confirm the dataset directory (`face_recognition_data/training_dataset/`) contains
   user subdirectories with captured images.
2. Queue the `capture_dataset` Celery task (via **Add Photos** or `tasks.capture_dataset.delay`) to rebuild
   encrypted samples. Monitor progress through `/tasks/<task-id>/`.

### Liveness Check Failures
When `_passes_liveness_check` fails, the API still returns a JSON payload but sets
`"spoofed": true` and leaves `"recognized": false`.

**Action**
1. Advise the user to retry with better lighting, blink twice, or tilt their head slightly so the motion gate can detect parallax.
2. Review liveness heuristics if false positives occur frequently.
3. Run `python manage.py evaluate_liveness --samples-root /path/to/liveness_samples` with representative clips to validate new `RECOGNITION_LIVENESS_*` thresholds before deploying them.

### Distance Metric Fallbacks
`_calculate_embedding_distance` attempts cosine, Euclidean, or Manhattan metrics. If
all fail, it logs `"Failed to compute fallback distance"` and skips the candidate.

**Action**
1. Inspect embeddings for corruption or unexpected data types.
2. Recompute embeddings by re-running the capture workflow and verifying the corresponding Celery job succeeds.

> ℹ️ Plan to link to recognition service health dashboards (model accuracy,
> liveness failure rate) once monitoring is deployed.

## Handling Failed Batch API Tasks

1. Locate the `task_id` from the batch submission response (`202 Accepted`).
2. Query task status:
   ```bash
   celery -A attendance_system_facial_recognition inspect query_task <task-id>
   ```
   or, if result backend is enabled:
   ```bash
   celery -A attendance_system_facial_recognition result <task-id>
   ```
3. If the task failed, review the Celery result for traceback details and requeue the
   normalized records:
   ```bash
   python manage.py shell <<'PY'
   from recognition.tasks import process_attendance_batch
   records = [...]  # original normalized payload
   process_attendance_batch.delay(records)
   PY
   ```
4. For persistent failures, verify dependent services (database, filesystem access,
   face recognition models) and consult Celery worker logs for stack traces.

> ℹ️ Document alert integrations for task failure rates or retry exhaustion when
> those tools are in place.
