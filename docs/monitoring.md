# Monitoring and Health Instrumentation

The attendance capture pipeline now emits Prometheus metrics and structured
health signals so operators can proactively identify webcam or model issues.
This document explains the exported metrics, alert thresholds, and the admin
surfaces that expose the collected information.

## Prometheus metrics

The `recognition.monitoring` module registers its collectors in a dedicated
Prometheus registry. The following metric families are available via the
`/monitoring/metrics/` endpoint (staff authentication required):

| Metric | Type | Labels | Description |
| --- | --- | --- | --- |
| `webcam_manager_start_total` | Counter | `status` | Counts camera start attempts by outcome (`success`, `failure`). |
| `webcam_manager_start_latency_seconds` | Histogram | – | Distribution of camera start latency. |
| `webcam_manager_stop_total` | Counter | `status` | Counts camera shutdown attempts (`success`, `failure`, `timeout`). |
| `webcam_manager_stop_latency_seconds` | Histogram | – | Distribution of shutdown latency. |
| `webcam_manager_running` | Gauge | – | 1 while the capture loop is active. |
| `webcam_active_consumers` | Gauge | – | Active frame consumer contexts. |
| `webcam_frame_delay_seconds` | Histogram | – | Time between successive frames. |
| `webcam_frame_drop_total` | Counter | – | Number of empty-frame reads from the camera. |
| `webcam_last_frame_timestamp_seconds` | Gauge | – | UNIX timestamp of the latest captured frame. |
| `recognition_stage_duration_seconds` | Histogram | `stage` | Duration of critical recognition stages such as `dataset_index_load`, `deepface_warmup`, `deepface_inference`, and `recognition_iteration`. |
| `recognition_warmup_alerts_total` | Counter | `stage` | Warm-up stages that exceeded alert thresholds. |

## Alert thresholds

Thresholds are configurable through environment variables (defaults shown
below). All values are expressed in seconds unless noted otherwise.

| Setting | Default | Purpose |
| --- | --- | --- |
| `RECOGNITION_CAMERA_START_ALERT_SECONDS` | `3.0` | Maximum acceptable camera start latency before emitting a warning alert. |
| `RECOGNITION_FRAME_DELAY_ALERT_SECONDS` | `0.75` | Frame-to-frame delay threshold that triggers frame health alerts. |
| `RECOGNITION_MODEL_LOAD_ALERT_SECONDS` | `4.0` | Maximum time allowed for loading encrypted embeddings before warning. |
| `RECOGNITION_WARMUP_ALERT_SECONDS` | `3.0` | Maximum time for the first DeepFace inference (warm-up). |
| `RECOGNITION_LOOP_ALERT_SECONDS` | `1.5` | Target upper bound for each recognition loop iteration. |
| `RECOGNITION_HEALTH_ALERT_HISTORY` | `50` | Number of recent alerts retained for display in the admin dashboard. |

Values can be overridden in production by exporting the environment variables or
by setting them directly in `settings/base.py` for bespoke deployments.

## Admin dashboard

Staff users can review the aggregated health data at
`/admin/health/`. The dashboard displays:

* Current webcam lifecycle state, active consumer count, and start/stop counts.
* Recent frame timing information, including the latest frame delay and drop
  counts.
* Per-stage duration samples for dataset loading, DeepFace warm-up, inference,
  and the outer recognition loop.
* Recent alerts exceeding configured thresholds.
* A direct link to the Prometheus metrics endpoint for scraping.

All timestamps on the dashboard are expressed in UTC ISO-8601 format. Alerts are
retained according to `RECOGNITION_HEALTH_ALERT_HISTORY` and surface both the
message and structured metadata used by downstream alerting systems.

## Instrumented lifecycle hooks

`recognition.webcam_manager.WebcamManager` emits metrics and structured logs for
start and stop events, including warm-up latency and thread join timeouts. Frame
consumers update the active consumer gauge automatically. The capture loop
records frame delays and empty-frame occurrences, enabling alerting on stalls.

Within `_mark_attendance`, the system measures dataset loading time, the first
DeepFace inference (warm-up), subsequent inference durations, and the total
iteration runtime. Threshold breaches (such as a slow warm-up) surface alerts
and increment the corresponding Prometheus counters.

These signals provide actionable visibility into hardware failures, model
regressions, or deployment misconfiguration, while the Prometheus endpoint makes
it straightforward to integrate with existing monitoring stacks.
