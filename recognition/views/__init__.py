"""
Recognition views package.

This package reorganizes the monolithic views.py into focused modules:
- utils: Shared utilities (Sentry, logging, rate limiting)
- config: Configuration getter functions  
- liveness_checks: Liveness detection (extracted or imported from legacy)
- api: Face recognition API endpoints
- attendance: Attendance marking workflows
- dashboards: Dashboard views
- analytics: Attendance analytics and visualization
- dataset: Dataset management and training
- reporting: Attendance viewing and reporting

For backward compatibility, all public views are re-exported from this __init__.py.
The legacy views.py file is preserved as views_legacy.py for reference and gradual migration.
"""

# Import all public views from the legacy module
# This maintains backward compatibility while we gradually migrate code
from recognition.views_legacy import (
    # API views
    FaceRecognitionAPI,
    enqueue_attendance_batch,
    monitoring_metrics,
    task_status,
    # Attendance marking views
    mark_your_attendance,
    mark_your_attendance_out,
    attendance_session,
    attendance_session_feed,
    # Dashboard views
    home,
    dashboard,
    # Dataset & training views
    add_photos,
    train,
    # Reporting views
    view_attendance_home,
    view_attendance_date,
    view_attendance_employee,
    view_my_attendance_employee_login,
    # Utility views
    not_authorised,
    # Helper functions used by analytics
    check_validity_times,
    # Chart generation functions
    hours_vs_employee_given_date,
    # Dashboard helper functions
    total_number_employees,
    employees_present_today,
    this_week_emp_count_vs_date,
    last_week_emp_count_vs_date,
    # Constants and objects used by tasks
    DATA_ROOT,
    TRAINING_DATASET_ROOT,
    _dataset_embedding_cache,
    # Internal helper functions used by tasks
    _get_or_compute_cached_embedding,
    _is_headless_environment,
    _get_face_detection_backend,
    _get_face_recognition_model,
    _should_enforce_detection,
    _get_recognition_training_seed,
    _get_recognition_training_test_split_ratio,
    _build_dataset_embeddings_for_matching,
    _load_dataset_embeddings_for_matching,
    _passes_liveness_check,
    # Attendance update functions used by tasks
    update_attendance_in_db_in,
    update_attendance_in_db_out,
)

# Re-export utilities and config from new modules
from .utils import (
    _record_sentry_breadcrumb,
    _bind_request_to_sentry_scope,
    _RecognitionAttemptLogger,
    _resolve_recognition_site,
    _attach_attempt_user,
    attendance_rate_limited,
    _enqueue_attendance_records,
    _describe_async_result,
    username_present,
)

from .config import (
    get_face_recognition_model,
    get_face_detection_backend,
    should_enforce_detection,
    get_deepface_distance_metric,
    is_liveness_enabled,
    is_lightweight_liveness_enabled,
    get_lightweight_liveness_min_frames,
    get_lightweight_liveness_threshold,
    get_lightweight_liveness_window,
)

__all__ = [
    # API views
    "FaceRecognitionAPI",
    "enqueue_attendance_batch",
    "monitoring_metrics",
    "task_status",
    # Attendance marking views
    "mark_your_attendance",
    "mark_your_attendance_out",
    "attendance_session",
    "attendance_session_feed",
    # Dashboard views
    "home",
    "dashboard",
    # Dataset & training views
    "add_photos",
    "train",
    # Reporting views
    "view_attendance_home",
    "view_attendance_date",
    "view_attendance_employee",
    "view_my_attendance_employee_login",
    # Utility views
    "not_authorised",
    # Helper functions
    "check_validity_times",
    # Chart generation functions
    "hours_vs_employee_given_date",
    # Dashboard helper functions
    "total_number_employees",
    "employees_present_today",
    "this_week_emp_count_vs_date",
    "last_week_emp_count_vs_date",
    # Constants and objects
    "DATA_ROOT",
    "TRAINING_DATASET_ROOT",
    "_dataset_embedding_cache",
    # Internal helper functions
    "_get_or_compute_cached_embedding",
    "_is_headless_environment",
    "_get_face_detection_backend",
    "_get_face_recognition_model",
    "_should_enforce_detection",
    "_get_recognition_training_seed",
    "_get_recognition_training_test_split_ratio",
    "_build_dataset_embeddings_for_matching",
    "_load_dataset_embeddings_for_matching",
    "_passes_liveness_check",
    # Attendance update functions
    "update_attendance_in_db_in",
    "update_attendance_in_db_out",
    # Utilities (now organized in separate modules)
    "_record_sentry_breadcrumb",
    "_bind_request_to_sentry_scope",
    "_RecognitionAttemptLogger",
    "_resolve_recognition_site",
    "_attach_attempt_user",
    "attendance_rate_limited",
    "_enqueue_attendance_records",
    "_describe_async_result",
    "username_present",
    # Configuration functions
    "get_face_recognition_model",
    "get_face_detection_backend",
    "should_enforce_detection",
    "get_deepface_distance_metric",
    "is_liveness_enabled",
    "is_lightweight_liveness_enabled",
    "get_lightweight_liveness_min_frames",
    "get_lightweight_liveness_threshold",
    "get_lightweight_liveness_window",
]
