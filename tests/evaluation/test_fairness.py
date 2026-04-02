"""Unit tests for the fairness audit helpers."""

from pathlib import Path

import pytest
from PIL import Image

from src.evaluation.face_recognition_eval import SampleEvaluation
from src.evaluation.fairness import (
    AnnotatedSample,
    FairnessAuditConfig,
    GroupMetrics,
    SampleContext,
    ThresholdRecommendation,
    _bucketize_role,
    _resolve_user_contexts,
    compute_group_metrics,
    compute_threshold_recommendations,
    estimate_lighting_bucket,
)


def create_sample(
    tmp_path: Path, username: str, *, match: str | None, distance: float | None
) -> SampleEvaluation:
    user_dir = tmp_path / username
    user_dir.mkdir(parents=True, exist_ok=True)
    image_path = user_dir / "sample.jpg"
    image_path.write_bytes(b"fake")
    return SampleEvaluation(
        image_path=image_path,
        ground_truth=username,
        match_username=match,
        distance=distance,
        embedding_available=True,
    )


def test_estimate_lighting_bucket_distinguishes_ranges(tmp_path):
    img_path = tmp_path / "dark.png"
    Image.new("L", (5, 5), color=30).save(img_path)
    assert estimate_lighting_bucket(img_path) == "low_light"

    bright_path = tmp_path / "bright.png"
    Image.new("L", (5, 5), color=220).save(bright_path)
    assert estimate_lighting_bucket(bright_path) == "bright_light"


def test_compute_group_metrics_respects_context(tmp_path):
    sample_a = create_sample(tmp_path, "alice", match="alice", distance=0.2)
    sample_b = create_sample(tmp_path, "bob", match=None, distance=None)

    annotated = [
        AnnotatedSample(
            sample=sample_a,
            context=SampleContext(
                username="alice",
                role_bucket="staff_or_admin",
                site_bucket="hq",
                source_bucket="webcam",
                lighting_bucket="bright_light",
            ),
        ),
        AnnotatedSample(
            sample=sample_b,
            context=SampleContext(
                username="bob",
                role_bucket="employee",
                site_bucket="hq",
                source_bucket="webcam",
                lighting_bucket="low_light",
            ),
        ),
    ]

    rows = compute_group_metrics(
        annotated,
        threshold=0.5,
        group_name="role",
        value_getter=lambda ctx: ctx.role_bucket,
    )

    assert {row["group"] for row in rows} == {"staff_or_admin", "employee"}
    staff_row = next(row for row in rows if row["group"] == "staff_or_admin")
    assert staff_row["accuracy"] == 1.0


def test_fairness_audit_config_defaults(tmp_path, monkeypatch):
    import types

    mock_views = types.ModuleType("mock_views")
    mock_views.DEFAULT_DISTANCE_THRESHOLD = 0.4
    mock_views.TRAINING_DATASET_ROOT = str(tmp_path / "dataset")

    from src.evaluation import face_recognition_eval

    monkeypatch.setattr(face_recognition_eval, "_recognition_views", lambda: mock_views)

    config = FairnessAuditConfig(
        reports_dir=tmp_path / "reports", dataset_root=tmp_path / "dataset"
    )
    assert config.reports_dir == tmp_path / "reports"
    assert config.dataset_root == tmp_path / "dataset"
    assert config.evaluation_reports_dir == tmp_path / "reports" / "evaluation_snapshot"

    eval_config = config.to_evaluation_config()
    assert eval_config.reports_dir == config.evaluation_reports_dir
    assert eval_config.dataset_root == config.dataset_root


def test_bucketize_role():
    class MockUser:
        def __init__(self, is_staff=False, is_superuser=False):
            self.is_staff = is_staff
            self.is_superuser = is_superuser

    assert _bucketize_role(None) == "unregistered"
    assert _bucketize_role(MockUser()) == "employee"
    assert _bucketize_role(MockUser(is_staff=True)) == "staff_or_admin"
    assert _bucketize_role(MockUser(is_superuser=True)) == "staff_or_admin"


@pytest.mark.django_db
def test_resolve_user_contexts_db():
    from django.contrib.auth.models import User

    from users.models import RecognitionAttempt

    # Create users
    User.objects.create_user(username="employee_a", password="password")
    User.objects.create_superuser(username="admin_a", password="password")

    # Create some attempts to test site/source counting
    RecognitionAttempt.objects.create(
        username="employee_a", successful=True, site="hq", source="webcam"
    )
    RecognitionAttempt.objects.create(
        username="employee_a", successful=True, site="hq", source="mobile"
    )
    RecognitionAttempt.objects.create(
        username="employee_a", successful=True, site="branch", source="webcam"
    )

    user_map = _resolve_user_contexts(["employee_a", "admin_a", "non_existent"])

    assert "employee_a" in user_map
    assert user_map["employee_a"].role_bucket == "employee"
    assert user_map["employee_a"].site_bucket == "hq"  # Most common
    assert user_map["employee_a"].source_bucket == "webcam"  # Most common

    assert "admin_a" in user_map
    assert user_map["admin_a"].role_bucket == "staff_or_admin"

    assert "non_existent" in user_map
    assert user_map["non_existent"].role_bucket == "unregistered"


def test_compute_threshold_recommendations():
    metrics = {
        "metrics_by_lighting": GroupMetrics(
            name="metrics_by_lighting",
            rows=[
                {"group": "low_light", "samples": 20, "far": 0.01, "frr": 0.20},  # High FRR
                {"group": "bright_light", "samples": 15, "far": 0.10, "frr": 0.02},  # High FAR
                {
                    "group": "moderate_light",
                    "samples": 5,
                    "far": 0.50,
                    "frr": 0.50,
                },  # Ignored (samples < 10)
            ],
            csv_path=Path("dummy.csv"),
        )
    }

    recs = compute_threshold_recommendations(metrics, current_threshold=0.4, min_samples=10)

    assert len(recs) == 2

    low_light_rec = next(r for r in recs if r.group_value == "low_light")
    assert low_light_rec.group_type == "lighting"
    # FRR is 0.20, adjustment is min(0.1, 0.20 * 0.3) = min(0.1, 0.06) = 0.06
    # 0.4 + 0.06 = 0.46
    assert low_light_rec.recommended_threshold == pytest.approx(0.46)
    assert "High FRR" in low_light_rec.adjustment_reason

    bright_light_rec = next(r for r in recs if r.group_value == "bright_light")
    assert bright_light_rec.group_type == "lighting"
    # FAR is 0.10, adjustment is min(0.1, 0.10 * 0.5) = min(0.1, 0.05) = 0.05
    # 0.4 - 0.05 = 0.35
    assert bright_light_rec.recommended_threshold == pytest.approx(0.35)
    assert "High FAR" in bright_light_rec.adjustment_reason


def test_write_threshold_recommendations_csv(tmp_path):
    from src.evaluation.fairness import write_threshold_recommendations_csv

    recs = [
        ThresholdRecommendation(
            group_type="lighting",
            group_value="low_light",
            current_threshold=0.4,
            recommended_threshold=0.46,
            adjustment_reason="High FRR",
            far=0.01,
            frr=0.20,
            sample_count=20,
        )
    ]
    csv_path = tmp_path / "recs.csv"
    write_threshold_recommendations_csv(recs, csv_path)

    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "lighting" in content
    assert "low_light" in content
    assert "0.46" in content


def test_fairness_audit_config_validation():
    with pytest.raises(ValueError, match="limit_samples must be positive when provided"):
        FairnessAuditConfig(limit_samples=-1)
