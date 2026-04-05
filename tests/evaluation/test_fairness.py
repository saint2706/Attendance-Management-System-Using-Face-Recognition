"""Unit tests for the fairness audit helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.evaluation.face_recognition_eval import SampleEvaluation
from src.evaluation.fairness import (
    AnnotatedSample,
    FairnessAuditConfig,
    SampleContext,
    ThresholdRecommendation,
    _bucketize_role,
    _resolve_user_contexts,
    annotate_samples,
    compute_group_metrics,
    compute_threshold_recommendations,
    estimate_lighting_bucket,
    run_fairness_audit,
    write_threshold_recommendations_csv,
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

    # Test unknown format or failure
    invalid_path = tmp_path / "invalid.txt"
    invalid_path.write_text("not an image")
    assert estimate_lighting_bucket(invalid_path) == "unknown"


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


@patch("src.evaluation.face_recognition_eval._recognition_views")
def test_fairness_audit_config_post_init(mock_views):
    mock_views_module = MagicMock()
    mock_views_module.DEFAULT_DISTANCE_THRESHOLD = 0.4
    mock_views_module.TRAINING_DATASET_ROOT = "test/root"
    mock_views.return_value = mock_views_module

    config = FairnessAuditConfig(limit_samples=10)
    assert config.limit_samples == 10

    with pytest.raises(ValueError, match="limit_samples must be positive"):
        FairnessAuditConfig(limit_samples=-5)

    eval_config = config.to_evaluation_config()
    assert eval_config.limit_samples == 10
    assert eval_config.reports_dir == config.evaluation_reports_dir


def test_bucketize_role():
    assert _bucketize_role(None) == "unregistered"

    user = MagicMock()
    user.is_superuser = True
    user.is_staff = False
    assert _bucketize_role(user) == "staff_or_admin"

    user.is_superuser = False
    user.is_staff = False
    assert _bucketize_role(user) == "employee"


@patch("users.models.RecognitionAttempt")
@patch("django.contrib.auth.get_user_model")
def test_resolve_user_contexts(mock_get_user_model, mock_attempt):
    mock_user = MagicMock()
    mock_user.username = "alice"
    mock_user.is_staff = True
    mock_user.is_superuser = False

    mock_user_model = MagicMock()
    mock_user_model.objects.filter.return_value = [mock_user]
    mock_get_user_model.return_value = mock_user_model

    # Mocking out the database calls that _resolve_user_contexts makes
    # Side effects returning the site list and then the source list
    mock_attempt.objects.filter.return_value.exclude.return_value.values.return_value.annotate.side_effect = [
        [{"username": "alice", "site": "hq", "total": 5}],
        [{"username": "alice", "source": "cam1", "total": 3}],
    ]

    contexts = _resolve_user_contexts(["alice", "unknown"])

    assert "alice" in contexts
    assert contexts["alice"].role_bucket == "staff_or_admin"
    assert contexts["alice"].site_bucket == "hq"

    assert "unknown" in contexts
    assert contexts["unknown"].role_bucket == "unregistered"


@patch("src.evaluation.fairness._resolve_user_contexts")
def test_annotate_samples(mock_resolve, tmp_path):
    from src.evaluation.face_recognition_eval import UNKNOWN_LABEL

    mock_resolve.return_value = {
        "alice": SampleContext("alice", "staff_or_admin", "hq", "cam1", "unknown")
    }

    sample = create_sample(tmp_path, "alice", match="alice", distance=0.2)
    sample_unknown = create_sample(tmp_path, UNKNOWN_LABEL, match=None, distance=None)

    annotated = annotate_samples([sample, sample_unknown])

    assert len(annotated) == 2
    assert annotated[0].context.role_bucket == "staff_or_admin"
    assert annotated[1].context.role_bucket == UNKNOWN_LABEL


@patch("src.evaluation.fairness.annotate_samples")
@patch("src.evaluation.face_recognition_eval._recognition_views")
@patch("src.evaluation.fairness.run_face_recognition_evaluation")
def test_run_fairness_audit(mock_run_eval, mock_views, mock_annotate, tmp_path):
    mock_annotate.return_value = []
    mock_views_module = MagicMock()
    mock_views_module.DEFAULT_DISTANCE_THRESHOLD = 0.4
    mock_views_module.TRAINING_DATASET_ROOT = "test/root"
    mock_views.return_value = mock_views_module

    mock_summary = MagicMock()
    mock_summary.metrics = {"threshold": 0.5, "accuracy": 0.9}
    sample = create_sample(tmp_path, "alice", match="alice", distance=0.2)
    mock_summary.samples = [sample]
    mock_run_eval.return_value = mock_summary

    config = FairnessAuditConfig(reports_dir=tmp_path / "reports")
    result = run_fairness_audit(config)

    assert result.summary_path.exists()
    assert len(result.group_metrics) == 4
    assert "metrics_by_role" in result.group_metrics


def test_compute_threshold_recommendations():
    group_metrics = {
        "metrics_by_role": MagicMock(
            rows=[
                {"group": "staff", "samples": 20, "far": 0.1, "frr": 0.01},
                {
                    "group": "employee",
                    "samples": 5,
                    "far": 0.1,
                    "frr": 0.01,
                },  # ignored due to min_samples
                {"group": "visitor", "samples": 50, "far": 0.01, "frr": 0.20},
            ]
        )
    }

    recs = compute_threshold_recommendations(group_metrics, current_threshold=0.5, min_samples=10)

    assert len(recs) == 2

    # staff: high FAR -> stricter threshold (lower)
    staff_rec = next(r for r in recs if r.group_value == "staff")
    assert staff_rec.recommended_threshold < 0.5

    # visitor: high FRR -> looser threshold (higher)
    visitor_rec = next(r for r in recs if r.group_value == "visitor")
    assert visitor_rec.recommended_threshold > 0.5


def test_write_threshold_recommendations_csv(tmp_path):
    recs = [
        ThresholdRecommendation(
            group_type="role",
            group_value="staff",
            current_threshold=0.5,
            recommended_threshold=0.45,
            adjustment_reason="High FAR",
            far=0.1,
            frr=0.01,
            sample_count=20,
        )
    ]

    out_path = tmp_path / "recs.csv"
    write_threshold_recommendations_csv(recs, out_path)

    content = out_path.read_text()
    assert "group_type,group_value,current_threshold" in content
    assert "role,staff,0.5,0.45,High FAR,0.1000,0.0100,20" in content
