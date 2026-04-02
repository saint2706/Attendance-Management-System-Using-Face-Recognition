"""Unit tests for the fairness audit helpers."""

from pathlib import Path

from PIL import Image

from src.evaluation.face_recognition_eval import SampleEvaluation
import pytest
from src.evaluation import UNKNOWN_LABEL
import csv
from unittest.mock import patch, MagicMock

from src.evaluation.fairness import (
    AnnotatedSample,
    SampleContext,
    compute_group_metrics,
    estimate_lighting_bucket,
    FairnessAuditConfig,
    _bucketize_role,
    _resolve_user_contexts,
    annotate_samples,
    _write_group_csv,
    _write_summary_markdown,
    compute_threshold_recommendations,
    write_threshold_recommendations_csv,
    run_fairness_audit,
    GroupMetrics,
    ThresholdRecommendation,
)
from src.evaluation.face_recognition_eval import EvaluationSummary
from users.models import User, RecognitionAttempt


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


class TestFairnessAuditConfig:
    def test_invalid_limit_samples(self):
        with pytest.raises(ValueError, match="limit_samples must be positive when provided"):
            FairnessAuditConfig(limit_samples=0)
        with pytest.raises(ValueError, match="limit_samples must be positive when provided"):
            FairnessAuditConfig(limit_samples=-5)

    def test_to_evaluation_config(self, tmp_path):
        config = FairnessAuditConfig(
            reports_dir=tmp_path / "fairness",
            dataset_root=tmp_path / "dataset",
            test_split_csv=tmp_path / "split.csv",
            threshold=0.45,
            limit_samples=100,
        )
        eval_config = config.to_evaluation_config()

        assert eval_config.reports_dir == config.evaluation_reports_dir
        assert eval_config.dataset_root == tmp_path / "dataset"
        assert eval_config.test_split_csv == tmp_path / "split.csv"
        assert eval_config.threshold == 0.45
        assert eval_config.limit_samples == 100


def test_bucketize_role():
    assert _bucketize_role(None) == "unregistered"

    class MockUser:
        def __init__(self, is_superuser=False, is_staff=False):
            self.is_superuser = is_superuser
            self.is_staff = is_staff

    assert _bucketize_role(MockUser(is_superuser=True)) == "staff_or_admin"
    assert _bucketize_role(MockUser(is_staff=True)) == "staff_or_admin"
    assert _bucketize_role(MockUser()) == "employee"


@pytest.mark.django_db
class TestResolveUserContexts:
    def test_resolve_empty_usernames(self):
        assert _resolve_user_contexts([]) == {}
        assert _resolve_user_contexts([UNKNOWN_LABEL]) == {}

    def test_resolve_with_db_data(self):
        User.objects.create(username="alice", is_staff=True)
        User.objects.create(username="bob")

        RecognitionAttempt.objects.create(username="alice", site="hq", source="webcam")
        RecognitionAttempt.objects.create(username="alice", site="hq", source="webcam")
        RecognitionAttempt.objects.create(username="alice", site="branch", source="mobile")

        RecognitionAttempt.objects.create(username="bob", site="branch", source="mobile")

        contexts = _resolve_user_contexts(["alice", "bob", "carol"])

        assert len(contexts) == 3

        assert contexts["alice"].role_bucket == "staff_or_admin"
        assert contexts["alice"].site_bucket == "hq"
        assert contexts["alice"].source_bucket == "webcam"

        assert contexts["bob"].role_bucket == "employee"
        assert contexts["bob"].site_bucket == "branch"
        assert contexts["bob"].source_bucket == "mobile"

        assert contexts["carol"].role_bucket == "unregistered"
        assert contexts["carol"].site_bucket == "unspecified"
        assert contexts["carol"].source_bucket == "unspecified"


@pytest.mark.django_db
def test_annotate_samples(tmp_path):
    # Setup test users before running annotate_samples to avoid KeyError
    User.objects.create(username="alice", is_staff=True)
    # create default context since bob is not in User

    alice_img = tmp_path / "alice.jpg"
    Image.new("L", (5, 5), color=220).save(alice_img)

    bob_img = tmp_path / "bob.jpg"
    Image.new("L", (5, 5), color=30).save(bob_img)

    samples = [
        SampleEvaluation(
            image_path=alice_img,
            ground_truth="alice",
            match_username="alice",
            distance=0.2,
            embedding_available=True,
        ),
        SampleEvaluation(
            image_path=bob_img,
            ground_truth="bob",
            match_username=None,
            distance=None,
            embedding_available=False,
        ),
        SampleEvaluation(
            image_path=tmp_path / "nonexistent.jpg",
            ground_truth=UNKNOWN_LABEL,
            match_username=None,
            distance=None,
            embedding_available=False,
        ),
    ]

    annotated = annotate_samples(samples)

    assert len(annotated) == 3

    assert annotated[0].context.username == "alice"
    assert annotated[0].context.role_bucket == "staff_or_admin"
    assert annotated[0].context.lighting_bucket == "bright_light"

    assert annotated[1].context.username == "bob"
    assert annotated[1].context.role_bucket == "unregistered"
    assert annotated[1].context.lighting_bucket == "low_light"

    assert annotated[2].context.username == UNKNOWN_LABEL
    assert annotated[2].context.role_bucket == UNKNOWN_LABEL
    assert annotated[2].context.lighting_bucket == "unknown"


class TestFairnessIO:
    def test_write_group_csv_empty(self, tmp_path):
        out_path = tmp_path / "empty.csv"
        _write_group_csv([], out_path)

        assert out_path.exists()
        assert out_path.read_text() == "group\n"

    def test_write_group_csv(self, tmp_path):
        out_path = tmp_path / "groups.csv"
        rows = [
            {"group": "staff", "accuracy": 1.0, "far": 0.0},
            {"group": "employee", "accuracy": 0.9, "far": 0.1},
        ]

        _write_group_csv(rows, out_path)

        assert out_path.exists()
        with out_path.open() as f:
            reader = csv.DictReader(f)
            data = list(reader)
            assert len(data) == 2
            assert data[0]["group"] == "staff"

    def test_write_summary_markdown(self, tmp_path):
        out_path = tmp_path / "summary.md"

        summary = EvaluationSummary(
            metrics={"samples": 100, "accuracy": 0.95},
            threshold_sweep=[],
            samples=[],
            y_true=[],
            y_pred=[],
            labels=[],
            artifact_paths={},
        )

        group_metrics = {
            "metrics_by_role": GroupMetrics(
                name="metrics_by_role",
                rows=[{"group": "staff", "samples": 50, "accuracy": 1.0}],
                csv_path=tmp_path / "role.csv",
            )
        }

        _write_summary_markdown(summary=summary, group_metrics=group_metrics, output_path=out_path)

        assert out_path.exists()
        content = out_path.read_text()
        assert "# Fairness & Robustness Audit" in content
        assert "| accuracy | 0.9500 |" in content
        assert "## Metrics By Role" in content
        assert "| staff |" in content

    def test_compute_threshold_recommendations(self):
        group_metrics = {
            "metrics_by_role": GroupMetrics(
                name="metrics_by_role",
                rows=[
                    {"group": "employee", "samples": 50, "far": 0.01, "frr": 0.20},  # High FRR
                    {"group": "visitor", "samples": 20, "far": 0.15, "frr": 0.05},  # High FAR
                    {"group": "staff", "samples": 5, "far": 0.20, "frr": 0.0},  # Low samples
                ],
                csv_path=Path("fake"),
            )
        }

        recs = compute_threshold_recommendations(
            group_metrics, current_threshold=0.4, min_samples=10
        )

        assert len(recs) == 2

        employee_rec = next(r for r in recs if r.group_value == "employee")
        assert employee_rec.recommended_threshold > 0.4
        assert "High FRR" in employee_rec.adjustment_reason

        visitor_rec = next(r for r in recs if r.group_value == "visitor")
        assert visitor_rec.recommended_threshold < 0.4
        assert "High FAR" in visitor_rec.adjustment_reason

    def test_write_threshold_recommendations_csv(self, tmp_path):
        out_path = tmp_path / "recs.csv"

        recs = [
            ThresholdRecommendation(
                group_type="role",
                group_value="employee",
                current_threshold=0.4,
                recommended_threshold=0.45,
                adjustment_reason="High FRR",
                far=0.01,
                frr=0.20,
                sample_count=50,
            )
        ]

        write_threshold_recommendations_csv(recs, out_path)

        assert out_path.exists()
        with out_path.open() as f:
            reader = csv.DictReader(f)
            data = list(reader)
            assert len(data) == 1
            assert data[0]["group_value"] == "employee"


@pytest.mark.django_db
@patch("src.evaluation.face_recognition_eval._recognition_views")
@patch("src.evaluation.fairness.run_face_recognition_evaluation")
def test_run_fairness_audit(mock_eval, mock_views, tmp_path):
    mock_module = MagicMock()
    mock_module.DEFAULT_DISTANCE_THRESHOLD = 0.4
    mock_views.return_value = mock_module
    # Setup mock evaluation result
    mock_summary = EvaluationSummary(
        metrics={"threshold": 0.4, "accuracy": 0.9},
        threshold_sweep=[],
        samples=[
            SampleEvaluation(
                image_path=tmp_path / "alice.jpg",
                ground_truth="alice",
                match_username="alice",
                distance=0.2,
                embedding_available=True,
            )
        ],
        y_true=["alice"],
        y_pred=["alice"],
        labels=["alice"],
        artifact_paths={},
    )
    mock_eval.return_value = mock_summary

    config = FairnessAuditConfig(
        reports_dir=tmp_path / "fairness", dataset_root=tmp_path / "dataset"
    )

    result = run_fairness_audit(config)

    assert result.evaluation == mock_summary
    assert result.summary_path.exists()
    assert "metrics_by_role" in result.group_metrics
    assert result.group_metrics["metrics_by_role"].csv_path.exists()
