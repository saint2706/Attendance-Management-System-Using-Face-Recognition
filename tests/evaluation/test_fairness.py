"""Unit tests for the fairness audit helpers."""

from pathlib import Path

from PIL import Image

from src.evaluation.face_recognition_eval import SampleEvaluation
from src.evaluation.fairness import AnnotatedSample, SampleContext, compute_group_metrics, estimate_lighting_bucket


def create_sample(tmp_path: Path, username: str, *, match: str | None, distance: float | None) -> SampleEvaluation:
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
