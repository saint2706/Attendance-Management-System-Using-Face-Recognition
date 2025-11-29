"""Database models for the recognition app."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone

logger = logging.getLogger(__name__)


class ThresholdProfile(models.Model):
    """Named threshold profile for recognition distance settings per site."""

    name = models.CharField(
        max_length=100,
        unique=True,
        help_text="Unique profile name (e.g., 'strict_office', 'lenient_lab')",
    )
    description = models.TextField(
        blank=True,
        help_text="Description of when/where this profile should be used",
    )
    distance_threshold = models.FloatField(
        default=0.4,
        validators=[MinValueValidator(0.0), MaxValueValidator(2.0)],
        help_text="Recognition distance threshold (lower = stricter matching)",
    )
    target_far = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Target False Accept Rate used when selecting this threshold",
    )
    target_frr = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Target False Reject Rate used when selecting this threshold",
    )
    selection_method = models.CharField(
        max_length=32,
        blank=True,
        choices=[
            ("eer", "Equal Error Rate (EER)"),
            ("f1", "Optimal F1 Score"),
            ("far", "Target False Accept Rate"),
            ("frr", "Target False Reject Rate"),
            ("manual", "Manually Specified"),
        ],
        default="manual",
        help_text="Method used to select the threshold",
    )
    sites = models.TextField(
        blank=True,
        help_text="Comma-separated list of site codes where this profile applies",
    )
    is_default = models.BooleanField(
        default=False,
        help_text="Use this profile as the fallback when no site-specific profile matches",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-is_default", "name"]
        verbose_name = "Threshold Profile"
        verbose_name_plural = "Threshold Profiles"

    def __str__(self) -> str:
        default_indicator = " (default)" if self.is_default else ""
        return f"{self.name}{default_indicator} @ {self.distance_threshold:.4f}"

    def save(self, *args, **kwargs) -> None:
        """Ensure only one default profile exists."""
        if self.is_default:
            ThresholdProfile.objects.filter(is_default=True).exclude(pk=self.pk).update(
                is_default=False
            )
        super().save(*args, **kwargs)

    @classmethod
    def get_for_site(cls, site_code: str) -> Optional["ThresholdProfile"]:
        """Return the threshold profile applicable for the given site code."""
        if site_code:
            for profile in cls.objects.all():
                site_list = [s.strip().lower() for s in profile.sites.split(",") if s.strip()]
                if site_code.lower() in site_list:
                    return profile
        return cls.objects.filter(is_default=True).first()

    @classmethod
    def get_threshold_for_site(cls, site_code: str) -> float:
        """Return the recognition threshold for the given site, or the system default."""
        profile = cls.get_for_site(site_code)
        if profile:
            return profile.distance_threshold
        return float(getattr(settings, "RECOGNITION_DISTANCE_THRESHOLD", 0.4))


class LivenessResult(models.Model):
    """Persisted liveness check result with confidence scoring."""

    class ChallengeType(models.TextChoices):
        MOTION = "motion", "Motion Detection"
        BLINK = "blink", "Blink Detection"
        HEAD_TURN = "head_turn", "Head Turn"
        ANTI_SPOOF = "anti_spoof", "Anti-Spoofing Model"

    class ChallengeStatus(models.TextChoices):
        PASSED = "passed", "Passed"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"
        TIMEOUT = "timeout", "Timeout"

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    username = models.CharField(max_length=150, blank=True)
    site = models.CharField(max_length=100, blank=True)
    source = models.CharField(max_length=32, blank=True)

    # Challenge details
    challenge_type = models.CharField(
        max_length=32,
        choices=ChallengeType.choices,
        default=ChallengeType.MOTION,
    )
    challenge_status = models.CharField(
        max_length=16,
        choices=ChallengeStatus.choices,
        default=ChallengeStatus.PASSED,
    )

    # Confidence scores
    liveness_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Confidence score from liveness check (0.0 to 1.0)",
    )
    motion_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Motion-based liveness score",
    )
    threshold_used = models.FloatField(
        null=True,
        blank=True,
        help_text="Threshold applied for the liveness decision",
    )

    # Frame metadata
    frames_analyzed = models.PositiveIntegerField(
        default=0,
        help_text="Number of frames analyzed for liveness",
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["created_at", "challenge_status"]),
            models.Index(fields=["username", "created_at"]),
        ]
        verbose_name = "Liveness Result"
        verbose_name_plural = "Liveness Results"

    def __str__(self) -> str:
        status = self.challenge_status
        username = self.username or "unknown"
        return f"{username} {self.challenge_type} {status} @ {self.created_at:%Y-%m-%d %H:%M:%S}"


class RecognitionOutcomeQuerySet(models.QuerySet["RecognitionOutcome"]):
    """Custom queryset with helpers for outcome analytics."""

    def accepted(self) -> "RecognitionOutcomeQuerySet":
        """Return only accepted recognition outcomes."""

        return self.filter(accepted=True)

    def rejected(self) -> "RecognitionOutcomeQuerySet":
        """Return only rejected recognition outcomes."""

        return self.filter(accepted=False)


class RecognitionOutcome(models.Model):
    """Persisted snapshot of a recognition decision made during attendance flows."""

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    username = models.CharField(max_length=150, blank=True)
    direction = models.CharField(max_length=12, blank=True)
    source = models.CharField(max_length=32, blank=True)
    accepted = models.BooleanField()
    confidence = models.FloatField(null=True, blank=True)
    distance = models.FloatField(null=True, blank=True)
    threshold = models.FloatField(null=True, blank=True)
    liveness_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Liveness confidence score (0.0 to 1.0)",
    )
    liveness_passed = models.BooleanField(
        null=True,
        blank=True,
        help_text="Whether the liveness check passed",
    )
    profile_name = models.CharField(
        max_length=100,
        blank=True,
        help_text="Name of the threshold profile used",
    )

    objects: RecognitionOutcomeQuerySet = RecognitionOutcomeQuerySet.as_manager()

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["created_at", "direction"]),
            models.Index(fields=["accepted", "created_at"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - human readability
        status = "accepted" if self.accepted else "rejected"
        username = self.username or "unknown"
        return f"{username} {status} @ {self.created_at:%Y-%m-%d %H:%M:%S}"

    @classmethod
    def prune_expired(cls) -> None:
        """Delete outcomes older than the configured retention window."""

        retention_days: Optional[int] = getattr(settings, "RECOGNITION_OUTCOME_RETENTION_DAYS", 30)
        if retention_days in (None, "none", ""):
            return

        try:
            days = int(retention_days)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            logger.debug(
                "Invalid RECOGNITION_OUTCOME_RETENTION_DAYS=%r; skipping prune.",
                retention_days,
            )
            return

        if days <= 0:
            logger.debug("Retention set to %s days; skipping prune.", days)
            return

        cutoff = timezone.now() - timedelta(days=days)
        cls.objects.filter(created_at__lt=cutoff).delete()


class ModelEvaluationResult(models.Model):
    """Persisted evaluation metrics for tracking model health over time."""

    class EvaluationType(models.TextChoices):
        SCHEDULED_NIGHTLY = "nightly", "Scheduled Nightly"
        SCHEDULED_WEEKLY = "weekly", "Scheduled Weekly"
        FAIRNESS_AUDIT = "fairness", "Fairness Audit"
        LIVENESS_EVAL = "liveness", "Liveness Evaluation"
        MANUAL = "manual", "Manual Evaluation"

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    evaluation_type = models.CharField(
        max_length=16,
        choices=EvaluationType.choices,
        default=EvaluationType.MANUAL,
    )

    # Core evaluation metrics
    accuracy = models.FloatField(
        null=True,
        blank=True,
        help_text="Overall accuracy of the model",
    )
    precision = models.FloatField(
        null=True,
        blank=True,
        help_text="Weighted precision score",
    )
    recall = models.FloatField(
        null=True,
        blank=True,
        help_text="Weighted recall score",
    )
    f1_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Weighted F1 score",
    )
    far = models.FloatField(
        null=True,
        blank=True,
        help_text="False Accept Rate",
    )
    frr = models.FloatField(
        null=True,
        blank=True,
        help_text="False Reject Rate",
    )

    # Evaluation metadata
    samples_evaluated = models.PositiveIntegerField(
        default=0,
        help_text="Number of samples used for evaluation",
    )
    threshold_used = models.FloatField(
        null=True,
        blank=True,
        help_text="Distance threshold used for evaluation",
    )
    identities_evaluated = models.PositiveIntegerField(
        default=0,
        help_text="Number of unique identities in evaluation set",
    )

    # Liveness-specific metrics
    liveness_pass_rate = models.FloatField(
        null=True,
        blank=True,
        help_text="Liveness check pass rate (0.0 to 1.0)",
    )
    liveness_samples = models.PositiveIntegerField(
        default=0,
        help_text="Number of liveness checks evaluated",
    )

    # Task tracking
    task_id = models.CharField(
        max_length=255,
        blank=True,
        help_text="Celery task ID that generated this result",
    )
    duration_seconds = models.FloatField(
        null=True,
        blank=True,
        help_text="Time taken to run the evaluation in seconds",
    )
    error_message = models.TextField(
        blank=True,
        help_text="Error message if evaluation failed",
    )
    success = models.BooleanField(
        default=True,
        help_text="Whether the evaluation completed successfully",
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["created_at", "evaluation_type"]),
            models.Index(fields=["evaluation_type", "success"]),
        ]
        verbose_name = "Model Evaluation Result"
        verbose_name_plural = "Model Evaluation Results"

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{status} {self.get_evaluation_type_display()} @ {self.created_at:%Y-%m-%d %H:%M}"

    @classmethod
    def get_latest(
        cls, evaluation_type: Optional[str] = None, successful_only: bool = True
    ) -> Optional["ModelEvaluationResult"]:
        """Return the most recent evaluation result."""
        qs = cls.objects.all()
        if evaluation_type:
            qs = qs.filter(evaluation_type=evaluation_type)
        if successful_only:
            qs = qs.filter(success=True)
        return qs.order_by("-created_at").first()

    @classmethod
    def get_previous(
        cls, current: "ModelEvaluationResult", evaluation_type: Optional[str] = None
    ) -> Optional["ModelEvaluationResult"]:
        """Return the evaluation result immediately before the provided one."""
        qs = cls.objects.filter(created_at__lt=current.created_at, success=True)
        if evaluation_type:
            qs = qs.filter(evaluation_type=evaluation_type)
        return qs.order_by("-created_at").first()

    def compute_trend(self, previous: Optional["ModelEvaluationResult"] = None) -> dict:
        """Compute metric trends compared to the previous evaluation."""
        if previous is None:
            previous = self.get_previous(self, self.evaluation_type)

        if previous is None:
            return {"has_previous": False, "trends": {}}

        trends = {}
        metrics = ["accuracy", "precision", "recall", "f1_score", "far", "frr"]

        for metric in metrics:
            current_val = getattr(self, metric)
            previous_val = getattr(previous, metric)

            if current_val is None or previous_val is None:
                continue

            diff = current_val - previous_val
            # For FAR and FRR, lower is better
            if metric in ("far", "frr"):
                direction = "improved" if diff < 0 else ("degraded" if diff > 0 else "stable")
            else:
                direction = "improved" if diff > 0 else ("degraded" if diff < 0 else "stable")

            trends[metric] = {
                "current": current_val,
                "previous": previous_val,
                "diff": diff,
                "direction": direction,
            }

        return {
            "has_previous": True,
            "previous_date": previous.created_at,
            "trends": trends,
        }
