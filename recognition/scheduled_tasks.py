"""Scheduled Celery tasks for model evaluation, fairness audits, and liveness checks."""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db.models import Avg, Count, Q
from django.utils import timezone

from celery import shared_task

from .models import LivenessResult, ModelEvaluationResult

logger = logging.getLogger(__name__)


def _get_reports_dir() -> Path:
    """Return the directory for evaluation reports."""
    return Path(settings.BASE_DIR) / "reports"


def _run_face_recognition_evaluation(
    evaluation_type: str,
    task_id: str,
    limit_samples: int | None = None,
) -> ModelEvaluationResult:
    """Execute face recognition evaluation and persist results."""
    from src.evaluation.face_recognition_eval import EvaluationConfig, run_face_recognition_evaluation

    start_time = time.time()
    reports_dir = _get_reports_dir() / "evaluation" / evaluation_type
    reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = EvaluationConfig(
            reports_dir=reports_dir,
            limit_samples=limit_samples,
        )
        summary = run_face_recognition_evaluation(config)

        duration = time.time() - start_time
        metrics = summary.metrics

        # Count unique identities in the evaluation
        identities = {s.ground_truth for s in summary.samples if s.ground_truth}

        result = ModelEvaluationResult.objects.create(
            evaluation_type=evaluation_type,
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1"),
            far=metrics.get("far"),
            frr=metrics.get("frr"),
            samples_evaluated=metrics.get("samples", 0),
            threshold_used=metrics.get("threshold"),
            identities_evaluated=len(identities),
            task_id=task_id,
            duration_seconds=duration,
            success=True,
        )

        logger.info(
            "Evaluation completed: type=%s accuracy=%.4f samples=%d duration=%.2fs",
            evaluation_type,
            result.accuracy or 0,
            result.samples_evaluated,
            duration,
        )
        return result

    except Exception as exc:
        duration = time.time() - start_time
        result = ModelEvaluationResult.objects.create(
            evaluation_type=evaluation_type,
            task_id=task_id,
            duration_seconds=duration,
            success=False,
            error_message=str(exc),
        )
        logger.exception("Evaluation failed: type=%s error=%s", evaluation_type, exc)
        return result


@shared_task(bind=True, name="recognition.scheduled_tasks.run_scheduled_evaluation")
def run_scheduled_evaluation(
    self,
    evaluation_type: str = "nightly",
    limit_samples: int | None = None,
) -> dict:
    """
    Run a scheduled face recognition evaluation.

    Args:
        evaluation_type: Type of evaluation ('nightly' or 'weekly')
        limit_samples: Optional limit on number of samples to evaluate

    Returns:
        Dictionary with evaluation results and metadata
    """
    task_id = self.request.id or ""
    logger.info("Starting scheduled %s evaluation (task_id=%s)", evaluation_type, task_id)

    result = _run_face_recognition_evaluation(
        evaluation_type=evaluation_type,
        task_id=task_id,
        limit_samples=limit_samples,
    )

    # Compute trend vs previous
    trend_data = result.compute_trend()

    return {
        "evaluation_id": result.id,
        "evaluation_type": evaluation_type,
        "success": result.success,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "far": result.far,
        "frr": result.frr,
        "samples_evaluated": result.samples_evaluated,
        "duration_seconds": result.duration_seconds,
        "error_message": result.error_message or None,
        "trend": trend_data,
    }


@shared_task(bind=True, name="recognition.scheduled_tasks.run_fairness_audit")
def run_fairness_audit(
    self,
    limit_samples: int | None = None,
) -> dict:
    """
    Run a scheduled fairness audit.

    Returns:
        Dictionary with audit results and metadata
    """
    from src.evaluation.fairness import FairnessAuditConfig, run_fairness_audit as _run_audit

    task_id = self.request.id or ""
    start_time = time.time()
    reports_dir = _get_reports_dir() / "fairness"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting scheduled fairness audit (task_id=%s)", task_id)

    try:
        config = FairnessAuditConfig(
            reports_dir=reports_dir,
            limit_samples=limit_samples,
        )
        audit_result = _run_audit(config)
        duration = time.time() - start_time

        # Extract key metrics from the evaluation
        metrics = audit_result.evaluation.metrics
        identities = {
            s.ground_truth for s in audit_result.evaluation.samples if s.ground_truth
        }

        result = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.FAIRNESS_AUDIT,
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1"),
            far=metrics.get("far"),
            frr=metrics.get("frr"),
            samples_evaluated=metrics.get("samples", 0),
            threshold_used=metrics.get("threshold"),
            identities_evaluated=len(identities),
            task_id=task_id,
            duration_seconds=duration,
            success=True,
        )

        logger.info(
            "Fairness audit completed: accuracy=%.4f samples=%d duration=%.2fs",
            result.accuracy or 0,
            result.samples_evaluated,
            duration,
        )

        return {
            "evaluation_id": result.id,
            "evaluation_type": "fairness",
            "success": True,
            "accuracy": result.accuracy,
            "samples_evaluated": result.samples_evaluated,
            "duration_seconds": duration,
            "group_metrics": list(audit_result.group_metrics.keys()),
            "summary_path": str(audit_result.summary_path),
        }

    except Exception as exc:
        duration = time.time() - start_time
        result = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.FAIRNESS_AUDIT,
            task_id=task_id,
            duration_seconds=duration,
            success=False,
            error_message=str(exc),
        )
        logger.exception("Fairness audit failed: error=%s", exc)
        return {
            "evaluation_id": result.id,
            "evaluation_type": "fairness",
            "success": False,
            "error_message": str(exc),
            "duration_seconds": duration,
        }


@shared_task(bind=True, name="recognition.scheduled_tasks.run_liveness_evaluation")
def run_liveness_evaluation(
    self,
    days_back: int = 7,
) -> dict:
    """
    Run a scheduled liveness evaluation based on recent liveness check results.

    This evaluates the liveness detection system by analyzing recent check results
    and computing pass rates and confidence metrics.

    Args:
        days_back: Number of days of liveness results to analyze

    Returns:
        Dictionary with liveness evaluation metrics
    """
    task_id = self.request.id or ""
    start_time = time.time()

    logger.info(
        "Starting scheduled liveness evaluation (task_id=%s, days_back=%d)",
        task_id,
        days_back,
    )

    try:
        since = timezone.now() - timedelta(days=days_back)

        # Aggregate liveness results
        results = LivenessResult.objects.filter(created_at__gte=since)
        total_checks = results.count()

        if total_checks == 0:
            duration = time.time() - start_time
            result = ModelEvaluationResult.objects.create(
                evaluation_type=ModelEvaluationResult.EvaluationType.LIVENESS_EVAL,
                liveness_samples=0,
                liveness_pass_rate=None,
                task_id=task_id,
                duration_seconds=duration,
                success=True,
            )
            logger.info("Liveness evaluation: no samples in the last %d days", days_back)
            return {
                "evaluation_id": result.id,
                "evaluation_type": "liveness",
                "success": True,
                "liveness_samples": 0,
                "liveness_pass_rate": None,
                "duration_seconds": duration,
                "message": f"No liveness checks in the last {days_back} days",
            }

        passed_count = results.filter(challenge_status="passed").count()
        pass_rate = passed_count / total_checks

        # Get average confidence for passed checks
        avg_confidence = results.filter(challenge_status="passed").aggregate(
            avg_confidence=Avg("liveness_confidence")
        )["avg_confidence"]

        # Get breakdown by challenge type
        by_challenge = (
            results.values("challenge_type")
            .annotate(
                total=Count("id"),
                passed=Count("id", filter=Q(challenge_status="passed")),
            )
        )

        duration = time.time() - start_time

        result = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.LIVENESS_EVAL,
            liveness_samples=total_checks,
            liveness_pass_rate=pass_rate,
            task_id=task_id,
            duration_seconds=duration,
            success=True,
        )

        logger.info(
            "Liveness evaluation completed: samples=%d pass_rate=%.4f duration=%.2fs",
            total_checks,
            pass_rate,
            duration,
        )

        return {
            "evaluation_id": result.id,
            "evaluation_type": "liveness",
            "success": True,
            "liveness_samples": total_checks,
            "liveness_pass_rate": pass_rate,
            "avg_confidence": avg_confidence,
            "duration_seconds": duration,
            "by_challenge": list(by_challenge),
        }

    except Exception as exc:
        duration = time.time() - start_time
        result = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.LIVENESS_EVAL,
            task_id=task_id,
            duration_seconds=duration,
            success=False,
            error_message=str(exc),
        )
        logger.exception("Liveness evaluation failed: error=%s", exc)
        return {
            "evaluation_id": result.id,
            "evaluation_type": "liveness",
            "success": False,
            "error_message": str(exc),
            "duration_seconds": duration,
        }


__all__ = [
    "run_scheduled_evaluation",
    "run_fairness_audit",
    "run_liveness_evaluation",
]
