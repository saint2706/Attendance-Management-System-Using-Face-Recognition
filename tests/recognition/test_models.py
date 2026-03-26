from datetime import timedelta

from django.test import TestCase
from django.test.utils import override_settings
from django.utils import timezone

from recognition.models import (
    LivenessResult,
    ModelEvaluationResult,
    RecognitionOutcome,
    ThresholdProfile,
)


class TestThresholdProfileGroup(TestCase):
    def test_get_for_group(self):
        # Setup
        ThresholdProfile.objects.create(
            name="light_profile",
            distance_threshold=0.35,
            group_type=ThresholdProfile.GroupType.LIGHTING,
            group_value="low_light",
        )

        # Act
        profile = ThresholdProfile.get_for_group(ThresholdProfile.GroupType.LIGHTING, "low_light")

        # Assert
        assert profile is not None
        assert profile.name == "light_profile"

        # Missing args
        assert ThresholdProfile.get_for_group("", "low_light") is None
        assert ThresholdProfile.get_for_group(ThresholdProfile.GroupType.LIGHTING, "") is None
        assert ThresholdProfile.get_for_group("", "") is None

    def test_get_threshold_for_group(self):
        ThresholdProfile.objects.create(
            name="light_profile",
            distance_threshold=0.35,
            group_type=ThresholdProfile.GroupType.LIGHTING,
            group_value="low_light",
        )
        ThresholdProfile.objects.create(name="site_profile", distance_threshold=0.45, sites="site1")
        ThresholdProfile.objects.create(
            name="default_profile", distance_threshold=0.4, is_default=True
        )

        # Hit group
        assert (
            ThresholdProfile.get_threshold_for_group(
                ThresholdProfile.GroupType.LIGHTING, "low_light"
            )
            == 0.35
        )
        # Fallback to site
        assert (
            ThresholdProfile.get_threshold_for_group(
                ThresholdProfile.GroupType.ROLE, "admin", fallback_site="site1"
            )
            == 0.45
        )
        # Fallback to default
        assert (
            ThresholdProfile.get_threshold_for_group(
                ThresholdProfile.GroupType.ROLE, "admin", fallback_site="unknown"
            )
            == 0.4
        )

    @override_settings(RECOGNITION_DISTANCE_THRESHOLD=0.5)
    def test_get_threshold_for_group_no_default(self):
        assert (
            ThresholdProfile.get_threshold_for_group(
                ThresholdProfile.GroupType.ROLE, "admin", fallback_site="unknown"
            )
            == 0.5
        )


class TestLivenessResultModel(TestCase):
    def test_str_representation(self):
        liveness = LivenessResult.objects.create(
            username="testuser",
            challenge_type=LivenessResult.ChallengeType.MOTION,
            challenge_status=LivenessResult.ChallengeStatus.PASSED,
        )
        assert "testuser" in str(liveness)
        assert "motion" in str(liveness)
        assert "passed" in str(liveness)

        liveness_no_user = LivenessResult.objects.create(
            challenge_type=LivenessResult.ChallengeType.BLINK,
            challenge_status=LivenessResult.ChallengeStatus.FAILED,
        )
        assert "unknown" in str(liveness_no_user)
        assert "blink" in str(liveness_no_user)
        assert "failed" in str(liveness_no_user)


class TestRecognitionOutcomeModel(TestCase):
    def test_queryset_methods(self):
        RecognitionOutcome.objects.create(username="user1", accepted=True)
        RecognitionOutcome.objects.create(username="user2", accepted=False)

        assert RecognitionOutcome.objects.accepted().count() == 1
        assert RecognitionOutcome.objects.accepted().first().username == "user1"
        assert RecognitionOutcome.objects.rejected().count() == 1
        assert RecognitionOutcome.objects.rejected().first().username == "user2"

    def test_prune_expired(self):
        # Create a recent outcome
        RecognitionOutcome.objects.create(username="recent", accepted=True)

        # Create an old outcome
        old_outcome = RecognitionOutcome.objects.create(username="old", accepted=False)
        old_outcome.created_at = timezone.now() - timedelta(days=40)
        old_outcome.save()

        assert RecognitionOutcome.objects.count() == 2

        # Test pruning
        RecognitionOutcome.prune_expired()

        assert RecognitionOutcome.objects.count() == 1
        assert RecognitionOutcome.objects.first().username == "recent"

    @override_settings(RECOGNITION_OUTCOME_RETENTION_DAYS=None)
    def test_prune_expired_none(self):
        RecognitionOutcome.objects.create(username="recent", accepted=True)
        old_outcome = RecognitionOutcome.objects.create(username="old", accepted=False)
        old_outcome.created_at = timezone.now() - timedelta(days=40)
        old_outcome.save()

        RecognitionOutcome.prune_expired()
        assert RecognitionOutcome.objects.count() == 2

    @override_settings(RECOGNITION_OUTCOME_RETENTION_DAYS="none")
    def test_prune_expired_string_none(self):
        RecognitionOutcome.objects.create(username="recent", accepted=True)
        old_outcome = RecognitionOutcome.objects.create(username="old", accepted=False)
        old_outcome.created_at = timezone.now() - timedelta(days=40)
        old_outcome.save()

        RecognitionOutcome.prune_expired()
        assert RecognitionOutcome.objects.count() == 2

    @override_settings(RECOGNITION_OUTCOME_RETENTION_DAYS=0)
    def test_prune_expired_zero(self):
        RecognitionOutcome.objects.create(username="recent", accepted=True)
        old_outcome = RecognitionOutcome.objects.create(username="old", accepted=False)
        old_outcome.created_at = timezone.now() - timedelta(days=40)
        old_outcome.save()

        RecognitionOutcome.prune_expired()
        assert RecognitionOutcome.objects.count() == 2


class TestModelEvaluationResultModel(TestCase):
    def test_str_representation(self):
        eval_success = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY, success=True
        )
        assert "✓" in str(eval_success)
        assert "Scheduled Nightly" in str(eval_success)

        eval_fail = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.MANUAL, success=False
        )
        assert "✗" in str(eval_fail)
        assert "Manual Evaluation" in str(eval_fail)

    def test_get_latest(self):
        old_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY, success=True
        )
        old_eval.created_at = timezone.now() - timedelta(days=1)
        old_eval.save()

        recent_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY, success=True
        )

        # Test latest general
        assert ModelEvaluationResult.get_latest().id == recent_eval.id

        # Test latest by type
        assert (
            ModelEvaluationResult.get_latest(
                evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY
            ).id
            == recent_eval.id
        )
        assert (
            ModelEvaluationResult.get_latest(
                evaluation_type=ModelEvaluationResult.EvaluationType.MANUAL
            )
            is None
        )

    def test_get_previous(self):
        older_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY, success=True
        )
        older_eval.created_at = timezone.now() - timedelta(days=2)
        older_eval.save()

        old_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY, success=True
        )
        old_eval.created_at = timezone.now() - timedelta(days=1)
        old_eval.save()

        recent_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY, success=True
        )

        # Test get previous
        prev = ModelEvaluationResult.get_previous(recent_eval)
        assert prev.id == old_eval.id

        prev_older = ModelEvaluationResult.get_previous(old_eval)
        assert prev_older.id == older_eval.id

        prev_none = ModelEvaluationResult.get_previous(older_eval)
        assert prev_none is None

    def test_compute_trend(self):
        old_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.90,
            precision=0.90,
            recall=0.90,
            f1_score=0.90,
            far=0.05,
            frr=0.05,
        )
        old_eval.created_at = timezone.now() - timedelta(days=1)
        old_eval.save()

        recent_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.95,  # Improved
            precision=0.85,  # Degraded
            recall=0.90,  # Stable
            f1_score=0.9001,  # Stable (within epsilon)
            far=0.02,  # Improved (lower is better)
            frr=0.08,  # Degraded (lower is better)
        )

        trend = recent_eval.compute_trend()
        assert trend["has_previous"] is True
        assert trend["trends"]["accuracy"]["direction"] == "improved"
        assert trend["trends"]["precision"]["direction"] == "degraded"
        assert trend["trends"]["recall"]["direction"] == "stable"
        assert trend["trends"]["f1_score"]["direction"] == "stable"
        assert trend["trends"]["far"]["direction"] == "improved"
        assert trend["trends"]["frr"]["direction"] == "degraded"

    def test_compute_trend_no_previous(self):
        recent_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
        )

        trend = recent_eval.compute_trend()
        assert trend["has_previous"] is False

    def test_compute_trend_missing_metrics(self):
        old_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.90,
        )
        old_eval.created_at = timezone.now() - timedelta(days=1)
        old_eval.save()

        recent_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.95,
        )

        trend = recent_eval.compute_trend()
        assert trend["has_previous"] is True
        assert "accuracy" in trend["trends"]
        assert "precision" not in trend["trends"]


class TestThresholdProfileStr(TestCase):
    def test_str_with_group(self):
        profile = ThresholdProfile.objects.create(
            name="light_profile",
            distance_threshold=0.35,
            group_type=ThresholdProfile.GroupType.LIGHTING,
            group_value="low_light",
        )
        assert "light_profile" in str(profile)
        assert "[lighting:low_light]" in str(profile)

    def test_str_with_group_default(self):
        profile = ThresholdProfile.objects.create(
            name="light_profile",
            distance_threshold=0.35,
            is_default=True,
            group_type=ThresholdProfile.GroupType.LIGHTING,
            group_value="low_light",
        )
        assert "light_profile" in str(profile)
        assert "(default)" in str(profile)
        assert "[lighting:low_light]" in str(profile)


class TestModelEvaluationResultLatestAndPreviousUnsuccessful(TestCase):
    def test_get_latest_unsuccessful(self):
        eval1 = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY, success=False
        )
        assert ModelEvaluationResult.get_latest(successful_only=False).id == eval1.id

    def test_get_previous_with_type(self):
        current = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY, success=True
        )
        old_nightly = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY, success=True
        )
        old_nightly.created_at = current.created_at - timedelta(days=2)
        old_nightly.save()
        old_weekly = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_WEEKLY, success=True
        )
        old_weekly.created_at = current.created_at - timedelta(days=1)
        old_weekly.save()

        prev = ModelEvaluationResult.get_previous(
            current, evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY
        )
        assert prev.id == old_nightly.id


class TestThresholdProfileGetForSiteEdgeCases(TestCase):
    def test_get_for_site_matches_list(self):
        profile = ThresholdProfile.objects.create(
            name="site_profile", distance_threshold=0.45, sites="site1, site2,site3 "
        )
        assert ThresholdProfile.get_for_site("site2") == profile
        assert ThresholdProfile.get_for_site("site3") == profile

    def test_get_for_site_matches_partial_but_not_exact(self):
        """Test that get_for_site handles icontains match without exact match."""
        ThresholdProfile.objects.create(name="site1", distance_threshold=0.4, sites="test_site_1")

        profile = ThresholdProfile.get_for_site("test_site")
        assert profile is None


class TestModelEvaluationResultComputeTrendExplicit(TestCase):
    def test_compute_trend_explicit_previous(self):
        old_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.90,
        )
        recent_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.95,
        )
        trend = recent_eval.compute_trend(previous=old_eval)
        assert trend["has_previous"] is True
        assert trend["trends"]["accuracy"]["direction"] == "improved"


class TestThresholdProfileDefaults(TestCase):
    def test_get_for_site_fallback_to_default(self):
        default_profile = ThresholdProfile.objects.create(
            name="default_profile", distance_threshold=0.4, is_default=True
        )
        assert ThresholdProfile.get_for_site("unknown_site").id == default_profile.id

    def test_get_threshold_for_site_with_profile(self):
        ThresholdProfile.objects.create(name="site_profile", distance_threshold=0.45, sites="site1")
        assert ThresholdProfile.get_threshold_for_site("site1") == 0.45


class TestModelTrendNoPrevous(TestCase):
    def test_compute_trend_no_previous_found(self):
        recent_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.95,
        )
        trend = recent_eval.compute_trend()
        assert trend["has_previous"] is False
        assert trend["trends"] == {}


class TestThresholdProfileSiteFallbackHit(TestCase):
    def test_get_threshold_for_group_site_fallback_hit(self):
        ThresholdProfile.objects.create(name="site_profile", distance_threshold=0.45, sites="site1")
        # Should match site profile before default profile
        ThresholdProfile.objects.create(
            name="default_profile", distance_threshold=0.4, is_default=True
        )
        assert (
            ThresholdProfile.get_threshold_for_group(
                ThresholdProfile.GroupType.ROLE, "admin", fallback_site="site1"
            )
            == 0.45
        )


class TestThresholdProfileDefaultsFallback(TestCase):
    @override_settings(RECOGNITION_DISTANCE_THRESHOLD=0.6)
    def test_get_threshold_for_site_default(self):
        assert ThresholdProfile.get_threshold_for_site("unknown") == 0.6

    def test_get_threshold_for_site_default_is_default_profile(self):
        ThresholdProfile.objects.create(
            name="default_profile", distance_threshold=0.35, is_default=True
        )
        # Note: get_threshold_for_site calls get_for_site, which falls back to default profile.
        assert ThresholdProfile.get_threshold_for_site("unknown") == 0.35


class TestModelEvaluationResultTrendDegraded(TestCase):
    def test_compute_trend_degraded_accuracy(self):
        old_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.95,
        )
        recent_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.90,
        )
        trend = recent_eval.compute_trend(previous=old_eval)
        assert trend["has_previous"] is True
        assert trend["trends"]["accuracy"]["direction"] == "degraded"


class TestThresholdProfileGetForSiteEmpty(TestCase):
    def test_get_for_site_empty_string(self):
        default_profile = ThresholdProfile.objects.create(
            name="default_profile", distance_threshold=0.4, is_default=True
        )
        assert ThresholdProfile.get_for_site("").id == default_profile.id


class TestModelEvaluationResultStableAccuracy(TestCase):
    def test_compute_trend_stable_accuracy(self):
        old_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.90,
        )
        recent_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            accuracy=0.90,
        )
        trend = recent_eval.compute_trend(previous=old_eval)
        assert trend["has_previous"] is True
        assert trend["trends"]["accuracy"]["direction"] == "stable"


class TestThresholdProfileDefaultsFallbackHitEmpty(TestCase):
    def test_get_threshold_for_group_no_group_or_site_matches(self):
        ThresholdProfile.objects.create(
            name="default_profile", distance_threshold=0.35, is_default=True
        )
        assert (
            ThresholdProfile.get_threshold_for_group(
                ThresholdProfile.GroupType.ROLE, "admin", fallback_site="unknown_site"
            )
            == 0.35
        )


class TestModelEvaluationResultStableFRR(TestCase):
    def test_compute_trend_stable_frr(self):
        old_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            frr=0.05,
        )
        recent_eval = ModelEvaluationResult.objects.create(
            evaluation_type=ModelEvaluationResult.EvaluationType.SCHEDULED_NIGHTLY,
            success=True,
            frr=0.05,
        )
        trend = recent_eval.compute_trend(previous=old_eval)
        assert trend["has_previous"] is True
        assert trend["trends"]["frr"]["direction"] == "stable"


class TestThresholdProfileSiteFallbackEmptyHit(TestCase):
    def test_get_threshold_for_group_fallback_site_has_no_profile(self):
        ThresholdProfile.objects.create(
            name="default_profile", distance_threshold=0.35, is_default=True
        )
        # fallback_site="site1" is provided but site1 has no profile.
        # So it returns None, and then goes to default profile logic
        assert (
            ThresholdProfile.get_threshold_for_group(
                ThresholdProfile.GroupType.ROLE, "admin", fallback_site="site1"
            )
            == 0.35
        )


class TestThresholdProfileDefaultsFallbackHitEmpty2(TestCase):
    @override_settings(RECOGNITION_DISTANCE_THRESHOLD=0.6)
    def test_get_threshold_for_group_no_group_or_site_matches_no_default(self):
        # We pass fallback_site="unknown_site" -> profile is not found, default profile does not exist
        # Falls back to settings.RECOGNITION_DISTANCE_THRESHOLD
        assert (
            ThresholdProfile.get_threshold_for_group(
                ThresholdProfile.GroupType.ROLE, "admin", fallback_site="unknown_site"
            )
            == 0.6
        )


class TestThresholdProfileSiteFallbackEmptyHit4(TestCase):
    def test_get_threshold_for_group_no_site_but_default_exists(self):
        ThresholdProfile.objects.create(
            name="default_profile", distance_threshold=0.35, is_default=True
        )
        # fallback_site="" -> skips the if fallback_site block
        # falls through to the Use default profile or system setting block
        assert (
            ThresholdProfile.get_threshold_for_group(
                ThresholdProfile.GroupType.ROLE, "admin", fallback_site=""
            )
            == 0.35
        )

    @override_settings(RECOGNITION_DISTANCE_THRESHOLD=0.6)
    def test_get_threshold_for_group_no_site_and_no_default(self):
        # fallback_site="" -> skips the if fallback_site block
        # falls through to the Use default profile or system setting block
        assert (
            ThresholdProfile.get_threshold_for_group(
                ThresholdProfile.GroupType.ROLE, "admin", fallback_site=""
            )
            == 0.6
        )
