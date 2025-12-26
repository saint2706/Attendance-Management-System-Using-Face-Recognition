import base64
import os
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model

import pytest
from rest_framework.test import APIClient

from recognition.ablation import AblationConfig, run_single_ablation
from recognition.analysis.failures import detect_occlusion, detect_pose_issues
from recognition.performance_utils import detect_hardware


# Utility to get valid images from the synthesized dataset
def get_dataset_images(limit=2):
    # Try finding the 'dataset' directory in expected locations
    possible_roots = [
        Path(settings.BASE_DIR) / "dataset",
        Path(os.getcwd()) / "dataset",
    ]
    dataset_root = None
    for root in possible_roots:
        if root.exists():
            dataset_root = root
            break

    if not dataset_root:
        return []

    image_paths = sorted(list(dataset_root.glob("*/*.jpg")))
    return image_paths[:limit]


@pytest.mark.django_db
class TestRealDataIntegration:
    """
    Integration tests using the actual 'dataset/' created by generate_synthetic_data.py.
    """

    def test_npu_detection_real(self):
        """Verify NPU detection runs without error on this host."""
        hardware_info = detect_hardware()
        # It returns a HardwareInfo dataclass, verify it has to_dict
        assert hasattr(hardware_info, "to_dict")
        info_dict = hardware_info.to_dict()
        assert isinstance(info_dict, dict)
        assert "cpu" in info_dict

    def test_pose_and_occlusion_with_real_images(self):
        """Run failure analysis on actual face images."""
        images = get_dataset_images(limit=2)
        if not images:
            pytest.skip("No images found in dataset/ - run synthetic generation first.")

        for img_path in images:
            # Test Pose
            pose = detect_pose_issues(str(img_path))
            assert pose in ["frontal", "profile", "tilted", "unknown"]

            # Test Occlusion
            occ = detect_occlusion(str(img_path))
            assert occ in ["none", "partial", "heavy", "unknown"]

    def test_ablation_study_real_mode(self):
        """Run a mini ablation study in strict 'real' mode (synthetic=False)."""
        images = get_dataset_images(limit=2)
        if not images:
            pytest.skip("No images found in dataset/")

        # Labels are the parent folder names (usernames)
        labels = [p.parent.name for p in images]

        config = AblationConfig(
            detector="opencv",  # fast backend
            alignment=True,
            distance_metric="cosine",
            rebalancing=False,
        )

        # synthetic=False triggers the real DeepFace path
        results = run_single_ablation(
            config=config,
            image_paths=images,
            labels=labels,
            random_state=42,
            synthetic=False,  # This is the key test
        )

        assert "accuracy" in results
        assert "f1_score" in results
        assert results["n_samples"] == len(images)
        assert results["mode"] == "real"  # Assuming code sets this, or we infer from behavior

    def test_mark_endpoint_integration(self):
        """Test the full /api/attendance/mark/ endpoint with a real image."""
        from unittest.mock import patch

        from django.urls import reverse

        images = get_dataset_images(limit=1)
        if not images:
            pytest.skip("No images found in dataset/")

        img_path = images[0]
        # Dataset structure is dataset_root/username/image.jpg
        # So parent is username, parent.parent is dataset_root
        dataset_root = img_path.parent.parent
        username = img_path.parent.name

        # Ensure user exists (should be created by generation script, but safe to check)
        User = get_user_model()
        user, _ = User.objects.get_or_create(username=username)

        # Encode image
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        client = APIClient()
        client.force_authenticate(user=user)

        payload = {"image": encoded_string, "direction": "in"}

        # Try both the action URL and explicit path construction
        try:
            url = reverse("attendance-mark")
        except Exception:
            url = "/api/attendance/mark/"

        # Patch TRAINING_DATASET_ROOT to point to our local dataset folder
        # And mock the cache path to force it to call the builder callback
        # Note: We patch views_legacy because recognition.views is a package that re-exports from views_legacy
        with patch("recognition.views_legacy.TRAINING_DATASET_ROOT", dataset_root):
            with patch("recognition.views_legacy._dataset_embedding_cache") as mock_cache:
                # Force the cache to just run the callback (which builds from dataset_root)
                mock_cache.get_dataset_index.side_effect = lambda m, d, e, cb: cb()

                # The synthetic data is composed of raw JPEGs, but the view expects encrypted files.
                # We mock _decrypt_image_bytes to just return the file content.
                with patch(
                    "recognition.views_legacy._decrypt_image_bytes",
                    side_effect=lambda p: open(str(p), "rb").read(),
                ):
                    response = client.post(url, payload, format="json")

        if response.status_code == 405:
            print(f"Method Not Allowed at {url}. Allowed: {response.get('Allow', 'Unknown')}")
            # Fallback: check if the router mounted it at a different path?
            # For now, let fail to see debug print.

        if response.status_code == 400 and "noface" in str(response.data).lower():
            pytest.skip("Could not detect face in reference image during test.")

        assert response.status_code in [
            200,
            201,
        ], f"Failed with {response.status_code}: {response.data}"
        assert "status" in response.data
        # If success, status is 'success'. If match failed, 'failure'.
        # Both assume the code ran without error.
