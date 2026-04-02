import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.utils import timezone

import numpy as np
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system_facial_recognition.settings")

import django  # noqa: E402

# Only setup Django if it hasn't been configured yet (e.g., running standalone)
if not django.apps.apps.ready:
    django.setup()

from recognition.tasks import (  # noqa: E402
    TrainingPreconditionError,
    _iter_all_employee_encodings,
    capture_dataset,
    capture_dataset_sync,
    compute_face_encoding,
    incremental_face_training,
    load_existing_encodings,
    process_attendance_batch,
    save_employee_encodings,
    train_model_sync,
    train_recognition_model,
)
from src.common import InvalidToken  # noqa: E402
from users.models import Direction, Present, RecognitionAttempt, Time  # noqa: E402

# Mark tests as slow since they use Celery eager mode with DB transactions
pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.mark.django_db(transaction=True)
def test_process_attendance_batch_creates_records(settings, django_user_model):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    username = "celery-user"
    user = django_user_model.objects.create_user(username=username, password="pass1234")

    attempt_in = RecognitionAttempt.objects.create(
        username=username,
        direction=Direction.IN,
        site="lab",
        source="celery-test",
        successful=True,
    )
    attempt_out = RecognitionAttempt.objects.create(
        username=username,
        direction=Direction.OUT,
        site="lab",
        source="celery-test",
        successful=True,
    )

    records = [
        {
            "direction": "in",
            "present": {username: True},
            "attempt_ids": {username: attempt_in.id},
        },
        {
            "direction": "out",
            "present": {username: True},
            "attempt_ids": {username: attempt_out.id},
        },
    ]

    async_result = process_attendance_batch.delay(records)
    payload = async_result.get(timeout=5)

    assert payload["total"] == 2
    assert len(payload["results"]) == 2
    assert all(entry["status"] == "success" for entry in payload["results"])

    today = timezone.localdate()
    present_record = Present.objects.get(user=user, date=today)
    assert present_record.present is True

    times = Time.objects.filter(user=user, date=today).order_by("time")
    assert times.count() == 2
    assert times.first().direction == Direction.IN
    assert times.last().direction == Direction.OUT

    attempt_in.refresh_from_db()
    attempt_out.refresh_from_db()
    assert attempt_in.user == user
    assert attempt_in.time_record is not None
    assert attempt_out.user == user
    assert attempt_out.time_record is not None


@pytest.fixture
def mock_settings():
    with patch("recognition.tasks.settings") as mock:
        mock.RECOGNITION_HEADLESS_DATASET_FRAMES = 5
        mock.RECOGNITION_HEADLESS_FRAME_SLEEP = 0.001
        yield mock


@pytest.fixture
def mock_cache():
    with patch("recognition.tasks.cache") as mock:
        yield mock


@pytest.fixture
def mock_video_stream():
    with patch("recognition.tasks.VideoStream") as mock_vs:
        stream_instance = MagicMock()

        # Create a dummy frame (BGR format like OpenCV uses)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Make read() return frames then None to simulate end
        frames = [dummy_frame] * 10 + [None]
        stream_instance.read.side_effect = frames

        mock_vs.return_value.start.return_value = stream_instance
        yield stream_instance


@pytest.fixture
def mock_cv2():
    with patch("recognition.tasks.cv2") as mock:
        # Successful encoding
        mock.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        yield mock


@pytest.fixture
def mock_imutils():
    with patch("recognition.tasks.imutils") as mock:
        mock.resize.side_effect = lambda frame, width: frame
        yield mock


def test_load_existing_encodings_not_found(tmp_path):
    with patch("recognition.tasks.ENCODINGS_DIR", tmp_path):
        result = load_existing_encodings("missing_user")
        assert result.size == 0
        assert result.shape == (0, 0)


def test_load_existing_encodings_success(tmp_path):
    user_dir = tmp_path / "testuser"
    user_dir.mkdir(parents=True)
    encoding_path = user_dir / "encodings.npy.enc"

    # Create fake unencrypted array data
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    buffer = io.BytesIO()
    np.save(buffer, arr)

    with patch("recognition.tasks.ENCODINGS_DIR", tmp_path):
        with patch("recognition.tasks.decrypt_face_bytes", return_value=buffer.getvalue()):
            encoding_path.write_bytes(b"fake encrypted data")
            result = load_existing_encodings("testuser")
            assert result.shape == (2, 2)
            np.testing.assert_array_equal(result, arr)


def test_load_existing_encodings_invalid_token(tmp_path):
    user_dir = tmp_path / "testuser"
    user_dir.mkdir(parents=True)
    encoding_path = user_dir / "encodings.npy.enc"

    with patch("recognition.tasks.ENCODINGS_DIR", tmp_path):
        with patch("recognition.tasks.decrypt_face_bytes", side_effect=InvalidToken()):
            encoding_path.write_bytes(b"fake encrypted data")
            result = load_existing_encodings("testuser")
            assert result.size == 0


def test_compute_face_encoding():
    with patch("recognition.tasks._get_face_recognition_model", return_value="Facenet"):
        with patch("recognition.tasks._get_face_detection_backend", return_value="ssd"):
            with patch("recognition.tasks._should_enforce_detection", return_value=True):
                # Test successful retrieval
                with patch(
                    "recognition.tasks._get_or_compute_cached_embedding", return_value=[1.0, 2.0]
                ):
                    result = compute_face_encoding(Path("test.jpg"))
                    assert isinstance(result, np.ndarray)
                    np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

                # Test None return
                with patch("recognition.tasks._get_or_compute_cached_embedding", return_value=None):
                    result = compute_face_encoding(Path("test.jpg"))
                    assert result is None


def test_save_employee_encodings(tmp_path):
    with patch("recognition.tasks.ENCODINGS_DIR", tmp_path):
        with patch("recognition.tasks.encrypt_face_bytes", return_value=b"encrypted"):
            encodings = [[1.0, 2.0], [3.0, 4.0]]
            save_employee_encodings("testuser", encodings)

            # Check if directory and file were created
            encoding_path = tmp_path / "testuser" / "encodings.npy.enc"
            assert encoding_path.exists()
            assert encoding_path.read_bytes() == b"encrypted"


def test_iter_all_employee_encodings(tmp_path):
    with patch("recognition.tasks.ENCODINGS_DIR", tmp_path):
        # Empty dir
        results = list(_iter_all_employee_encodings())
        assert len(results) == 0

        # Setup dummy data
        user1_dir = tmp_path / "user1"
        user1_dir.mkdir()
        user2_dir = tmp_path / "user2"
        user2_dir.mkdir()

        # Mock load_existing_encodings
        def mock_load(emp_id):
            if emp_id == "user1":
                return np.array([[1.0, 2.0]])
            return np.array([])  # user2 has empty encodings

        with patch("recognition.tasks.load_existing_encodings", side_effect=mock_load):
            results = list(_iter_all_employee_encodings())
            assert len(results) == 1
            assert results[0][0] == "user1"
            np.testing.assert_array_equal(results[0][1], np.array([[1.0, 2.0]]))


def test_capture_dataset_sync(
    mock_settings, mock_cache, mock_video_stream, mock_cv2, mock_imutils, tmp_path
):
    with patch("recognition.tasks.TRAINING_DATASET_ROOT", tmp_path):
        with patch("recognition.tasks.encrypt_bytes", return_value=b"encrypted"):
            with patch("recognition.tasks.incremental_face_training.delay") as mock_delay:
                # Run sync capture
                result = capture_dataset_sync("testuser", max_frames=2, enqueue_training=True)

                assert result["username"] == "testuser"
                assert result["frames_captured"] == 2
                assert result["images_saved"] == 2

                # Check files were created
                user_dir = tmp_path / "testuser"
                assert (user_dir / "1.jpg").exists()
                assert (user_dir / "2.jpg").exists()

                # Check cache cleared
                assert mock_cache.delete.call_count >= 2

                # Check task was delayed
                mock_delay.assert_called_once()
                args = mock_delay.call_args[0]
                assert args[0] == "testuser"
                assert len(args[1]) == 2
                assert "1.jpg" in args[1][0]


def test_capture_dataset_celery_task():
    with patch("recognition.tasks.capture_dataset_sync") as mock_sync:
        mock_sync.return_value = {"username": "testuser", "images_saved": 5}

        # We patch the celery task's `update_state` directly since we're testing it as a regular function
        # The celery `bind=True` makes `self` the first arg, so we mock it

        # Test successful run
        with patch.object(capture_dataset, "update_state"):
            result = capture_dataset("testuser")

            assert result["images_saved"] == 5

            # Test exception handling
            mock_sync.side_effect = Exception("Capture failed")
            with pytest.raises(Exception, match="Capture failed"):
                capture_dataset("testuser")


def test_train_model_sync_no_data(tmp_path):
    with patch("recognition.tasks.TRAINING_DATASET_ROOT", tmp_path):
        with pytest.raises(TrainingPreconditionError, match="No training data found"):
            train_model_sync()


def test_train_model_sync_no_usable_data(tmp_path):
    # Setup dummy files
    user1_dir = tmp_path / "user1"
    user1_dir.mkdir(parents=True)
    (user1_dir / "1.jpg").touch()

    with patch("recognition.tasks.TRAINING_DATASET_ROOT", tmp_path):
        # Mock embedding returning None
        with patch("recognition.tasks._get_or_compute_cached_embedding", return_value=None):
            with pytest.raises(TrainingPreconditionError, match="No usable training data found"):
                train_model_sync()


def test_train_model_sync_success(tmp_path):
    # Setup dummy files (more data to avoid stratify issues)
    user1_dir = tmp_path / "user1"
    user1_dir.mkdir(parents=True)
    for i in range(5):
        (user1_dir / f"{i}.jpg").touch()

    user2_dir = tmp_path / "user2"
    user2_dir.mkdir(parents=True)
    for i in range(5):
        (user2_dir / f"{i}.jpg").touch()

    with patch("recognition.tasks.TRAINING_DATASET_ROOT", tmp_path):
        with patch("recognition.tasks.DATA_ROOT", tmp_path):
            with patch("recognition.tasks._get_or_compute_cached_embedding") as mock_emb:
                # Return dummy embeddings based on user
                def mock_embedding(path, *args, **kwargs):
                    if "user1" in str(path):
                        return np.array([1.0, 0.0])
                    return np.array([0.0, 1.0])

                mock_emb.side_effect = mock_embedding

                with patch("recognition.tasks.encrypt_bytes", side_effect=lambda x: b"enc_" + x):
                    with patch("recognition.tasks._dataset_embedding_cache"):
                        with patch("recognition.tasks.embedding_cache"):
                            # Mock FAISS index
                            with patch("recognition.faiss_index.FAISSIndex"):
                                result = train_model_sync()

                                assert result["accuracy"] >= 0.0
                                assert len(result["unique_classes"]) == 2
                                assert "user1" in result["unique_classes"]
                                assert "user2" in result["unique_classes"]

                                # Verify outputs
                                assert (tmp_path / "svc.sav").exists()
                                assert (tmp_path / "classes.npy").exists()
                                assert (tmp_path / "classification_report.txt").exists()


def test_train_recognition_model():
    with patch("recognition.tasks.train_model_sync") as mock_sync:
        mock_sync.return_value = {"accuracy": 0.99}

        with patch.object(train_recognition_model, "update_state"):
            result = train_recognition_model(initiated_by="admin")

            assert result["accuracy"] == 0.99

            # Test precondition error
            mock_sync.side_effect = TrainingPreconditionError("missing")
            with pytest.raises(TrainingPreconditionError):
                train_recognition_model()


def test_incremental_face_training(tmp_path):

    with patch.object(incremental_face_training, "update_state"):
        # Test empty images
        result = incremental_face_training("user1", [])
        assert result["status"] == "skipped"
        assert result["images_provided"] == 0

        # Test valid training
        with patch("recognition.tasks.compute_face_encoding", return_value=np.array([1.0, 2.0])):
            with patch(
                "recognition.tasks.load_existing_encodings", return_value=np.array([[3.0, 4.0]])
            ):
                with patch("recognition.tasks.save_employee_encodings") as mock_save:
                    # Mock iter_all returning 2 classes to avoid insufficient-classes error
                    mock_iter = [
                        ("user1", np.array([[1.0, 2.0], [3.0, 4.0]])),
                        ("user2", np.array([[5.0, 6.0]])),
                    ]
                    with patch(
                        "recognition.tasks._iter_all_employee_encodings", return_value=mock_iter
                    ):
                        with patch("recognition.tasks._train_classifier") as mock_train:
                            mock_classifier = MagicMock()
                            mock_train.return_value = mock_classifier
                            with patch("recognition.tasks._persist_model"):
                                with patch("recognition.faiss_index.FAISSIndex"):
                                    with patch("recognition.tasks._dataset_embedding_cache"):
                                        with patch("recognition.tasks.embedding_cache"):
                                            result = incremental_face_training(
                                                "user1", ["img1.jpg"]
                                            )

                                            assert result["status"] == "trained"
                                            assert result["images_provided"] == 1
                                            assert result["encodings_total"] == 2
                                            assert len(result["classes"]) == 2

                                            # Verify save was called with combined array
                                            mock_save.assert_called_once()
                                            saved_user, saved_arr = mock_save.call_args[0]
                                            assert saved_user == "user1"
                                            assert saved_arr.shape == (2, 2)
