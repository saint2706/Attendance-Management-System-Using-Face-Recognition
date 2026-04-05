from unittest.mock import patch

from src.common.demo_data import (
    SYNTHETIC_USERS,
    _render_avatar_frame,
    generate_encrypted_dataset,
    sync_demo_dataset,
)


def test_render_avatar_frame():
    # Render frame
    frame_bytes = _render_avatar_frame("test_user", 0, (255, 0, 0))

    # Check if we got bytes out
    assert isinstance(frame_bytes, bytes)
    assert len(frame_bytes) > 0


@patch("src.common.demo_data.encrypt_bytes")
def test_generate_encrypted_dataset(mock_encrypt, tmp_path):
    mock_encrypt.side_effect = lambda x: b"encrypted_" + x

    created_files = generate_encrypted_dataset(tmp_path)

    # Check that the correct number of files was created
    expected_count = len(SYNTHETIC_USERS) * 3  # 3 users, 3 frames each
    assert len(created_files) == expected_count

    for username in SYNTHETIC_USERS:
        user_dir = tmp_path / username
        assert user_dir.exists()
        assert user_dir.is_dir()

        for i in range(3):
            file_path = user_dir / f"{username}_frame_{i + 1:02d}.jpg"
            assert file_path.exists()
            assert file_path in created_files


@patch("src.common.demo_data.generate_encrypted_dataset")
def test_sync_demo_dataset(mock_generate, tmp_path):
    sample_root = tmp_path / "sample"
    training_root = tmp_path / "training"

    # Create some mock output files from generate_encrypted_dataset
    sample_root.mkdir()
    mock_file = sample_root / "test.jpg"
    mock_file.write_bytes(b"test data")

    mock_generate.return_value = [mock_file]

    sync_demo_dataset(sample_root, training_root)

    # Check that directories were created
    assert sample_root.exists()
    assert training_root.exists()

    # Check that generate was called
    mock_generate.assert_called_once_with(sample_root)

    # Check that files were copied
    dest_file = training_root / "test.jpg"
    assert dest_file.exists()
    assert dest_file.read_bytes() == b"test data"
