"""Re-encrypt stored artifacts with a fresh Fernet key set."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterable

from django.conf import settings
from django.core.cache import cache
from django.core.management.base import BaseCommand, CommandError

from src.common.crypto import FaceDataEncryption, InvalidToken, _FernetWrapper


class Command(BaseCommand):
    """Rotate encrypted artifacts to a new Fernet key without downtime."""

    help = (
        "Re-encrypt stored recognition datasets, models, and encodings using a new "
        "Fernet key. Use during key rotation after staging the new key alongside the "
        "existing one."
    )

    def add_arguments(self, parser) -> None:  # pragma: no cover - argparse wiring
        parser.add_argument(
            "--new-data-key",
            required=True,
            help="New base64 Fernet key for general data encryption.",
        )
        parser.add_argument(
            "--new-face-key",
            required=True,
            help="New base64 Fernet key for facial encoding encryption.",
        )
        parser.add_argument(
            "--data-root",
            type=Path,
            default=Path(settings.BASE_DIR) / "face_recognition_data",
            help="Path to encrypted model artifacts (defaults to face_recognition_data).",
        )
        parser.add_argument(
            "--dataset-root",
            type=Path,
            default=Path(settings.BASE_DIR) / "face_recognition_data" / "training_dataset",
            help="Path to encrypted dataset captures.",
        )
        parser.add_argument(
            "--backup-dir",
            type=Path,
            help=(
                "Optional directory to write backups of data-root and dataset-root "
                "before mutation."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List files that would be re-encrypted without writing changes.",
        )

    def handle(self, *args, **options) -> None:
        data_root: Path = options["data_root"].resolve()
        dataset_root: Path = options["dataset_root"].resolve()
        new_data_key: str = options["new_data_key"]
        new_face_key: str = options["new_face_key"]
        backup_dir: Path | None = options.get("backup_dir")
        dry_run: bool = options["dry_run"]

        if not data_root.exists():
            raise CommandError(f"data-root does not exist: {data_root}")
        if not dataset_root.exists():
            raise CommandError(f"dataset-root does not exist: {dataset_root}")

        if backup_dir:
            backup_dir.mkdir(parents=True, exist_ok=True)
            self.stdout.write(self.style.NOTICE(f"Backing up data to {backup_dir}"))
            self._backup_tree(data_root, backup_dir / "data_root_backup")
            self._backup_tree(dataset_root, backup_dir / "dataset_root_backup")

        data_decryptor = _FernetWrapper("DATA_ENCRYPTION_KEY")
        face_decryptor = FaceDataEncryption()
        data_encryptor = _FernetWrapper("DATA_ENCRYPTION_KEY", key_override=new_data_key)
        face_encryptor = FaceDataEncryption(key=new_face_key)

        dataset_files = list(self._iter_files(dataset_root))
        encodings_root = data_root / "encodings"
        face_files = list(self._iter_files(encodings_root, suffix=".enc"))
        model_files = [
            path
            for path in self._iter_files(data_root)
            if not path.is_relative_to(encodings_root) and not path.is_relative_to(dataset_root)
        ]

        self.stdout.write(
            self.style.NOTICE(
                "Found "
                f"{len(dataset_files)} dataset files, {len(face_files)} encoding files, "
                f"and {len(model_files)} model artifacts to re-encrypt."
            )
        )

        reencrypted = 0
        for path in dataset_files:
            reencrypted += self._reencrypt_file(
                path, data_decryptor.decrypt, data_encryptor.encrypt, dry_run
            )
        for path in face_files:
            reencrypted += self._reencrypt_file(
                path, face_decryptor.decrypt, face_encryptor.encrypt, dry_run
            )
        for path in model_files:
            reencrypted += self._reencrypt_file(
                path, data_decryptor.decrypt, data_encryptor.encrypt, dry_run
            )

        if dry_run:
            self.stdout.write(self.style.SUCCESS("Dry-run complete; no files modified."))
            return

        self.stdout.write(self.style.SUCCESS(f"Re-encrypted {reencrypted} files."))

        # Invalidate dataset health cache as file metadata has changed
        cache.delete("recognition:health:dataset_snapshot")
        cache.delete("recognition:dataset_state")
        self.stdout.write(self.style.NOTICE("Invalidated dataset health cache."))

    def _backup_tree(self, source: Path, destination: Path) -> None:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

    def _iter_files(self, root: Path, suffix: str | None = None) -> Iterable[Path]:
        if not root.exists():
            return []
        if root.is_file():
            return [root]
        files: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if suffix and path.suffix != suffix:
                continue
            files.append(path)
        return files

    def _reencrypt_file(
        self,
        path: Path,
        decryptor: Callable[[bytes], bytes],
        encryptor: Callable[[bytes], bytes],
        dry_run: bool,
    ) -> int:
        try:
            payload = path.read_bytes()
            decrypted = decryptor(payload)
            updated = encryptor(decrypted)
        except InvalidToken as exc:
            raise CommandError(f"Failed to decrypt {path} with the current key: {exc}") from exc
        except OSError as exc:  # pragma: no cover - filesystem errors
            raise CommandError(f"Unable to process {path}: {exc}") from exc

        if dry_run:
            self.stdout.write(f"Would rotate: {path}")
            return 0

        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_bytes(updated)
        tmp_path.replace(path)
        return 1
