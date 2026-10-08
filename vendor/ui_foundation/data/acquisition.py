from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import stat
import tarfile
import time
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Optional

from tqdm.auto import tqdm

from ui_foundation.config import ExperimentConfig

LOGGER = logging.getLogger(__name__)

SnapshotDownload = Callable[..., str]


class DisabledTqdm(tqdm):
    """Per-call tqdm class for suppressing Hugging Face Hub progress output."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


class ArchiveIntegrityError(RuntimeError):
    """Raised when an archive fails checksum or structural verification."""


class PartialArchiveError(tarfile.ReadError):
    """A TAR stream ended after yielding zero or more complete regular members."""

    def __init__(
        self,
        message: str,
        *,
        completed_members: int,
        reused_members: int,
        incomplete_member: Optional[str],
    ) -> None:
        super().__init__(message)
        self.completed_members = completed_members
        self.reused_members = reused_members
        self.incomplete_member = incomplete_member


_ARCHIVE_CORRUPTION_ERRORS = (
    EOFError,
    gzip.BadGzipFile,
    tarfile.CompressionError,
    tarfile.ReadError,
    zipfile.BadZipFile,
)


class DatasetAcquirer(ABC):
    """Materialize a dataset as ordinary files on local storage."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @abstractmethod
    def materialize(self) -> Path:
        raise NotImplementedError


class LocalDatasetAcquirer(DatasetAcquirer):
    def materialize(self) -> Path:
        root = Path(self.config.data.root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Local dataset root does not exist: {root}")
        return root


class HuggingFaceDatasetAcquirer(DatasetAcquirer):
    """Download a complete Hugging Face repository snapshot, then extract archives locally.

    This class deliberately uses ``snapshot_download`` instead of the datasets streaming API.
    Training only begins after the selected files are present on disk.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        snapshot_download_fn: Optional[SnapshotDownload] = None,
    ) -> None:
        super().__init__(config)
        self._snapshot_download_fn = snapshot_download_fn

    def materialize(self) -> Path:
        download_dir = self._download_dir()
        extraction_dir = self._extraction_dir()
        download_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "Downloading Hugging Face dataset %s at revision %s to %s",
            self.config.data.hf_repo_id,
            self.config.data.hf_revision,
            download_dir,
        )
        snapshot_download = self._snapshot_download_fn or self._load_snapshot_download()
        token = os.environ.get(self.config.data.hf_token_env) or None
        snapshot_path = Path(
            snapshot_download(
                repo_id=self.config.data.hf_repo_id,
                repo_type="dataset",
                revision=self.config.data.hf_revision,
                local_dir=str(download_dir),
                allow_patterns=self.config.data.hf_allow_patterns or None,
                ignore_patterns=self.config.data.hf_ignore_patterns or None,
                max_workers=self.config.data.hf_max_workers,
                force_download=self.config.data.hf_force_download,
                local_files_only=self.config.data.hf_local_files_only,
                token=token,
                tqdm_class=tqdm if self.config.runtime.progress_bars else DisabledTqdm,
            )
        ).resolve()

        self._write_source_metadata(snapshot_path, download_dir)
        if not self.config.data.extract_archives:
            LOGGER.info("Archive extraction is disabled; dataset root is %s", snapshot_path)
            return snapshot_path

        archives = list(discover_archives(snapshot_path))
        if len(archives) == 0:
            if contains_direct_dataset_files(snapshot_path):
                LOGGER.info(
                    "No archives were found, but direct dataset files exist; using %s",
                    snapshot_path,
                )
                return snapshot_path
            raise FileNotFoundError(
                "The downloaded snapshot contains no supported archives or directly scannable "
                "dataset files. Check data.hf_allow_patterns and data.hf_ignore_patterns."
            )

        extractor = ArchiveExtractor(
            destination=extraction_dir,
            source_root=snapshot_path,
            show_progress=self.config.runtime.progress_bars,
            remove_archives=self.config.data.remove_archives_after_extraction,
            verify_archives=self.config.data.verify_archives,
            skip_corrupt_archives=self.config.data.skip_corrupt_archives,
            salvage_truncated_archives=self.config.data.salvage_truncated_archives,
            expected_sha256=load_huggingface_sha256s(snapshot_path, archives),
        )
        extractor.extract_all(archives)
        return extraction_dir.resolve()

    def _download_dir(self) -> Path:
        if self.config.data.hf_download_dir != "":
            return Path(self.config.data.hf_download_dir).expanduser().resolve()
        return (
            Path(self.config.runtime.cache_dir)
            / "datasets"
            / self.config.data.dataset
            / "huggingface"
        ).resolve()

    def _extraction_dir(self) -> Path:
        if self.config.data.root != "":
            return Path(self.config.data.root).expanduser().resolve()
        return (
            Path(self.config.runtime.cache_dir)
            / "datasets"
            / self.config.data.dataset
            / "extracted"
        ).resolve()

    @staticmethod
    def _load_snapshot_download() -> SnapshotDownload:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face dataset acquisition requires huggingface_hub. "
                "Install the project requirements before training."
            ) from exc
        return snapshot_download

    def _write_source_metadata(self, snapshot_path: Path, download_dir: Path) -> None:
        metadata = {
            "source": "huggingface",
            "repo_id": self.config.data.hf_repo_id,
            "requested_revision": self.config.data.hf_revision,
            "snapshot_path": str(snapshot_path),
            "download_dir": str(download_dir),
            "allow_patterns": list(self.config.data.hf_allow_patterns),
            "ignore_patterns": list(self.config.data.hf_ignore_patterns),
            "materialized_unix_time": time.time(),
        }
        path = download_dir / ".ui_place_dataset_source.json"
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


class ArchiveExtractor:
    """Incremental, path-safe extraction for ZIP and TAR-family archives."""

    def __init__(
        self,
        destination: str | Path,
        source_root: str | Path,
        show_progress: bool = True,
        remove_archives: bool = False,
        verify_archives: bool = True,
        skip_corrupt_archives: bool = False,
        salvage_truncated_archives: bool = False,
        expected_sha256: Optional[Mapping[Path, str]] = None,
    ) -> None:
        self.destination = Path(destination).resolve()
        self.source_root = Path(source_root).resolve()
        self.show_progress = show_progress
        self.remove_archives = remove_archives
        self.verify_archives = verify_archives
        self.skip_corrupt_archives = skip_corrupt_archives
        self.salvage_truncated_archives = salvage_truncated_archives
        self.expected_sha256 = {
            Path(path).resolve(): digest.lower()
            for path, digest in (expected_sha256 or {}).items()
        }
        self.marker_dir = self.destination / ".ui_place_extraction"
        self.verification_dir = self.marker_dir / "verification"

    def extract_all(self, archives: Iterable[Path]) -> None:
        archive_list = sorted(Path(path).resolve() for path in archives)
        self.destination.mkdir(parents=True, exist_ok=True)
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        if self.verify_archives:
            self.verification_dir.mkdir(parents=True, exist_ok=True)
            archive_list = self._preflight(archive_list)

        progress = tqdm(
            archive_list,
            desc="extract archives",
            unit="archive",
            disable=not self.show_progress,
            dynamic_ncols=True,
        )
        for archive_path in progress:
            progress.set_postfix_str(archive_path.name)
            if self._is_complete(archive_path):
                LOGGER.info("Already extracted: %s", archive_path)
                continue
            try:
                self._extract_one(archive_path)
            except _ARCHIVE_CORRUPTION_ERRORS as exc:
                if self._can_salvage(archive_path) and isinstance(exc, PartialArchiveError):
                    recovered_members = exc.completed_members + exc.reused_members
                    if recovered_members > 0:
                        self._write_marker(
                            archive_path,
                            outcome="salvaged",
                            recovered_members=recovered_members,
                            newly_extracted_members=exc.completed_members,
                            reused_members=exc.reused_members,
                            incomplete_member=exc.incomplete_member,
                            warning=str(exc),
                        )
                        LOGGER.warning(
                            "Salvaged %d complete member(s) from truncated archive %s; "
                            "discarded incomplete tail member %r",
                            recovered_members,
                            archive_path,
                            exc.incomplete_member,
                        )
                        continue
                if not self.skip_corrupt_archives:
                    raise
                self._mark_extraction_corruption(archive_path, exc)
                self._write_marker(
                    archive_path,
                    outcome="skipped_corrupt",
                    warning=str(exc),
                )
                LOGGER.error(
                    "Skipping archive that failed extraction after verification: %s (%s)",
                    archive_path,
                    exc,
                )
                continue
            self._write_marker(archive_path)
            if self.remove_archives:
                archive_path.unlink(missing_ok=True)

    def _extract_one(self, archive_path: Path) -> None:
        LOGGER.info("Extracting %s to %s", archive_path, self.destination)
        lower_name = archive_path.name.lower()
        if lower_name.endswith(".zip"):
            self._extract_zip(archive_path)
            return
        if lower_name.endswith((".tar", ".tar.gz", ".tgz")):
            self._extract_tar(archive_path)
            return
        raise ValueError(f"Unsupported archive format: {archive_path}")

    def _extract_zip(self, archive_path: Path) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            total_bytes = sum(member.file_size for member in members)
            progress = tqdm(
                total=total_bytes,
                desc=f"extract {archive_path.name}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
                disable=not self.show_progress,
                dynamic_ncols=True,
            )
            with progress:
                for member in members:
                    if is_zip_symlink(member):
                        LOGGER.warning("Skipping ZIP symlink: %s:%s", archive_path, member.filename)
                        progress.update(member.file_size)
                        continue
                    target = self._safe_target(member.filename)
                    if self._existing_file_matches(target, member.file_size):
                        progress.update(member.file_size)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".ui_foundation_partial")
                    try:
                        with archive.open(member, "r") as source, temporary.open("wb") as output:
                            copy_with_progress(source, output, progress)
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                    temporary.replace(target)

    def _extract_tar(self, archive_path: Path) -> None:
        # Stream TAR archives in one pass. Calling getmembers() on multi-gigabyte .tar.gz
        # shards would first decompress the entire archive to index it and then seek back for
        # extraction, roughly doubling decompression work.
        progress = tqdm(
            total=archive_path.stat().st_size,
            desc=f"extract {archive_path.name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            disable=not self.show_progress,
            dynamic_ncols=True,
        )
        completed_members = 0
        reused_members = 0
        incomplete_member: Optional[str] = None
        temporary: Optional[Path] = None
        try:
            with progress, archive_path.open("rb") as compressed_source:
                monitored_source = ProgressReader(compressed_source, progress)
                with tarfile.open(fileobj=monitored_source, mode="r|*") as archive:
                    for member in archive:
                        incomplete_member = member.name
                        if member.isdir():
                            self._safe_target(member.name).mkdir(parents=True, exist_ok=True)
                            incomplete_member = None
                            continue
                        if not member.isfile():
                            LOGGER.warning(
                                "Skipping non-regular TAR member: %s:%s",
                                archive_path,
                                member.name,
                            )
                            incomplete_member = None
                            continue
                        target = self._safe_target(member.name)
                        if self._existing_file_matches(target, member.size):
                            reused_members += 1
                            incomplete_member = None
                            continue
                        source = archive.extractfile(member)
                        if source is None:
                            raise OSError(f"Could not read archive member {member.name!r}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_name(target.name + ".ui_foundation_partial")
                        try:
                            with source, temporary.open("wb") as output:
                                shutil.copyfileobj(source, output, length=1024 * 1024)
                        except BaseException:
                            temporary.unlink(missing_ok=True)
                            temporary = None
                            raise
                        temporary.replace(target)
                        temporary = None
                        completed_members += 1
                        incomplete_member = None
        except _ARCHIVE_CORRUPTION_ERRORS as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise PartialArchiveError(
                f"archive ended before its TAR stream completed: {exc}",
                completed_members=completed_members,
                reused_members=reused_members,
                incomplete_member=incomplete_member,
            ) from exc

    def _preflight(self, archive_list: list[Path]) -> list[Path]:
        verified: list[Path] = []
        skipped: list[Path] = []
        salvageable: list[Path] = []
        progress = tqdm(
            archive_list,
            desc="verify archives",
            unit="archive",
            disable=not self.show_progress,
            dynamic_ncols=True,
        )
        for archive_path in progress:
            progress.set_postfix_str(archive_path.name)
            if self._is_complete(archive_path):
                verified.append(archive_path)
                continue
            result = self._verification_result(archive_path)
            if result.get("status") == "valid":
                verified.append(archive_path)
                continue
            if self._can_salvage(archive_path, result):
                verified.append(archive_path)
                salvageable.append(archive_path)
                LOGGER.warning(
                    "Queueing publisher-truncated archive for partial recovery: %s (%s)",
                    archive_path,
                    result.get("reason", "unknown reason"),
                )
                continue
            if not self.skip_corrupt_archives:
                raise ArchiveIntegrityError(
                    f"Archive integrity verification failed for {archive_path}: "
                    f"{result.get('reason', 'unknown reason')}"
                )
            skipped.append(archive_path)
            LOGGER.error(
                "Skipping corrupt archive during preflight: %s (%s)",
                archive_path,
                result.get("reason", "unknown reason"),
            )
        if skipped:
            LOGGER.warning(
                "Archive preflight skipped %d corrupt archive(s); cached reports are in %s",
                len(skipped),
                self.verification_dir,
            )
        if salvageable:
            LOGGER.warning(
                "Archive preflight queued %d publisher-verified truncated archive(s) "
                "for partial recovery",
                len(salvageable),
            )
        return verified

    def _can_salvage(
        self,
        archive_path: Path,
        verification: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if not self.salvage_truncated_archives:
            return False
        if not archive_path.name.lower().endswith((".tar", ".tar.gz", ".tgz")):
            return False
        result = verification or self._read_verification_marker(archive_path)
        if result is None or result.get("status") != "corrupt":
            return False
        expected_sha256 = self.expected_sha256.get(archive_path)
        if expected_sha256 is None or result.get("actual_sha256") != expected_sha256:
            return False
        failure_kind = result.get("failure_kind")
        return failure_kind == "structure" or (
            failure_kind is None
            and str(result.get("reason", "")).startswith("archive structure is invalid:")
        )

    def _verification_result(self, archive_path: Path) -> dict[str, Any]:
        cached = self._read_verification_marker(archive_path)
        if cached is not None:
            return cached

        expected_sha256 = self.expected_sha256.get(archive_path)
        actual_sha256 = self._sha256(archive_path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            return self._write_verification_marker(
                archive_path,
                status="corrupt",
                expected_sha256=expected_sha256,
                actual_sha256=actual_sha256,
                failure_kind="checksum",
                reason="SHA-256 does not match the Hugging Face download metadata",
            )

        try:
            self._verify_structure(archive_path)
        except _ARCHIVE_CORRUPTION_ERRORS as exc:
            return self._write_verification_marker(
                archive_path,
                status="corrupt",
                expected_sha256=expected_sha256,
                actual_sha256=actual_sha256,
                failure_kind="structure",
                reason=f"archive structure is invalid: {exc}",
            )

        return self._write_verification_marker(
            archive_path,
            status="valid",
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            failure_kind="",
            reason="",
        )

    def _sha256(self, archive_path: Path) -> str:
        digest = hashlib.sha256()
        progress = tqdm(
            total=archive_path.stat().st_size,
            desc=f"hash {archive_path.name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            disable=not self.show_progress,
            dynamic_ncols=True,
        )
        with progress, archive_path.open("rb") as source:
            while chunk := source.read(4 * 1024 * 1024):
                digest.update(chunk)
                progress.update(len(chunk))
        return digest.hexdigest()

    def _verify_structure(self, archive_path: Path) -> None:
        lower_name = archive_path.name.lower()
        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as archive:
                failed_member = archive.testzip()
            if failed_member is not None:
                raise zipfile.BadZipFile(f"CRC check failed for {failed_member!r}")
            return
        if lower_name.endswith((".tar.gz", ".tgz")):
            progress = tqdm(
                total=archive_path.stat().st_size,
                desc=f"check {archive_path.name}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
                disable=not self.show_progress,
                dynamic_ncols=True,
            )
            with progress, archive_path.open("rb") as source:
                monitored = ProgressReader(source, progress)
                with gzip.GzipFile(fileobj=monitored, mode="rb") as decompressed:
                    while decompressed.read(4 * 1024 * 1024):
                        pass
            return
        if lower_name.endswith(".tar"):
            with tarfile.open(archive_path, mode="r:") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        raise tarfile.ReadError(f"could not read member {member.name!r}")
                    with source:
                        while source.read(4 * 1024 * 1024):
                            pass
            return
        raise ValueError(f"Unsupported archive format: {archive_path}")

    def _verification_marker_path(self, archive_path: Path) -> Path:
        return self.verification_dir / self._marker_path(archive_path).name

    def _read_verification_marker(self, archive_path: Path) -> Optional[dict[str, Any]]:
        marker_path = self._verification_marker_path(archive_path)
        if not marker_path.is_file():
            return None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if marker.get("archive") != self._archive_fingerprint(archive_path):
            return None
        if marker.get("expected_sha256") != self.expected_sha256.get(archive_path):
            return None
        return marker

    def _write_verification_marker(
        self,
        archive_path: Path,
        *,
        status: str,
        expected_sha256: Optional[str],
        actual_sha256: str,
        failure_kind: str,
        reason: str,
    ) -> dict[str, Any]:
        marker_path = self._verification_marker_path(archive_path)
        temporary = marker_path.with_suffix(".json.tmp")
        payload = {
            "archive": self._archive_fingerprint(archive_path),
            "status": status,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "failure_kind": failure_kind,
            "reason": reason,
            "verified_unix_time": time.time(),
        }
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(marker_path)
        return payload

    def _mark_extraction_corruption(self, archive_path: Path, exc: BaseException) -> None:
        cached = self._read_verification_marker(archive_path) or {}
        self._write_verification_marker(
            archive_path,
            status="corrupt",
            expected_sha256=self.expected_sha256.get(archive_path),
            actual_sha256=str(cached.get("actual_sha256", "")),
            failure_kind="extraction",
            reason=f"extraction failed after integrity verification: {exc}",
        )

    def _safe_target(self, raw_name: str) -> Path:
        normalized = PurePosixPath(raw_name.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"Unsafe archive member path: {raw_name!r}")
        target = (self.destination / Path(*normalized.parts)).resolve()
        if target != self.destination and self.destination not in target.parents:
            raise ValueError(f"Archive member escapes extraction directory: {raw_name!r}")
        return target

    @staticmethod
    def _existing_file_matches(path: Path, expected_size: int) -> bool:
        try:
            return path.is_file() and path.stat().st_size == expected_size
        except OSError:
            return False

    def _marker_path(self, archive_path: Path) -> Path:
        try:
            relative = archive_path.relative_to(self.source_root).as_posix()
        except ValueError:
            relative = str(archive_path)
        digest = hashlib.blake2b(relative.encode("utf-8"), digest_size=12).hexdigest()
        return self.marker_dir / f"{digest}.json"

    def _archive_fingerprint(self, archive_path: Path) -> dict[str, Any]:
        stat_result = archive_path.stat()
        try:
            relative = archive_path.relative_to(self.source_root).as_posix()
        except ValueError:
            relative = str(archive_path)
        return {
            "relative_path": relative,
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
        }

    def _is_complete(self, archive_path: Path) -> bool:
        marker_path = self._marker_path(archive_path)
        if not marker_path.is_file():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return marker.get("archive") == self._archive_fingerprint(archive_path)

    def _write_marker(
        self,
        archive_path: Path,
        *,
        outcome: str = "complete",
        **details: Any,
    ) -> None:
        marker_path = self._marker_path(archive_path)
        temporary = marker_path.with_suffix(".json.tmp")
        payload = {
            "archive": self._archive_fingerprint(archive_path),
            "outcome": outcome,
            "completed_unix_time": time.time(),
            **details,
        }
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(marker_path)


class DatasetMaterializer:
    """Factory-backed application service that resolves ``data.root``."""

    def __init__(
        self,
        config: ExperimentConfig,
        snapshot_download_fn: Optional[SnapshotDownload] = None,
    ) -> None:
        self.config = config
        self.snapshot_download_fn = snapshot_download_fn

    def prepare(self) -> Path:
        acquirer = self._build_acquirer()
        root = acquirer.materialize().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Materialized dataset root is not a directory: {root}")
        self.config.data.root = str(root)
        LOGGER.info("Materialized dataset root: %s", root)
        return root

    def _build_acquirer(self) -> DatasetAcquirer:
        if self.config.data.source == "local":
            return LocalDatasetAcquirer(self.config)
        if self.config.data.source == "huggingface":
            return HuggingFaceDatasetAcquirer(
                self.config,
                snapshot_download_fn=self.snapshot_download_fn,
            )
        raise ValueError(f"Unsupported data source: {self.config.data.source}")


def discover_archives(root: str | Path) -> Iterable[Path]:
    root_path = Path(root)
    internal_cache = root_path / ".cache"
    for path in root_path.rglob("*"):
        if not path.is_file() or path.is_relative_to(internal_cache):
            continue
        lower_name = path.name.lower()
        if lower_name.endswith(_ARCHIVE_SUFFIXES):
            yield path


def load_huggingface_sha256s(
    snapshot_root: str | Path,
    archives: Iterable[Path],
) -> dict[Path, str]:
    """Read LFS SHA-256 values persisted by huggingface_hub beside downloaded files."""

    root = Path(snapshot_root).resolve()
    metadata_root = root / ".cache" / "huggingface" / "download"
    checksums: dict[Path, str] = {}
    for archive in archives:
        archive_path = Path(archive).resolve()
        try:
            relative = archive_path.relative_to(root)
        except ValueError:
            continue
        metadata_path = metadata_root / relative.parent / f"{relative.name}.metadata"
        try:
            lines = metadata_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if len(lines) < 2:
            continue
        candidate = lines[1].strip().lower()
        if len(candidate) == 64 and all(character in "0123456789abcdef" for character in candidate):
            checksums[archive_path] = candidate
    return checksums


def contains_direct_dataset_files(root: str | Path) -> bool:
    root_path = Path(root)
    probes = (
        next(root_path.rglob("*.png"), None),
        next(root_path.rglob("screenshot_state_mapping.csv"), None),
    )
    return any(path is not None for path in probes)


def is_zip_symlink(member: zipfile.ZipInfo) -> bool:
    mode = member.external_attr >> 16
    return stat.S_ISLNK(mode)


class ProgressReader:
    """File-like proxy that reports compressed bytes consumed by streaming TAR reads."""

    def __init__(self, source, progress) -> None:
        self.source = source
        self.progress = progress

    def read(self, size: int = -1):
        data = self.source.read(size)
        self.progress.update(len(data))
        return data

    def readinto(self, buffer) -> int:
        count = self.source.readinto(buffer)
        if count is None:
            return 0
        self.progress.update(count)
        return count

    def __getattr__(self, name: str):
        return getattr(self.source, name)


def copy_with_progress(source, destination, progress, chunk_size: int = 1024 * 1024) -> None:
    while True:
        chunk = source.read(chunk_size)
        if chunk == b"":
            return
        destination.write(chunk)
        progress.update(len(chunk))
