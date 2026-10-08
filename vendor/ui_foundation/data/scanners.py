from __future__ import annotations

import csv
import json
import logging
import multiprocessing
import os
import re
from abc import ABC, abstractmethod
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ui_foundation.config import DataConfig
from ui_foundation.data.image_metadata import ImageMetadataReader
from ui_foundation.data.records import ScreenRecord
from ui_foundation.data.splits import PackageSplitAssigner

LOGGER = logging.getLogger(__name__)
_PACKAGE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+){1,}$")
_XML_PACKAGE_PATTERN = re.compile(rb"package=[\'\"]([^\'\"]+)[\'\"]")
_SYSTEM_PACKAGE_PREFIXES = (
    "android",
    "com.android.",
    "com.google.android.permissioncontroller",
)


class DatasetScanner(ABC):
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.root = Path(config.root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root must be an extracted directory: {self.root}")
        self.split_assigner = PackageSplitAssigner(
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
            seed=config.split_seed,
        )
        self.metadata_reader = ImageMetadataReader()

    def scan(self) -> Iterator[ScreenRecord]:
        emitted = 0
        for record in self._scan_records():
            split = self.split_assigner.assign(record.package_name)
            yield replace(record, split=split)
            emitted += 1
            if self.config.max_records is not None and emitted >= self.config.max_records:
                return

    def image_metadata(self, path: Path) -> tuple[int, int, str]:
        width = 0
        height = 0
        sha256 = ""
        if self.config.scan_image_metadata:
            try:
                width, height = self.metadata_reader.read_size(path)
            except Exception as exc:
                LOGGER.warning("Could not read image dimensions for %s: %s", path, exc)
        if self.config.scan_hashes:
            try:
                sha256 = self.metadata_reader.sha256(path)
            except Exception as exc:
                LOGGER.warning("Could not hash image %s: %s", path, exc)
        return width, height, sha256

    @abstractmethod
    def _scan_records(self) -> Iterable[ScreenRecord]:
        raise NotImplementedError


def walk_files_once(root: Path) -> Iterator[tuple[Path, list[str]]]:
    for directory, _, file_names in os.walk(root, followlinks=False):
        yield Path(directory), file_names


class MoGUIDatasetScanner(DatasetScanner):
    def _scan_records(self) -> Iterator[ScreenRecord]:
        for directory, file_names in walk_files_once(self.root):
            names = set(file_names)
            for file_name in sorted(file_names):
                if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    continue
                image_path = directory / file_name
                xml_path = image_path.with_suffix(".xml")
                if xml_path.name not in names:
                    continue
                width, height, sha256 = self.image_metadata(image_path)
                relative_path = image_path.relative_to(self.root)
                path_package = infer_mogui_path_package(image_path, self.root)
                xml_package = (
                    infer_xml_package(xml_path)
                    if path_package == "" or self.config.scan_xml_package_mismatch
                    else ""
                )
                package_name = path_package or xml_package or image_path.parent.name or "unknown.mogui"
                is_external = (
                    path_package != ""
                    and xml_package != ""
                    and path_package != xml_package
                )
                yield ScreenRecord(
                    record_id=f"mogui:{relative_path.as_posix()}",
                    source="mogui",
                    image_path=str(image_path.resolve()),
                    xml_path=str(xml_path.resolve()),
                    package_name=package_name,
                    trace_id=str(image_path.parent.relative_to(self.root)),
                    is_external=is_external,
                    image_width=width,
                    image_height=height,
                    image_sha256=sha256,
                )


class MobileViewsDatasetScanner(DatasetScanner):
    def _scan_records(self) -> Iterator[ScreenRecord]:
        traces = iter(discover_mobileviews_traces(self.root))
        workers = self.config.mobileviews_scan_workers
        if workers < 1:
            raise ValueError("data.mobileviews_scan_workers must be at least 1")
        LOGGER.info("MobileViews native/legacy reader: %d scan workers; full screenshots only", workers)
        if workers == 1:
            for trace in traces:
                yield from scan_mobileviews_trace(self.config, trace)
            return
        # Bound queued per-app work and preserve deterministic ordering. Spawn
        # avoids forking an already initialized CUDA runtime in evaluate.py.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = deque()
            try:
                for _ in range(workers * 2):
                    trace = next(traces, None)
                    if trace is None:
                        break
                    pending.append(pool.submit(scan_mobileviews_trace, self.config, trace))
                while pending:
                    yield from pending.popleft().result()
                    trace = next(traces, None)
                    if trace is not None:
                        pending.append(pool.submit(scan_mobileviews_trace, self.config, trace))
            finally:
                for future in pending:
                    future.cancel()

    def _scan_trace(self, mapping_path: Path) -> Iterator[ScreenRecord]:
        from ui_foundation.data.mobileviews import is_full_screenshot

        rows = self._read_mapping_rows(mapping_path)
        resolved_rows: list[tuple[int, dict[str, str], Path, Path, dict]] = []
        foreground_packages: Counter[str] = Counter()
        for sequence_index, row in enumerate(rows):
            image_path = self._resolve_reference(mapping_path.parent, row.get("screen_id", ""), "screen")
            state_path = self._resolve_reference(mapping_path.parent, row.get("vh_json_id", ""), "state")
            if image_path is None or state_path is None:
                continue
            if "views" in Path(row.get("screen_id", "")).parts or not is_full_screenshot(image_path, self.root):
                LOGGER.warning("Skipping non-full-screen MobileViews mapping: %s", image_path)
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("Skipping unreadable state JSON %s: %s", state_path, exc)
                continue
            foreground_activity = str(state.get("foreground_activity") or "")
            foreground_package, _ = canonicalize_activity(foreground_activity)
            if foreground_package != "":
                foreground_packages[foreground_package] += 1
            resolved_rows.append((sequence_index, row, image_path, state_path, state))

        app_package = infer_trace_package(foreground_packages)
        if app_package == "":
            LOGGER.warning("Could not infer app package for trace %s", mapping_path.parent)
            return

        trace_id = str(mapping_path.parent.relative_to(self.root))
        seen_state_ids: set[str] = set()
        for sequence_index, row, image_path, state_path, state in resolved_rows:
            del state_path
            foreground_activity = str(state.get("foreground_activity") or "")
            foreground_package, activity_name = canonicalize_activity(foreground_activity)
            if foreground_package == "" or activity_name == "":
                continue
            is_external = foreground_package != app_package
            if is_external and not self.config.include_external:
                continue
            state_id = str(state.get("state_str") or row.get("state_str") or "")
            if self.config.deduplicate_by_state and state_id != "" and state_id in seen_state_ids:
                continue
            if state_id != "":
                seen_state_ids.add(state_id)
            structure_id = str(
                state.get("state_str_content_free") or row.get("structure_str") or ""
            )
            xml_path = self._resolve_reference(mapping_path.parent, row.get("vh_xml_id", ""), "window_dump")
            width, height, sha256 = self.image_metadata(image_path)
            relative_image = image_path.relative_to(self.root)
            yield ScreenRecord(
                record_id=f"mobileviews:{relative_image.as_posix()}",
                source="mobileviews",
                image_path=str(image_path.resolve()),
                xml_path=str(xml_path.resolve()) if xml_path is not None else "",
                package_name=app_package,
                activity_name=f"{foreground_package}/{activity_name}",
                structure_id=structure_id,
                state_id=state_id,
                trace_id=trace_id,
                is_external=is_external,
                sequence_index=sequence_index,
                image_width=width,
                image_height=height,
                image_sha256=sha256,
            )

    @staticmethod
    def _read_mapping_rows(mapping_path: Path) -> list[dict[str, str]]:
        with mapping_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _resolve_reference(self, trace_dir: Path, raw_reference: str, fallback_prefix: str) -> Optional[Path]:
        if raw_reference == "":
            return None
        reference = Path(raw_reference)
        candidates: list[Path] = []
        if reference.is_absolute():
            candidates.append(reference)
        else:
            candidates.extend(
                [
                    self.root / reference,
                    trace_dir / reference,
                    trace_dir.parent / reference,
                    trace_dir / "states" / reference.name,
                ]
            )
            if len(reference.parts) > 1 and reference.parts[0] == trace_dir.name:
                candidates.append(trace_dir / Path(*reference.parts[1:]))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        suffix = reference.suffix
        stem = reference.stem
        if stem.startswith(fallback_prefix):
            fallback = trace_dir / "states" / f"{stem}{suffix}"
            if fallback.is_file():
                return fallback
        return None


def discover_mobileviews_traces(root: Path) -> Iterator[Path]:
    """Discover metadata without walking states/, views/, or web assets."""
    found = False
    for directory, children, files in os.walk(root, followlinks=False):
        has_states = "states" in children
        children[:] = sorted(
            name for name in children
            if name not in {"states", "views", "stylesheets", "scripts", "__MACOSX"}
            and not name.startswith(".")
        )
        path = Path(directory)
        # Prefer the native export if both layouts coexist: never double-count.
        if "utg.js" in files:
            found = True
            children[:] = []
            yield path / "utg.js"
        elif "screenshot_state_mapping.csv" in files:
            found = True
            children[:] = []
            yield path / "screenshot_state_mapping.csv"
        elif has_states and "actions.csv" in files:
            found = True
            children[:] = []
            yield path / "utg.js"  # Missing graph: use original state pairs.
    if not found:
        raise FileNotFoundError(
            f"No MobileViews traces below {root}: expected utg.js or states/ + actions.csv "
            "with states/screen_* images, or screenshot_state_mapping.csv."
        )


def scan_mobileviews_trace(config: DataConfig, path: Path) -> list[ScreenRecord]:
    """Picklable worker entry point; dataset files are read-only."""
    try:
        if path.name == "utg.js":
            from ui_foundation.data.mobileviews import read_graph_trace

            return list(read_graph_trace(config, path))
        return list(MobileViewsDatasetScanner(config)._scan_trace(path))
    except Exception as exc:
        raise ValueError(f"Cannot read MobileViews trace {path}: {exc}") from exc


def infer_mogui_path_package(image_path: Path, root: Path) -> str:
    relative = image_path.relative_to(root)
    for part in reversed(relative.parts[:-1]):
        candidate = part.strip()
        if _PACKAGE_PATTERN.fullmatch(candidate) is not None:
            return candidate
    return ""


def infer_xml_package(xml_path: Path, read_limit: int = 262144) -> str:
    try:
        with xml_path.open("rb") as handle:
            prefix = handle.read(read_limit)
    except OSError:
        return ""
    values = [match.decode("utf-8", errors="ignore") for match in _XML_PACKAGE_PATTERN.findall(prefix)]
    values = [value for value in values if _PACKAGE_PATTERN.fullmatch(value) is not None]
    if len(values) == 0:
        return ""
    counts = Counter(values)
    non_system = Counter(
        {package: count for package, count in counts.items() if not package.startswith(_SYSTEM_PACKAGE_PREFIXES)}
    )
    source = non_system if len(non_system) > 0 else counts
    return source.most_common(1)[0][0]


def canonicalize_activity(raw_value: str) -> tuple[str, str]:
    if raw_value == "" or "/" not in raw_value:
        return "", ""
    package_name, component = raw_value.split("/", 1)
    package_name = package_name.strip()
    component = component.strip()
    if package_name == "" or component == "":
        return "", ""
    if component.startswith("."):
        component = package_name + component
    return package_name, component


def infer_trace_package(packages: Counter[str]) -> str:
    if len(packages) == 0:
        return ""
    non_system = Counter(
        {
            package: count
            for package, count in packages.items()
            if not package.startswith(_SYSTEM_PACKAGE_PREFIXES)
        }
    )
    source = non_system if len(non_system) > 0 else packages
    return source.most_common(1)[0][0]


def build_scanner(config: DataConfig) -> DatasetScanner:
    if config.dataset == "mogui":
        return MoGUIDatasetScanner(config)
    if config.dataset == "mobileviews":
        return MobileViewsDatasetScanner(config)
    raise ValueError(f"Unsupported dataset scanner: {config.dataset}")
