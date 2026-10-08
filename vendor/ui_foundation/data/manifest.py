from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from tqdm.auto import tqdm

from ui_foundation.config import ExperimentConfig
from ui_foundation.data.mobileviews import MOBILEVIEWS_READER_VERSION
from ui_foundation.data.records import JsonlManifest, ManifestStatistics, ScreenRecord
from ui_foundation.data.scanners import build_scanner

LOGGER = logging.getLogger(__name__)


class ManifestManager:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.path = self._resolve_manifest_path()
        self.manifest = JsonlManifest(self.path)

    def prepare(self) -> Path:
        if self.path.exists() and not self.config.data.rebuild_manifest:
            LOGGER.info("Using existing manifest: %s", self.path)
            self._write_resolution_report(self.manifest.statistics())
            return self.path
        LOGGER.info("Scanning %s below %s", self.config.data.dataset, self.config.data.root)
        scanner = build_scanner(self.config.data)
        records = tqdm(
            scanner.scan(),
            desc=f"scan {self.config.data.dataset}",
            unit="screen",
            disable=not self.config.runtime.progress_bars,
            dynamic_ncols=True,
        )
        statistics = self.manifest.write(records)
        if statistics.total == 0:
            raise RuntimeError("Dataset scan produced no records")
        LOGGER.info(
            "Manifest: %d screens, %d packages, train=%d val=%d test=%d",
            statistics.total,
            len(statistics.packages),
            statistics.count("train"),
            statistics.count("val"),
            statistics.count("test"),
        )
        self._write_resolution_report(statistics)
        return self.path

    def load(self, split: str | None = None, max_records: int | None = None) -> list[ScreenRecord]:
        self.prepare()
        records = self.manifest.read(split=split, max_records=max_records)
        if len(records) == 0:
            raise RuntimeError(f"No records for split={split!r} in {self.path}")
        return records

    def iter_records(self, split: str | None = None, max_records: int | None = None):
        self.prepare()
        return self.manifest.iter_records(split=split, max_records=max_records)

    def _resolve_manifest_path(self) -> Path:
        if self.config.data.manifest != "":
            return Path(self.config.data.manifest).expanduser().resolve()
        root = str(Path(self.config.data.root).resolve())
        if self.config.data.dataset == "mobileviews":
            # Do not reuse pre-native-reader manifests or incompatible filters.
            identity = json.dumps({
                "root": root, "reader": MOBILEVIEWS_READER_VERSION,
                "include_external": self.config.data.include_external,
                "deduplicate_by_state": self.config.data.deduplicate_by_state,
                "scan_image_metadata": self.config.data.scan_image_metadata,
                "scan_hashes": self.config.data.scan_hashes,
                "max_records": self.config.data.max_records,
                "split_seed": self.config.data.split_seed,
                "fractions": [self.config.data.train_fraction, self.config.data.val_fraction,
                              self.config.data.test_fraction],
            }, sort_keys=True)
            digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).hexdigest()
            return Path(self.config.runtime.cache_dir) / "manifests" / f"{MOBILEVIEWS_READER_VERSION}_{digest}.jsonl"
        digest = hashlib.blake2b(root.encode("utf-8"), digest_size=8).hexdigest()
        return Path(self.config.runtime.cache_dir) / "manifests" / f"{self.config.data.dataset}_{digest}.jsonl"

    def _write_resolution_report(self, statistics: ManifestStatistics) -> None:
        report = {
            "dataset": self.config.data.dataset,
            "total": statistics.total,
            "resolutions": statistics.to_dict().get("resolutions", {}),
            "configured_buckets": [
                {
                    "width": bucket.width,
                    "height": bucket.height,
                    "patch_size": self.config.data.patch_size,
                    "patch_tokens": (bucket.width // self.config.data.patch_size)
                    * (bucket.height // self.config.data.patch_size),
                }
                for bucket in self.config.data.resolution_buckets
            ],
        }
        output = Path(self.config.runtime.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolution_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
