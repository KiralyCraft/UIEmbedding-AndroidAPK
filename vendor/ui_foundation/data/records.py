from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class ScreenRecord:
    record_id: str
    source: str
    image_path: str
    xml_path: str
    package_name: str
    activity_name: str = ""
    structure_id: str = ""
    state_id: str = ""
    trace_id: str = ""
    split: str = "train"
    is_external: bool = False
    sequence_index: int = -1
    image_width: int = 0
    image_height: int = 0
    image_sha256: str = ""
    # Native MobileViews representatives can occur many times in action traces.
    # Pairs are (continuous segment, observation index), never file-number order.
    trace_visits: tuple[tuple[int, int], ...] = ()

    @property
    def package_key(self) -> str:
        return self.package_name

    @property
    def activity_key(self) -> str:
        if self.activity_name == "":
            return ""
        return f"{self.package_name}::{self.activity_name}"

    @property
    def structure_key(self) -> str:
        if self.structure_id == "":
            return ""
        return f"{self.package_name}::{self.structure_id}"

    @property
    def aspect_ratio(self) -> float:
        if self.image_width <= 0 or self.image_height <= 0:
            return 0.0
        return self.image_height / self.image_width

    def label(self, name: str) -> str:
        value = getattr(self, name, "")
        return str(value or "")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "ScreenRecord":
        payload = json.loads(value)
        allowed = set(cls.__dataclass_fields__)
        filtered = {key: item for key, item in payload.items() if key in allowed}
        if "trace_visits" in filtered:
            filtered["trace_visits"] = tuple(tuple(visit) for visit in filtered["trace_visits"])
        return cls(**filtered)


@dataclass(slots=True)
class ManifestStatistics:
    total: int = 0
    splits: dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0, "test": 0})
    resolutions: dict[str, int] = field(default_factory=dict)
    packages: set[str] = field(default_factory=set)

    def add(self, record: ScreenRecord) -> None:
        self.total += 1
        self.splits[record.split] = self.splits.get(record.split, 0) + 1
        if record.image_width > 0 and record.image_height > 0:
            key = f"{record.image_width}x{record.image_height}"
            self.resolutions[key] = self.resolutions.get(key, 0) + 1
        self.packages.add(record.package_name)

    def count(self, split: str) -> int:
        return int(self.splits.get(split, 0))

    def to_dict(self) -> dict:
        resolutions = sorted(self.resolutions.items(), key=lambda item: (-item[1], item[0]))
        return {
            "total": self.total,
            "splits": dict(self.splits),
            "package_count": len(self.packages),
            "resolutions": dict(resolutions),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ManifestStatistics":
        return cls(
            total=int(value["total"]),
            splits=dict(value.get("splits", {})),
            resolutions=dict(value.get("resolutions", {})),
        )


class JsonlManifest:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.statistics_path = self.path.with_suffix(self.path.suffix + ".stats.json")

    def write(self, records: Iterable[ScreenRecord]) -> ManifestStatistics:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        statistics = ManifestStatistics()
        with temporary_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.to_json())
                handle.write("\n")
                statistics.add(record)
        temporary_path.replace(self.path)
        self.statistics_path.write_text(
            json.dumps(statistics.to_dict(), indent=2, sort_keys=False),
            encoding="utf-8",
        )
        return statistics

    def read(self, split: str | None = None, max_records: int | None = None) -> list[ScreenRecord]:
        return list(self.iter_records(split=split, max_records=max_records))

    def iter_records(
        self,
        split: str | None = None,
        max_records: int | None = None,
    ) -> Iterator[ScreenRecord]:
        emitted = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if stripped == "":
                    continue
                try:
                    record = ScreenRecord.from_json(stripped)
                except Exception as exc:
                    raise ValueError(f"Invalid manifest line {self.path}:{line_number}") from exc
                if split is not None and record.split != split:
                    continue
                yield record
                emitted += 1
                if max_records is not None and emitted >= max_records:
                    return

    def statistics(self) -> ManifestStatistics:
        if self.statistics_path.exists():
            payload = json.loads(self.statistics_path.read_text(encoding="utf-8"))
            return ManifestStatistics.from_dict(payload)
        statistics = ManifestStatistics()
        for record in self.iter_records():
            statistics.add(record)
        self.statistics_path.write_text(
            json.dumps(statistics.to_dict(), indent=2), encoding="utf-8"
        )
        return statistics
