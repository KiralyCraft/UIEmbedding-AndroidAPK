from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterator

from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ui_foundation.data.records import JsonlManifest, ScreenRecord


class ManifestMapDataset(Dataset[ScreenRecord]):
    def __init__(self, manifest_path: str, split: str, max_records: int | None = None) -> None:
        self.records = JsonlManifest(manifest_path).read(split=split, max_records=max_records)
        if len(self.records) == 0:
            raise RuntimeError(f"No records available for split={split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ScreenRecord:
        return self.records[index]


class ManifestStreamingDataset(IterableDataset[ScreenRecord]):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        shuffle_buffer_size: int,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        super().__init__()
        self.manifest_path = str(Path(manifest_path).resolve())
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.iteration = 0

    def __iter__(self) -> Iterator[ScreenRecord]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        shard_id = self.rank * worker_count + worker_id
        shard_count = self.world_size * worker_count
        iteration = self.iteration
        self.iteration += 1
        rng = random.Random(self.seed + shard_id * 104729 + iteration * 1000003)
        records = self._iter_byte_shard(shard_id, shard_count)
        if self.shuffle_buffer_size > 1:
            records = buffered_shuffle(records, self.shuffle_buffer_size, rng)
        yield from records

    def _iter_byte_shard(self, shard_id: int, shard_count: int) -> Iterator[ScreenRecord]:
        file_size = os.path.getsize(self.manifest_path)
        start = file_size * shard_id // shard_count
        end = file_size * (shard_id + 1) // shard_count
        with open(self.manifest_path, "rb") as handle:
            if start > 0:
                handle.seek(start - 1)
                if handle.read(1) != b"\n":
                    handle.readline()
            else:
                handle.seek(0)
            while True:
                position = handle.tell()
                if shard_id < shard_count - 1 and position >= end:
                    return
                line = handle.readline()
                if line == b"":
                    return
                stripped = line.strip()
                if stripped == b"":
                    continue
                record = ScreenRecord.from_json(stripped.decode("utf-8"))
                if record.split == self.split:
                    yield record


def buffered_shuffle(
    records: Iterator[ScreenRecord],
    buffer_size: int,
    rng: random.Random,
) -> Iterator[ScreenRecord]:
    buffer: list[ScreenRecord] = []
    iterator = iter(records)
    for _ in range(buffer_size):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break
    while len(buffer) > 0:
        try:
            incoming = next(iterator)
        except StopIteration:
            while len(buffer) > 0:
                yield buffer.pop(rng.randrange(len(buffer)))
            return
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = incoming
