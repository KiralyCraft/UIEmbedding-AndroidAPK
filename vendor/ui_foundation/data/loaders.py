from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from ui_foundation.config import ExperimentConfig
from ui_foundation.data.collators import DistillationBatchCollator, EvaluationBatchCollator, SSLBatchCollator
from ui_foundation.data.datasets import ManifestMapDataset, ManifestStreamingDataset
from ui_foundation.utils.runtime import DistributedContext


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class DataLoaderFactory:
    def __init__(self, config: ExperimentConfig, context: DistributedContext) -> None:
        self.config = config
        self.context = context

    def build_ssl(self, manifest_path: str, split: str = "train") -> DataLoader:
        dataset = ManifestStreamingDataset(
            manifest_path=manifest_path,
            split=split,
            shuffle_buffer_size=self.config.data.shuffle_buffer_size,
            seed=self.config.runtime.seed,
            rank=self.context.rank,
            world_size=self.context.world_size,
        )
        collator = SSLBatchCollator(self.config.data, self.config.ssl, self.config.runtime.seed + self.context.rank)
        return self._loader(dataset, collator, self.config.training.batch_size, shuffle=False)

    def build_distillation(self, manifest_path: str, split: str = "train") -> DataLoader:
        dataset = ManifestStreamingDataset(
            manifest_path=manifest_path,
            split=split,
            shuffle_buffer_size=self.config.data.shuffle_buffer_size,
            seed=self.config.runtime.seed,
            rank=self.context.rank,
            world_size=self.context.world_size,
        )
        collator = DistillationBatchCollator(self.config.data, self.config.distillation)
        return self._loader(dataset, collator, self.config.training.batch_size, shuffle=False)

    def build_evaluation(self, manifest_path: str, split: str | None = None) -> DataLoader:
        selected_split = split or self.config.evaluation.split
        dataset = ManifestMapDataset(
            manifest_path=manifest_path,
            split=selected_split,
            max_records=self.config.evaluation.max_records,
        )
        sampler = None
        if self.context.distributed:
            sampler = ExactDistributedSampler(len(dataset), self.context.rank, self.context.world_size)
        bucket = self.config.data.resolution_buckets[-1]
        collator = EvaluationBatchCollator(self.config.data, bucket)
        kwargs = self._worker_kwargs()
        return DataLoader(
            dataset,
            batch_size=self.config.evaluation.batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collator,
            **kwargs,
        )

    def _loader(self, dataset, collator, batch_size: int, shuffle: bool) -> DataLoader:
        kwargs = self._worker_kwargs()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=self.config.training.drop_last,
            collate_fn=collator,
            **kwargs,
        )

    def _worker_kwargs(self) -> dict:
        kwargs = {
            "num_workers": self.config.runtime.num_workers,
            "pin_memory": self.config.runtime.pin_memory,
            "persistent_workers": self.config.runtime.persistent_workers and self.config.runtime.num_workers > 0,
            "worker_init_fn": seed_worker,
            "in_order": self.config.runtime.in_order,
        }
        if self.config.runtime.num_workers > 0:
            kwargs["prefetch_factor"] = self.config.runtime.prefetch_factor
        return kwargs


class ExactDistributedSampler(Sampler[int]):
    """Shard a map dataset without padding or duplicate evaluation records."""

    def __init__(self, length: int, rank: int, world_size: int) -> None:
        self.length = length
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.length:
            return 0
        return (self.length - 1 - self.rank) // self.world_size + 1
