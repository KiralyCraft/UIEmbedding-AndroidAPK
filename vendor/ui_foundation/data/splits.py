from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PackageSplitAssigner:
    train_fraction: float
    val_fraction: float
    test_fraction: float
    seed: int

    def assign(self, package_name: str) -> str:
        digest = hashlib.blake2b(
            f"{self.seed}:{package_name}".encode("utf-8"), digest_size=8
        ).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False) / float(2**64)
        if value < self.train_fraction:
            return "train"
        if value < self.train_fraction + self.val_fraction:
            return "val"
        return "test"
