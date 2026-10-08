from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass(slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def close(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(device_name: str, backend: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    if distributed:
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
        selected_backend = backend if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=selected_backend, init_method="env://")
    return DistributedContext(rank, local_rank, world_size, distributed, device)


def seed_everything(seed: int, deterministic: bool, rank: int = 0) -> None:
    effective = seed + rank * 100003
    random.seed(effective)
    np.random.seed(effective % (2**32 - 1))
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_amp_dtype(mode: str | bool, device: torch.device) -> torch.dtype | None:
    if isinstance(mode, bool):
        normalized = "auto" if mode else "off"
    else:
        normalized = str(mode).lower()
    if device.type != "cuda":
        return None
    if normalized == "off" or normalized == "none" or normalized == "fp32":
        return None
    if normalized == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        raise RuntimeError("BF16 was requested but is not supported on this CUDA device")
    if normalized == "fp16":
        return torch.float16
    if normalized == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"Unknown AMP mode: {mode}")


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=dtype)


def configure_storage(project_root: Path) -> None:
    data_root = project_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    defaults = {
        "HF_HOME": data_root / "cache" / "huggingface",
        "TORCH_HOME": data_root / "cache" / "torch",
        "WANDB_CACHE_DIR": data_root / "cache" / "wandb",
        "MPLCONFIGDIR": data_root / "cache" / "matplotlib",
        "TMPDIR": data_root / "tmp",
    }
    for key, path in defaults.items():
        if key not in os.environ:
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value
