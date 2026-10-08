from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ui_foundation.config import ExperimentConfig
from ui_foundation.models.distillation import UIDistillationModel
from ui_foundation.models.ssl_model import UIFoundationSSLModel

LOGGER = logging.getLogger(__name__)


def unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    while True:
        module = getattr(current, "module", None)
        if isinstance(module, nn.Module):
            current = module
            continue
        original = getattr(current, "_orig_mod", None)
        if isinstance(original, nn.Module):
            current = original
            continue
        return current



class CheckpointManager:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.directory = Path(config.runtime.output_dir) / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: torch.amp.GradScaler,
        step: int,
        best_metric: float,
        validation_metric: float | None,
        is_best: bool,
    ) -> Path:
        unwrapped = unwrap_model(model)
        payload: dict[str, Any] = {
            "format_version": 1,
            "config": self.config.to_dict(),
            "step": step,
            "best_metric": best_metric,
            "validation_metric": validation_metric,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": capture_rng_state(),
        }
        if isinstance(unwrapped, UIFoundationSSLModel):
            payload.update(
                {
                    "student_backbone_state_dict": unwrapped.student_backbone.state_dict(),
                    "teacher_backbone_state_dict": unwrapped.teacher_backbone.state_dict(),
                    "student_global_projector_state_dict": unwrapped.student_global_projector.state_dict(),
                    "teacher_global_projector_state_dict": unwrapped.teacher_global_projector.state_dict(),
                    "student_patch_projector_state_dict": unwrapped.student_patch_projector.state_dict(),
                    "teacher_patch_projector_state_dict": unwrapped.teacher_patch_projector.state_dict(),
                    "trainable_heads_state_dict": {
                        "global_predictor": unwrapped.global_predictor.state_dict(),
                        "patch_predictor": unwrapped.patch_predictor.state_dict(),
                        "xml_role_head": unwrapped.xml_role_head.state_dict(),
                        "xml_attribute_head": unwrapped.xml_attribute_head.state_dict(),
                        "xml_count_head": unwrapped.xml_count_head.state_dict(),
                        "xml_occupancy_head": unwrapped.xml_occupancy_head.state_dict(),
                    },
                }
            )
        elif isinstance(unwrapped, UIDistillationModel):
            payload.update(
                {
                    "student_model_state_dict": unwrapped.student.state_dict(),
                    "teacher_output_dim": unwrapped.teacher.output_dim,
                }
            )
        else:
            payload["model_state_dict"] = unwrapped.state_dict()

        path = self.directory / f"step_{step:09d}.pt"
        atomic_torch_save(payload, path)
        atomic_torch_save(payload, self.directory / "last.pt")
        if is_best:
            atomic_torch_save(payload, self.directory / "best.pt")
        self._prune()
        return path

    def resume(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: torch.amp.GradScaler,
    ) -> tuple[int, float]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._load_model(payload, unwrap_model(model), strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        if "scaler_state_dict" in payload:
            scaler.load_state_dict(payload["scaler_state_dict"])
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])
        step = int(payload.get("step", 0))
        best = float(payload.get("best_metric", float("inf")))
        LOGGER.info("Resumed %s at optimizer step %d", path, step)
        return step, best

    def initialize(self, path: str, model: nn.Module) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._load_model(payload, unwrap_model(model), strict=False)
        LOGGER.info("Initialized model weights from %s", path)

    def _load_model(self, payload: dict[str, Any], model: nn.Module, strict: bool) -> None:
        if isinstance(model, UIFoundationSSLModel):
            model.student_backbone.load_state_dict(payload["student_backbone_state_dict"], strict=strict)
            model.teacher_backbone.load_state_dict(payload["teacher_backbone_state_dict"], strict=strict)
            model.student_global_projector.load_state_dict(payload["student_global_projector_state_dict"], strict=strict)
            model.teacher_global_projector.load_state_dict(payload["teacher_global_projector_state_dict"], strict=strict)
            model.student_patch_projector.load_state_dict(payload["student_patch_projector_state_dict"], strict=strict)
            model.teacher_patch_projector.load_state_dict(payload["teacher_patch_projector_state_dict"], strict=strict)
            heads = payload.get("trainable_heads_state_dict", {})
            for name in (
                "global_predictor",
                "patch_predictor",
                "xml_role_head",
                "xml_attribute_head",
                "xml_count_head",
                "xml_occupancy_head",
            ):
                if name in heads:
                    getattr(model, name).load_state_dict(heads[name], strict=strict)
        elif isinstance(model, UIDistillationModel):
            model.student.load_state_dict(payload["student_model_state_dict"], strict=strict)
        else:
            model.load_state_dict(payload["model_state_dict"], strict=strict)

    def _prune(self) -> None:
        keep = self.config.runtime.keep_last_checkpoints
        paths = sorted(self.directory.glob("step_*.pt"))
        for path in paths[:-keep] if keep > 0 else paths:
            path.unlink(missing_ok=True)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
