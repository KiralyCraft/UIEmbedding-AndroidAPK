from __future__ import annotations

import importlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image, ImageDraw

from ui_foundation.config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


class Tracker(ABC):
    @abstractmethod
    def log(self, values: Mapping[str, float], step: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_ssl_visuals(self, batch: dict[str, Any], step: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def summary(self, values: Mapping[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> None:
        raise NotImplementedError


class NullTracker(Tracker):
    def log(self, values: Mapping[str, float], step: int) -> None:
        del values, step

    def log_ssl_visuals(self, batch: dict[str, Any], step: int) -> None:
        del batch, step

    def summary(self, values: Mapping[str, Any]) -> None:
        del values

    def finish(self) -> None:
        return


class WandbTracker(Tracker):
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled but wandb is not installed. Install requirements.txt "
                "or set wandb.enabled=false."
            ) from exc
        output = Path(config.runtime.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        run_id = config.wandb.run_id or None
        self.wandb = wandb
        self.run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity or None,
            dir=str(output.resolve()),
            id=run_id,
            resume=config.wandb.resume if run_id is not None else None,
            name=config.wandb.run_name or config.name,
            group=config.wandb.group or None,
            tags=config.wandb.tags or None,
            notes=config.wandb.notes or None,
            config=config.to_dict(),
            mode=config.wandb.mode,
            save_code=config.wandb.save_code,
        )
        (output / "wandb_run_id.txt").write_text(str(self.run.id) + "\n", encoding="utf-8")

    def log(self, values: Mapping[str, float], step: int) -> None:
        payload = {key: float(value) for key, value in values.items()}
        payload["global_step"] = int(step)
        self.run.log(payload)

    def log_ssl_visuals(self, batch: dict[str, Any], step: int) -> None:
        limit = min(self.config.wandb.visual_samples, len(batch["record_id"]))
        if limit <= 0:
            return
        table = self.wandb.Table(columns=["record_id", "teacher", "student", "masked_student", "masked_fraction"])
        for index in range(limit):
            teacher = tensor_to_pil(batch["teacher_images"][index], self.config)
            student = tensor_to_pil(batch["student_images"][index], self.config)
            masked = overlay_patch_mask(student, batch["patch_mask"][index], tuple(int(value) for value in batch["student_grid"]))
            fraction = float(batch["patch_mask"][index].float().mean())
            table.add_data(
                batch["record_id"][index],
                self.wandb.Image(teacher),
                self.wandb.Image(student),
                self.wandb.Image(masked),
                fraction,
            )
        self.run.log({"train/ssl_visuals": table, "global_step": step})

    def summary(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            self.run.summary[key] = value

    def finish(self) -> None:
        self.run.finish()


def build_tracker(config: ExperimentConfig, is_main: bool) -> Tracker:
    if not is_main or not config.wandb.enabled or config.wandb.mode == "disabled":
        return NullTracker()
    return WandbTracker(config)


def tensor_to_pil(tensor: torch.Tensor, config: ExperimentConfig) -> Image.Image:
    value = tensor.detach().float().cpu()
    mean = torch.tensor(config.data.normalize_mean).view(3, 1, 1)
    std = torch.tensor(config.data.normalize_std).view(3, 1, 1)
    value = (value * std + mean).clamp(0.0, 1.0)
    array = (value.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def overlay_patch_mask(image: Image.Image, mask: torch.Tensor, grid: tuple[int, int]) -> Image.Image:
    result = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    grid_h, grid_w = grid
    patch_w = result.width / grid_w
    patch_h = result.height / grid_h
    flattened = mask.detach().cpu().bool().reshape(grid_h, grid_w)
    for row in range(grid_h):
        for column in range(grid_w):
            if not bool(flattened[row, column]):
                continue
            left = int(column * patch_w)
            right = max(left + 1, int((column + 1) * patch_w))
            top = int(row * patch_h)
            bottom = max(top + 1, int((row + 1) * patch_h))
            draw.rectangle((left, top, right, bottom), fill=(0, 0, 0, 150))
    return Image.alpha_composite(result, overlay).convert("RGB")
