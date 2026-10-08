from __future__ import annotations

import contextlib
import logging
import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ui_foundation.config import ExperimentConfig
from ui_foundation.losses.distillation import DistillationLossOutput, UIDistillationLoss
from ui_foundation.losses.ssl_losses import LossOutput, UIFoundationSSLLoss
from ui_foundation.models.distillation import UIDistillationModel
from ui_foundation.models.ssl_model import UIFoundationSSLModel
from ui_foundation.training.checkpoints import CheckpointManager, unwrap_model
from ui_foundation.training.optim import trainable_parameters
from ui_foundation.training.tracking import Tracker
from ui_foundation.utils.runtime import (
    DistributedContext,
    autocast_context,
    move_to_device,
    resolve_amp_dtype,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainerState:
    step: int = 0
    best_validation: float = float("inf")


class CyclingLoader:
    """Turn a finite or sharded iterable DataLoader into an endless sequence of epochs."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.iterator: Iterator[Any] | None = None
        self.epochs_completed = 0

    def next(self) -> Any:
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.epochs_completed += 1
            self.iterator = iter(self.loader)
            try:
                return next(self.iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "The DataLoader yielded no batches. Check the split, rank sharding, "
                    "batch size, and training.drop_last setting."
                ) from exc


class StepTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        context: DistributedContext,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_loader: DataLoader,
        validation_loader: DataLoader | None,
        tracker: Tracker,
        checkpoints: CheckpointManager,
    ) -> None:
        self.config = config
        self.context = context
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_batches = CyclingLoader(train_loader)
        self.validation_batches = CyclingLoader(validation_loader) if validation_loader is not None else None
        self.tracker = tracker
        self.checkpoints = checkpoints
        self.amp_dtype = resolve_amp_dtype(config.runtime.amp, context.device)
        self.scaler = create_grad_scaler(context.device, self.amp_dtype)
        self.state = TrainerState()
        self.ssl_loss = UIFoundationSSLLoss(config.ssl)
        self.distillation_loss = UIDistillationLoss(config.distillation)
        self._backbone_frozen = False

    def restore_or_initialize(self) -> None:
        resume = self.config.training.resume
        if resume != "":
            self.state.step, self.state.best_validation = self.checkpoints.resume(
                resume,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
            )
        elif self.config.training.init_checkpoint != "":
            self.checkpoints.initialize(self.config.training.init_checkpoint, self.model)

    def train(self) -> TrainerState:
        self.restore_or_initialize()
        if self.state.step >= self.config.training.max_steps:
            LOGGER.info(
                "Checkpoint already reached max_steps=%d; no optimizer steps are required",
                self.config.training.max_steps,
            )
            return self.state

        self.model.train()
        progress = tqdm(
            total=self.config.training.max_steps,
            initial=self.state.step,
            desc=self.config.name,
            unit="step",
            disable=not (self.context.is_main and self.config.runtime.progress_bars),
            dynamic_ncols=True,
        )
        last_log_step = self.state.step
        last_log_time = time.perf_counter()
        try:
            while self.state.step < self.config.training.max_steps:
                next_step = self.state.step + 1
                self._set_backbone_frozen(next_step <= self.config.training.freeze_backbone_steps)
                should_log = next_step % self.config.runtime.log_every_steps == 0 or next_step == 1
                metrics, visual_batch = self._optimizer_step(next_step, collect_metrics=should_log)
                self.state.step = next_step
                if should_log:
                    elapsed = time.perf_counter() - last_log_time
                    completed_steps = self.state.step - last_log_step
                    metrics["train/steps_per_second"] = completed_steps / max(elapsed, 1.0e-9)
                    metrics["train/samples_per_second_per_rank"] = (
                        completed_steps
                        * self.config.training.batch_size
                        * self.config.training.gradient_accumulation_steps
                        / max(elapsed, 1.0e-9)
                    )
                    metrics["train/epoch_passes"] = float(self.train_batches.epochs_completed)
                    for index, group in enumerate(self.optimizer.param_groups):
                        name = str(group.get("name", f"group_{index}"))
                        metrics[f"lr/{name}"] = float(group["lr"])
                    reduced = reduce_metrics(metrics, self.context)
                    if self.context.is_main:
                        LOGGER.info(format_metrics(self.state.step, reduced))
                        self.tracker.log(reduced, self.state.step)
                        progress.set_postfix(loss=f"{reduced.get('train/loss', float('nan')):.4f}")
                    last_log_step = self.state.step
                    last_log_time = time.perf_counter()

                if (
                    self.context.is_main
                    and self.config.wandb.log_visuals
                    and self.state.step % self.config.wandb.visual_every_steps == 0
                    and self.config.task == "ssl_pretrain"
                ):
                    self.tracker.log_ssl_visuals(visual_batch, self.state.step)

                validation_value: float | None = None
                is_best = False
                if self._should_validate():
                    validation_metrics = self.validate()
                    validation_value = validation_metrics.get("validation/loss")
                    if self.context.is_main:
                        self.tracker.log(validation_metrics, self.state.step)
                        LOGGER.info(format_metrics(self.state.step, validation_metrics))
                    if validation_value is not None and math.isfinite(validation_value):
                        is_best = validation_value < self.state.best_validation
                        if is_best:
                            self.state.best_validation = validation_value

                should_save = (
                    self.state.step % self.config.runtime.save_every_steps == 0
                    or self.state.step == self.config.training.max_steps
                    or is_best
                )
                if should_save and self.context.is_main:
                    self.checkpoints.save(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        self.state.step,
                        self.state.best_validation,
                        validation_value,
                        is_best=is_best,
                    )
                progress.update(1)
        finally:
            progress.close()
        return self.state

    def validate(self) -> dict[str, float]:
        if self.validation_batches is None:
            return {}
        was_training = self.model.training
        self.model.eval()
        totals: dict[str, float] = {}
        sample_count = 0
        with torch.no_grad():
            for _ in range(self.config.training.validation_steps):
                batch = move_to_device(self.validation_batches.next(), self.context.device)
                batch_size = infer_batch_size(batch)
                with autocast_context(self.context.device, self.amp_dtype):
                    loss_output = self._forward_loss(batch)
                values = loss_output_to_metrics(loss_output, "validation")
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.0) + value * batch_size
                sample_count += batch_size
        if was_training:
            self.model.train()
        metrics = {key: value / max(sample_count, 1) for key, value in totals.items()}
        return reduce_metrics(metrics, self.context)

    def _optimizer_step(
        self,
        step: int,
        collect_metrics: bool,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = self.config.training.gradient_accumulation_steps
        totals: dict[str, torch.Tensor] = {}
        visual_batch: dict[str, Any] = {}
        for micro_step in range(accumulation):
            batch = self.train_batches.next()
            if micro_step == 0:
                visual_batch = batch
            batch = move_to_device(batch, self.context.device)
            sync_context = self._gradient_sync_context(micro_step, accumulation)
            with sync_context:
                with autocast_context(self.context.device, self.amp_dtype):
                    loss_output = self._forward_loss(batch)
                    scaled_loss = loss_output.total / accumulation
                self.scaler.scale(scaled_loss).backward()
            if collect_metrics:
                values = loss_output_to_metric_tensors(loss_output, "train")
                for key, value in values.items():
                    totals[key] = totals.get(key, value.new_zeros(())) + value / accumulation

        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(trainable_parameters(self.model)),
            self.config.optimizer.gradient_clip_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        underlying = unwrap_model(self.model)
        if isinstance(underlying, UIFoundationSSLModel):
            underlying.update_teacher(self._teacher_momentum(step))
        if not collect_metrics:
            return {}, visual_batch
        totals["train/gradient_norm"] = gradient_norm.detach().float()
        metrics = materialize_metric_tensors(totals)
        if isinstance(underlying, UIFoundationSSLModel):
            metrics["train/teacher_momentum"] = self._teacher_momentum(step)
        return metrics, visual_batch

    def _forward_loss(self, batch: dict[str, Any]) -> LossOutput | DistillationLossOutput:
        if self.config.task == "ssl_pretrain":
            grid_student = grid_tuple(batch["student_grid"])
            grid_teacher = grid_tuple(batch["teacher_grid"])
            output = self.model(
                batch["student_images"],
                batch["teacher_images"],
                batch["patch_mask"],
                grid_student,
                grid_teacher,
                batch.get("student_valid_patch_mask"),
                batch.get("teacher_valid_patch_mask"),
            )
            return self.ssl_loss(output, batch)
        if self.config.task == "distill":
            output = self.model(batch["student_images"], batch["teacher_images"])
            return self.distillation_loss(output)
        raise ValueError(f"Unsupported training task: {self.config.task}")

    def _gradient_sync_context(self, micro_step: int, accumulation: int):
        if isinstance(self.model, DistributedDataParallel) and micro_step < accumulation - 1:
            return self.model.no_sync()
        return contextlib.nullcontext()

    def _teacher_momentum(self, step: int) -> float:
        start = self.config.ssl.teacher_momentum_start
        end = self.config.ssl.teacher_momentum_end
        progress = min(max(step / max(self.config.training.max_steps, 1), 0.0), 1.0)
        return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0

    def _should_validate(self) -> bool:
        return (
            self.validation_batches is not None
            and self.config.training.validation_every_steps > 0
            and (
                self.state.step % self.config.training.validation_every_steps == 0
                or self.state.step == self.config.training.max_steps
            )
        )

    def _set_backbone_frozen(self, frozen: bool) -> None:
        if frozen == self._backbone_frozen:
            return
        underlying = unwrap_model(self.model)
        candidates: list[nn.Module] = []
        if isinstance(underlying, UIFoundationSSLModel):
            candidates.append(underlying.student_backbone)
        elif isinstance(underlying, UIDistillationModel):
            candidates.append(underlying.student.backbone)
        for module in candidates:
            for parameter in module.parameters():
                parameter.requires_grad_(not frozen)
        self._backbone_frozen = frozen
        LOGGER.info("Student backbone frozen=%s", frozen)


def create_grad_scaler(device: torch.device, amp_dtype: torch.dtype | None):
    enabled = device.type == "cuda" and amp_dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def infer_batch_size(batch: Mapping[str, Any]) -> int:
    for key in ("student_images", "teacher_images", "images"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    return 1


def loss_output_to_metrics(output: LossOutput | DistillationLossOutput, prefix: str) -> dict[str, float]:
    return materialize_metric_tensors(loss_output_to_metric_tensors(output, prefix))


def loss_output_to_metric_tensors(
    output: LossOutput | DistillationLossOutput,
    prefix: str,
) -> dict[str, torch.Tensor]:
    metrics = {f"{prefix}/loss": output.total.detach().float()}
    for key, value in output.components.items():
        metrics[f"{prefix}/{key}"] = value.detach().float()
    return metrics


def materialize_metric_tensors(metrics: Mapping[str, torch.Tensor]) -> dict[str, float]:
    if len(metrics) == 0:
        return {}
    keys = list(metrics)
    values = torch.stack([metrics[key].reshape(()) for key in keys]).cpu().tolist()
    return {key: float(value) for key, value in zip(keys, values, strict=True)}


def grid_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, torch.Tensor):
        items = value.detach().cpu().tolist()
    else:
        items = value
    if len(items) != 2:
        raise ValueError(f"Expected a two-dimensional grid, got {items!r}")
    return int(items[0]), int(items[1])


def reduce_metrics(metrics: Mapping[str, float], context: DistributedContext) -> dict[str, float]:
    result = {key: float(value) for key, value in metrics.items()}
    if not context.distributed or len(result) == 0:
        return result
    keys = sorted(result)
    values = torch.tensor([result[key] for key in keys], dtype=torch.float64, device=context.device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= context.world_size
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist(), strict=True)}


def format_metrics(step: int, metrics: Mapping[str, float]) -> str:
    selected = []
    for key in sorted(metrics):
        if key.endswith("/loss") or key.endswith("/global") or key.endswith("/masked_patch"):
            selected.append(f"{key}={metrics[key]:.5f}")
    if len(selected) == 0:
        selected = [f"{key}={value:.5f}" for key, value in list(sorted(metrics.items()))[:5]]
    return f"step={step} " + " ".join(selected)
