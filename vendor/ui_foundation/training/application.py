from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ui_foundation.config import ExperimentConfig
from ui_foundation.data.acquisition import DatasetMaterializer
from ui_foundation.data.loaders import DataLoaderFactory
from ui_foundation.data.manifest import ManifestManager
from ui_foundation.data.records import JsonlManifest
from ui_foundation.models.distillation import UIDistillationModel
from ui_foundation.models.ssl_model import UIFoundationSSLModel
from ui_foundation.training.checkpoints import CheckpointManager
from ui_foundation.training.optim import build_optimizer, build_scheduler
from ui_foundation.training.tracking import build_tracker
from ui_foundation.training.trainer import StepTrainer
from ui_foundation.utils.logging import configure_logging
from ui_foundation.utils.runtime import DistributedContext, seed_everything

LOGGER = logging.getLogger(__name__)


class TrainingApplication:
    def __init__(self, config: ExperimentConfig, context: DistributedContext) -> None:
        self.config = config
        self.context = context

    def run(self) -> None:
        output = Path(self.config.runtime.output_dir)
        if self.context.is_main:
            output.mkdir(parents=True, exist_ok=True)
        self.context.barrier()
        configure_logging(output, rank=self.context.rank)
        seed_everything(self.config.runtime.seed, self.config.runtime.deterministic, self.context.rank)
        manifest_path = prepare_data(self.config, self.context)
        if self.context.is_main:
            (output / "resolved_config.json").write_text(
                json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
            )

        model = self._build_model().to(self.context.device)
        optimizer = build_optimizer(self.config, model)
        scheduler = build_scheduler(self.config, optimizer)
        if self.config.runtime.compile:
            model = torch.compile(model)
        if self.context.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[self.context.local_rank] if self.context.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=self.config.runtime.find_unused_parameters,
            )

        loaders = DataLoaderFactory(self.config, self.context)
        if self.config.task == "ssl_pretrain":
            train_loader = loaders.build_ssl(str(manifest_path), split="train")
            validation_loader = self._optional_validation_loader(loaders, manifest_path, "ssl")
        elif self.config.task == "distill":
            train_loader = loaders.build_distillation(str(manifest_path), split="train")
            validation_loader = self._optional_validation_loader(loaders, manifest_path, "distill")
        else:
            raise ValueError(f"TrainingApplication cannot execute task {self.config.task!r}")

        tracker = build_tracker(self.config, self.context.is_main)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        effective_batch = (
            self.config.training.batch_size
            * self.config.training.gradient_accumulation_steps
            * self.context.world_size
        )
        if self.context.is_main:
            LOGGER.info(
                "Model parameters: total=%d trainable=%d; effective batch=%d",
                total_parameters,
                trainable_parameters,
                effective_batch,
            )
            tracker.summary(
                {
                    "model/parameters_total": total_parameters,
                    "model/parameters_trainable": trainable_parameters,
                    "training/effective_batch": effective_batch,
                }
            )
        checkpoints = CheckpointManager(self.config)
        trainer = StepTrainer(
            self.config,
            self.context,
            model,
            optimizer,
            scheduler,
            train_loader,
            validation_loader,
            tracker,
            checkpoints,
        )
        try:
            state = trainer.train()
            if self.context.is_main:
                tracker.summary(
                    {
                        "final_step": state.step,
                        "best_validation_loss": state.best_validation,
                        "manifest": str(manifest_path),
                    }
                )
        finally:
            tracker.finish()

    def _build_model(self):
        if self.config.task == "ssl_pretrain":
            return UIFoundationSSLModel(self.config)
        if self.config.task == "distill":
            return UIDistillationModel(self.config)
        raise ValueError(f"Unsupported training task {self.config.task!r}")

    def _optional_validation_loader(self, loaders: DataLoaderFactory, manifest_path: Path, kind: str):
        statistics = JsonlManifest(manifest_path).statistics()
        if statistics.count("val") <= 0 or self.config.training.validation_steps <= 0:
            LOGGER.warning("No validation records are available; checkpoint selection will use last.pt only")
            return None
        if kind == "ssl":
            return loaders.build_ssl(str(manifest_path), split="val")
        return loaders.build_distillation(str(manifest_path), split="val")


def prepare_data(config: ExperimentConfig, context: DistributedContext) -> Path:
    payload: list[str | None] = [None, None]
    if context.is_main:
        root = DatasetMaterializer(config).prepare()
        config.data.root = str(root)
        manifest = ManifestManager(config).prepare()
        payload = [str(root), str(manifest)]
    if context.distributed:
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is None or payload[1] is None:
        raise RuntimeError("Dataset preparation did not produce a root and manifest")
    config.data.root = payload[0]
    config.data.manifest = payload[1]
    context.barrier()
    return Path(payload[1])
