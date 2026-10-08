from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ui_foundation.config import ExperimentConfig


def build_optimizer(config: ExperimentConfig, model: nn.Module) -> Optimizer:
    backbone_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    backbone_no_decay: list[nn.Parameter] = []
    head_no_decay: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = "backbone" in name or name.startswith("student.model")
        no_decay = parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        if not config.optimizer.exclude_bias_and_norm_from_weight_decay:
            no_decay = False
        if is_backbone and no_decay:
            backbone_no_decay.append(parameter)
        elif is_backbone:
            backbone_parameters.append(parameter)
        elif no_decay:
            head_no_decay.append(parameter)
        else:
            head_parameters.append(parameter)

    groups = []
    if len(backbone_parameters) > 0:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": config.optimizer.learning_rate,
                "weight_decay": config.optimizer.weight_decay,
                "name": "backbone_decay",
            }
        )
    if len(backbone_no_decay) > 0:
        groups.append(
            {
                "params": backbone_no_decay,
                "lr": config.optimizer.learning_rate,
                "weight_decay": 0.0,
                "name": "backbone_no_decay",
            }
        )
    if len(head_parameters) > 0:
        groups.append(
            {
                "params": head_parameters,
                "lr": config.optimizer.head_learning_rate,
                "weight_decay": config.optimizer.weight_decay,
                "name": "head_decay",
            }
        )
    if len(head_no_decay) > 0:
        groups.append(
            {
                "params": head_no_decay,
                "lr": config.optimizer.head_learning_rate,
                "weight_decay": 0.0,
                "name": "head_no_decay",
            }
        )

    if config.optimizer.name.lower() != "adamw":
        raise ValueError("Only AdamW is currently supported")
    return torch.optim.AdamW(
        groups,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
    )


def build_scheduler(config: ExperimentConfig, optimizer: Optimizer) -> LambdaLR:
    max_steps = config.training.max_steps
    warmup_steps = min(config.optimizer.warmup_steps, max_steps - 1)
    minimum = config.optimizer.min_lr_ratio

    def scale(step: int) -> float:
        if step < warmup_steps:
            return max(1.0e-8, (step + 1) / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return minimum + (1.0 - minimum) * cosine

    return LambdaLR(optimizer, lr_lambda=scale)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
