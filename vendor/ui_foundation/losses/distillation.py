from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ui_foundation.config import DistillationConfig
from ui_foundation.models.distillation import DistillationOutput


@dataclass(slots=True)
class DistillationLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


class UIDistillationLoss:
    def __init__(self, config: DistillationConfig) -> None:
        self.config = config

    def __call__(self, output: DistillationOutput) -> DistillationLossOutput:
        global_loss = (2.0 - 2.0 * (output.student_embedding * output.teacher_embedding.detach()).sum(dim=-1)).mean()
        student_similarity = output.student_embedding @ output.student_embedding.T
        teacher_similarity = output.teacher_embedding.detach() @ output.teacher_embedding.detach().T
        relational_loss = F.smooth_l1_loss(student_similarity, teacher_similarity)
        dense_loss = (
            2.0
            - 2.0
            * (F.normalize(output.student_dense, dim=-1) * F.normalize(output.teacher_dense.detach(), dim=-1)).sum(dim=-1)
        ).mean()
        total = (
            self.config.global_weight * global_loss
            + self.config.relational_weight * relational_loss
            + self.config.dense_weight * dense_loss
        )
        return DistillationLossOutput(
            total=total,
            components={"global": global_loss, "relational": relational_loss, "dense": dense_loss},
        )
