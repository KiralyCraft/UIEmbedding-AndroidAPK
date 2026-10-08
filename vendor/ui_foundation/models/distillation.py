from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ui_foundation.config import ExperimentConfig, experiment_config_from_dict
from ui_foundation.models.backbones import build_backbone
from ui_foundation.models.heads import MobileEmbeddingHead, ProjectionMLP
from ui_foundation.models.ssl_model import align_patch_tokens


@dataclass(slots=True)
class DistillationOutput:
    student_embedding: torch.Tensor
    teacher_embedding: torch.Tensor
    student_dense: torch.Tensor
    teacher_dense: torch.Tensor


class FrozenFoundationTeacher(nn.Module):
    def __init__(self, checkpoint_path: str, repo_override: str = "", weights_override: str = "") -> None:
        super().__init__()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict) or "config" not in payload:
            raise ValueError("Teacher checkpoint must contain the resolved training config")
        teacher_config = experiment_config_from_dict(payload["config"])
        teacher_config.model.pretrained = False
        teacher_config.model.checkpoint = ""
        if repo_override != "":
            teacher_config.model.repo_dir = repo_override
        if weights_override != "":
            teacher_config.model.weights = weights_override
        self.backbone = build_backbone(teacher_config.model)
        self.projector = ProjectionMLP(
            self.backbone.feature_dim,
            teacher_config.model.projection_hidden_dim,
            teacher_config.model.projection_dim,
        )
        self.patch_projector = ProjectionMLP(
            self.backbone.feature_dim,
            teacher_config.model.projection_hidden_dim,
            teacher_config.model.projection_dim,
            layers=2,
        )
        self.backbone.load_state_dict(payload["teacher_backbone_state_dict"], strict=True)
        self.projector.load_state_dict(payload["teacher_global_projector_state_dict"], strict=True)
        self.patch_projector.load_state_dict(payload["teacher_patch_projector_state_dict"], strict=True)
        self.output_dim = teacher_config.model.projection_dim
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> FrozenFoundationTeacher:
        """Keep teacher inference behavior fixed when the student enters training mode."""
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        features = self.backbone.forward_features(images)
        embedding = F.normalize(self.projector(features.global_token), dim=-1)
        dense = F.normalize(self.patch_projector(features.patch_tokens), dim=-1)
        return embedding, dense, features.grid_size


class MobileNetStudent(nn.Module):
    def __init__(self, config: ExperimentConfig, output_dim: int) -> None:
        super().__init__()
        self.backbone = build_backbone(config.model)
        self.embedding_head = MobileEmbeddingHead(
            self.backbone.feature_dim,
            output_dim,
            config.model.dropout,
        )
        self.dense_head = nn.Linear(self.backbone.feature_dim, output_dim, bias=False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        features = self.backbone.forward_features(images)
        embedding = self.embedding_head(features.global_token)
        dense = F.normalize(self.dense_head(features.patch_tokens), dim=-1)
        return embedding, dense, features.grid_size


class UIDistillationModel(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        checkpoint = config.distillation.teacher_checkpoint
        if checkpoint == "":
            raise ValueError("distillation.teacher_checkpoint is required")
        self.teacher = FrozenFoundationTeacher(
            checkpoint,
            repo_override=config.model.repo_dir,
            weights_override=config.model.weights,
        )
        self.student = MobileNetStudent(config, self.teacher.output_dim)

    def forward(self, student_images: torch.Tensor, teacher_images: torch.Tensor) -> DistillationOutput:
        with torch.no_grad():
            teacher_embedding, teacher_dense, teacher_grid = self.teacher(teacher_images)
        student_embedding, student_dense, student_grid = self.student(student_images)
        teacher_dense = align_patch_tokens(teacher_dense, teacher_grid, student_grid)
        teacher_dense = F.normalize(teacher_dense, dim=-1)
        return DistillationOutput(
            student_embedding=student_embedding,
            teacher_embedding=teacher_embedding,
            student_dense=student_dense,
            teacher_dense=teacher_dense,
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        embedding, _, _ = self.student(images)
        return embedding
