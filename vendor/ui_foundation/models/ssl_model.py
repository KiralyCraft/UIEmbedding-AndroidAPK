from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ui_foundation.config import ExperimentConfig
from ui_foundation.models.backbones import BackboneFeatures, FeatureBackbone, build_backbone, clone_backbone
from ui_foundation.models.heads import PredictorMLP, ProjectionMLP


@dataclass(slots=True)
class SSLForwardOutput:
    student_global: torch.Tensor
    teacher_global: torch.Tensor
    student_patches: torch.Tensor
    teacher_patches: torch.Tensor
    student_patch_mean: torch.Tensor
    teacher_patch_mean: torch.Tensor
    xml_role_logits: torch.Tensor | None
    xml_attribute_logits: torch.Tensor | None
    xml_count_logits: torch.Tensor | None
    xml_occupancy_logits: torch.Tensor | None


class UIFoundationSSLModel(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.student_backbone = build_backbone(config.model)
        self.teacher_backbone = clone_backbone(self.student_backbone)
        feature_dim = self.student_backbone.feature_dim
        projection_dim = config.model.projection_dim
        self.student_global_projector = ProjectionMLP(
            feature_dim,
            config.model.projection_hidden_dim,
            projection_dim,
        )
        self.teacher_global_projector = copy.deepcopy(self.student_global_projector)
        self.student_patch_projector = ProjectionMLP(
            feature_dim,
            config.model.projection_hidden_dim,
            projection_dim,
            layers=2,
        )
        self.teacher_patch_projector = copy.deepcopy(self.student_patch_projector)
        self.global_predictor = PredictorMLP(projection_dim, config.model.predictor_hidden_dim)
        self.patch_predictor = PredictorMLP(projection_dim, config.model.predictor_hidden_dim)

        self.xml_role_head = nn.Linear(feature_dim, config.ssl.xml_role_classes)
        self.xml_attribute_head = nn.Linear(feature_dim, config.ssl.xml_attribute_count)
        self.xml_count_head = nn.Linear(feature_dim, config.ssl.xml_count_bins)
        self.xml_occupancy_head = nn.Linear(
            feature_dim,
            config.ssl.xml_occupancy_grid * config.ssl.xml_occupancy_grid,
        )
        self.xml_region_enabled = config.ssl.xml_region_weight > 0.0
        self.xml_structure_enabled = config.ssl.xml_structure_weight > 0.0
        if not self.xml_region_enabled:
            self.xml_role_head.requires_grad_(False)
            self.xml_attribute_head.requires_grad_(False)
        if not self.xml_structure_enabled:
            self.xml_count_head.requires_grad_(False)
            self.xml_occupancy_head.requires_grad_(False)
        self._freeze_teacher()

    def forward(
        self,
        student_images: torch.Tensor,
        teacher_images: torch.Tensor,
        patch_mask: torch.Tensor,
        student_grid: tuple[int, int],
        teacher_grid: tuple[int, int],
        student_valid_patch_mask: torch.Tensor | None = None,
        teacher_valid_patch_mask: torch.Tensor | None = None,
    ) -> SSLForwardOutput:
        student_features = self.student_backbone.forward_features(student_images, patch_mask)
        with torch.no_grad():
            teacher_features = self.teacher_backbone.forward_features(teacher_images, None)

        student_global = self.global_predictor(
            self.student_global_projector(student_features.global_token)
        )
        with torch.no_grad():
            teacher_global = self.teacher_global_projector(teacher_features.global_token)

        student_patches = self.patch_predictor(
            self.student_patch_projector(student_features.patch_tokens)
        )
        student_patch_mean = masked_mean(student_patches, student_valid_patch_mask)
        with torch.no_grad():
            teacher_patches_native = self.teacher_patch_projector(teacher_features.patch_tokens)
            teacher_patch_mean = masked_mean(teacher_patches_native, teacher_valid_patch_mask)
            teacher_patches = align_patch_tokens(teacher_patches_native, teacher_grid, student_grid)
        xml_role_logits = (
            self.xml_role_head(student_features.patch_tokens) if self.xml_region_enabled else None
        )
        xml_attribute_logits = (
            self.xml_attribute_head(student_features.patch_tokens) if self.xml_region_enabled else None
        )
        xml_count_logits = (
            self.xml_count_head(student_features.global_token) if self.xml_structure_enabled else None
        )
        xml_occupancy_logits = (
            self.xml_occupancy_head(student_features.global_token) if self.xml_structure_enabled else None
        )
        return SSLForwardOutput(
            student_global=student_global,
            teacher_global=teacher_global,
            student_patches=student_patches,
            teacher_patches=teacher_patches,
            student_patch_mean=student_patch_mean,
            teacher_patch_mean=teacher_patch_mean,
            xml_role_logits=xml_role_logits,
            xml_attribute_logits=xml_attribute_logits,
            xml_count_logits=xml_count_logits,
            xml_occupancy_logits=xml_occupancy_logits,
        )

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        pairs = (
            (self.student_backbone, self.teacher_backbone),
            (self.student_global_projector, self.teacher_global_projector),
            (self.student_patch_projector, self.teacher_patch_projector),
        )
        for student_module, teacher_module in pairs:
            for student, teacher in zip(student_module.parameters(), teacher_module.parameters()):
                teacher.data.mul_(momentum).add_(student.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def encode_teacher(self, images: torch.Tensor) -> tuple[torch.Tensor, BackboneFeatures]:
        features = self.teacher_backbone.forward_features(images)
        projected = F.normalize(self.teacher_global_projector(features.global_token), dim=-1)
        return projected, features

    def _freeze_teacher(self) -> None:
        for module in (
            self.teacher_backbone,
            self.teacher_global_projector,
            self.teacher_patch_projector,
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_backbone.eval()
        self.teacher_global_projector.eval()
        self.teacher_patch_projector.eval()
        return self


def align_patch_tokens(
    tokens: torch.Tensor,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> torch.Tensor:
    if source_grid == target_grid:
        return tokens
    batch, count, channels = tokens.shape
    source_h, source_w = source_grid
    if count != source_h * source_w:
        raise ValueError(f"Source token count {count} does not match grid {source_grid}")
    feature_map = tokens.transpose(1, 2).reshape(batch, channels, source_h, source_w)
    resized = F.interpolate(feature_map, size=target_grid, mode="bilinear", align_corners=False)
    return resized.flatten(2).transpose(1, 2)


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
