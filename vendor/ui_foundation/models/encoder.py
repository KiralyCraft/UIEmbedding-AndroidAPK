from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ui_foundation.config import ExperimentConfig, experiment_config_from_dict
from ui_foundation.models.backbones import FeatureBackbone, build_backbone
from ui_foundation.models.distillation import MobileNetStudent
from ui_foundation.models.heads import ProjectionMLP


class FoundationEncoder(nn.Module):
    def __init__(
        self,
        backbone: FeatureBackbone,
        projector: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        value = features.global_token
        if self.projector is not None:
            value = self.projector(value)
        return F.normalize(value, dim=-1)


def build_encoder(
    config: ExperimentConfig, checkpoint_path: str = "", *, memory_map: bool = False,
) -> nn.Module:
    selected_checkpoint = checkpoint_path or config.model.checkpoint
    if selected_checkpoint == "":
        backbone = build_backbone(config.model)
        return FoundationEncoder(backbone)

    payload = torch.load(selected_checkpoint, map_location="cpu", weights_only=False, mmap=memory_map)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint: {selected_checkpoint}")
    if "student_model_state_dict" in payload:
        if config.evaluation.representation == "raw_cls":
            raise ValueError("raw_cls requires a foundation backbone, not a distilled MobileNet checkpoint")
        checkpoint_config = experiment_config_from_dict(payload["config"])
        checkpoint_config.model.pretrained = False
        checkpoint_config.model.checkpoint = ""
        if config.model.repo_dir != "":
            checkpoint_config.model.repo_dir = config.model.repo_dir
        model = MobileNetStudent(checkpoint_config, payload.get("teacher_output_dim", checkpoint_config.model.projection_dim))
        model.load_state_dict(payload["student_model_state_dict"], strict=True)
        return _MobileStudentEncoder(model)

    checkpoint_config = experiment_config_from_dict(payload.get("config", config.to_dict()))
    checkpoint_config.model.pretrained = False
    checkpoint_config.model.checkpoint = ""
    if config.model.repo_dir != "":
        checkpoint_config.model.repo_dir = config.model.repo_dir
    backbone = build_backbone(checkpoint_config.model)
    component = config.model.checkpoint_component
    state_key = f"{component}_backbone_state_dict"
    if config.evaluation.representation == "raw_cls" and state_key not in payload:
        raise ValueError(f"raw_cls checkpoint is missing requested component {state_key}")
    if state_key not in payload:
        state_key = "teacher_backbone_state_dict" if "teacher_backbone_state_dict" in payload else "student_backbone_state_dict"
    backbone.load_state_dict(payload[state_key], strict=True)
    if config.evaluation.representation == "raw_cls":
        return FoundationEncoder(backbone)
    projector_key = f"{component}_global_projector_state_dict"
    if projector_key not in payload:
        projector_key = "teacher_global_projector_state_dict"
    projector = None
    if projector_key in payload:
        projector = ProjectionMLP(
            backbone.feature_dim,
            checkpoint_config.model.projection_hidden_dim,
            checkpoint_config.model.projection_dim,
        )
        projector.load_state_dict(payload[projector_key], strict=True)
    return FoundationEncoder(backbone, projector)


class _MobileStudentEncoder(nn.Module):
    def __init__(self, model: MobileNetStudent) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embedding, _, _ = self.model(images)
        return embedding
