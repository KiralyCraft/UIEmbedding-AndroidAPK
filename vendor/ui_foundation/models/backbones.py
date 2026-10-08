from __future__ import annotations

import copy
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ui_foundation.config import ExperimentConfig, ModelConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BackboneFeatures:
    global_token: torch.Tensor
    patch_tokens: torch.Tensor
    grid_size: tuple[int, int]

    def patch_map(self) -> torch.Tensor:
        batch, tokens, channels = self.patch_tokens.shape
        height, width = self.grid_size
        if tokens != height * width:
            raise ValueError(f"Patch token count {tokens} does not match grid {height}x{width}")
        return self.patch_tokens.transpose(1, 2).reshape(batch, channels, height, width)


class FeatureBackbone(nn.Module):
    feature_dim: int
    patch_size: int

    def forward_features(self, images: torch.Tensor, patch_mask: torch.Tensor | None = None) -> BackboneFeatures:
        raise NotImplementedError


class DINOv3Backbone(FeatureBackbone):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.loader != "torch_hub":
            raise ValueError("SSL training currently requires model.loader=torch_hub for native DINOv3 masks")
        repo_dir = Path(config.repo_dir).expanduser().resolve()
        if not repo_dir.is_dir():
            raise NotADirectoryError(f"DINOv3 repository does not exist: {repo_dir}")
        weights_path = Path(config.weights).expanduser()
        local_weights = config.pretrained and config.weights != "" and weights_path.is_file()
        kwargs: dict[str, Any] = {"pretrained": config.pretrained and not local_weights}
        if config.pretrained and not local_weights:
            kwargs["weights"] = config.weights
        self.model = torch.hub.load(str(repo_dir), config.hub_model, source="local", **kwargs)
        if local_weights:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)
        self.feature_dim = int(getattr(self.model, "num_features", getattr(self.model, "embed_dim", config.feature_dim)))
        raw_patch_size = getattr(self.model, "patch_size", config.patch_size)
        if isinstance(raw_patch_size, tuple):
            if raw_patch_size[0] != raw_patch_size[1]:
                raise ValueError(f"Only square DINOv3 patches are supported, received {raw_patch_size}")
            raw_patch_size = raw_patch_size[0]
        self.patch_size = int(raw_patch_size)
        if config.checkpoint != "":
            load_backbone_component(self, config.checkpoint, config.checkpoint_component)

    def forward_features(self, images: torch.Tensor, patch_mask: torch.Tensor | None = None) -> BackboneFeatures:
        output = self.model.forward_features(images, masks=patch_mask)
        global_token = output["x_norm_clstoken"]
        patch_tokens = output["x_norm_patchtokens"]
        grid_size = (images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size)
        return BackboneFeatures(global_token=global_token, patch_tokens=patch_tokens, grid_size=grid_size)


class MockPatchBackbone(FeatureBackbone):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.feature_dim = config.mock_feature_dim
        self.patch_embed = nn.Conv2d(3, self.feature_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward_features(self, images: torch.Tensor, patch_mask: torch.Tensor | None = None) -> BackboneFeatures:
        feature_map = self.patch_embed(images)
        batch, channels, height, width = feature_map.shape
        patches = feature_map.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        if patch_mask is not None:
            patches = torch.where(patch_mask.unsqueeze(-1), self.mask_token.to(patches.dtype), patches)
        global_token = patches.mean(dim=1)
        return BackboneFeatures(global_token=global_token, patch_tokens=patches, grid_size=(height, width))


class TimmMobileNetBackbone(FeatureBackbone):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        try:
            timm = importlib.import_module("timm")
        except ImportError as exc:
            raise RuntimeError("MobileNet experiments require timm>=1.0.20") from exc
        self.model = timm.create_model(
            config.mobilenet_name,
            pretrained=config.pretrained,
            num_classes=0,
            global_pool="",
        )
        self.patch_size = 32
        inferred = getattr(self.model, "num_features", config.feature_dim)
        self.feature_dim = int(inferred)
        if config.checkpoint != "":
            load_backbone_component(self, config.checkpoint, config.checkpoint_component)

    def forward_features(self, images: torch.Tensor, patch_mask: torch.Tensor | None = None) -> BackboneFeatures:
        del patch_mask
        feature_map = self.model.forward_features(images)
        if isinstance(feature_map, (tuple, list)):
            feature_map = feature_map[-1]
        if feature_map.ndim != 4:
            raise ValueError(f"Expected BCHW MobileNet feature map, received {tuple(feature_map.shape)}")
        batch, channels, height, width = feature_map.shape
        patch_tokens = feature_map.flatten(2).transpose(1, 2)
        global_token = F.adaptive_avg_pool2d(feature_map, 1).reshape(batch, channels)
        return BackboneFeatures(global_token=global_token, patch_tokens=patch_tokens, grid_size=(height, width))


def build_backbone(config: ModelConfig) -> FeatureBackbone:
    if config.family == "dinov3":
        return DINOv3Backbone(config)
    if config.family == "mobilenetv4":
        return TimmMobileNetBackbone(config)
    if config.family == "mock":
        return MockPatchBackbone(config)
    raise ValueError(f"Unsupported model family: {config.family}")


def clone_backbone(backbone: FeatureBackbone) -> FeatureBackbone:
    return copy.deepcopy(backbone)


def load_backbone_component(backbone: nn.Module, checkpoint_path: str, component: str) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = None
    candidates = [
        f"{component}_backbone_state_dict",
        f"{component}_state_dict",
        "backbone_state_dict",
        "encoder_state_dict",
        "model_state_dict",
    ]
    for key in candidates:
        if isinstance(payload, dict) and key in payload:
            state = payload[key]
            break
    if state is None:
        if isinstance(payload, dict):
            state = payload
        else:
            raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")
    cleaned = strip_prefixes(state, ("module.", "backbone.", "student_backbone.", "teacher_backbone."))
    incompatible = backbone.load_state_dict(cleaned, strict=False)
    if len(incompatible.missing_keys) > 0:
        LOGGER.warning("Missing backbone keys from %s: %s", checkpoint_path, incompatible.missing_keys[:20])
    if len(incompatible.unexpected_keys) > 0:
        LOGGER.warning("Unexpected backbone keys from %s: %s", checkpoint_path, incompatible.unexpected_keys[:20])


def strip_prefixes(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    result = dict(state)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            matching = {key[len(prefix) :]: value for key, value in result.items() if key.startswith(prefix)}
            if len(matching) == len(result) and len(matching) > 0:
                result = matching
                changed = True
                break
    return result
