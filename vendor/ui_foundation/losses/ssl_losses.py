from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ui_foundation.config import SSLConfig
from ui_foundation.models.ssl_model import SSLForwardOutput


@dataclass(slots=True)
class LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


class UIFoundationSSLLoss:
    def __init__(self, config: SSLConfig) -> None:
        self.config = config

    def __call__(self, output: SSLForwardOutput, batch: dict[str, torch.Tensor]) -> LossOutput:
        global_loss = cosine_regression(output.student_global, output.teacher_global)
        patch_mask = batch["patch_mask"] & batch["student_valid_patch_mask"]
        patch_loss = masked_cosine_regression(output.student_patches, output.teacher_patches, patch_mask)
        cross_resolution = cosine_regression(output.student_patch_mean, output.teacher_patch_mean)
        variance = variance_loss(output.student_global)
        covariance = covariance_loss(output.student_global)

        zero = output.student_global.new_zeros(())
        if self.config.xml_region_weight > 0.0:
            xml_role, xml_attributes = self._xml_region_losses(output, batch)
        else:
            xml_role, xml_attributes = zero, zero
        if self.config.xml_structure_weight > 0.0:
            xml_count, xml_occupancy = self._xml_structure_losses(output, batch)
        else:
            xml_count, xml_occupancy = zero, zero
        xml_region = xml_role + xml_attributes
        xml_structure = xml_count + xml_occupancy

        components = {
            "global": global_loss,
            "masked_patch": patch_loss,
            "cross_resolution": cross_resolution,
            "variance": variance,
            "covariance": covariance,
            "xml_role": xml_role,
            "xml_attributes": xml_attributes,
            "xml_count": xml_count,
            "xml_occupancy": xml_occupancy,
        }
        total = (
            self.config.global_weight * global_loss
            + self.config.masked_patch_weight * patch_loss
            + self.config.cross_resolution_weight * cross_resolution
            + self.config.variance_weight * variance
            + self.config.covariance_weight * covariance
            + self.config.xml_region_weight * xml_region
            + self.config.xml_structure_weight * xml_structure
        )
        return LossOutput(total=total, components=components)

    def _xml_region_losses(
        self,
        output: SSLForwardOutput,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output.xml_role_logits is None or output.xml_attribute_logits is None:
            raise RuntimeError("XML region loss is enabled but the model did not return XML region logits")
        valid = batch["xml_valid"] & batch["student_valid_patch_mask"] & ~batch["patch_mask"]
        quality = batch["xml_quality"].unsqueeze(1)
        weights = valid.to(output.xml_role_logits.dtype) * quality
        denominator = weights.sum().clamp_min(1.0)
        role_loss = F.cross_entropy(
            output.xml_role_logits.transpose(1, 2),
            batch["xml_roles"],
            reduction="none",
        )
        role_loss = (role_loss * weights).sum() / denominator
        attribute_loss = F.binary_cross_entropy_with_logits(
            output.xml_attribute_logits,
            batch["xml_attributes"],
            reduction="none",
        ).mean(dim=-1)
        attribute_loss = (attribute_loss * weights).sum() / denominator
        return role_loss, attribute_loss

    def _xml_structure_losses(
        self,
        output: SSLForwardOutput,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output.xml_count_logits is None or output.xml_occupancy_logits is None:
            raise RuntimeError("XML structure loss is enabled but the model did not return XML structure logits")
        quality = batch["xml_quality"]
        denominator = quality.sum().clamp_min(1.0)
        count_loss = F.cross_entropy(
            output.xml_count_logits,
            batch["xml_count_bin"],
            reduction="none",
        )
        count_loss = (count_loss * quality).sum() / denominator
        occupancy_loss = F.binary_cross_entropy_with_logits(
            output.xml_occupancy_logits,
            batch["xml_occupancy"],
            reduction="none",
        ).mean(dim=-1)
        occupancy_loss = (occupancy_loss * quality).sum() / denominator
        return count_loss, occupancy_loss


def cosine_regression(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher.detach(), dim=-1)
    return (2.0 - 2.0 * (student * teacher).sum(dim=-1)).mean()


def masked_cosine_regression(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher.detach(), dim=-1)
    values = 2.0 - 2.0 * (student * teacher).sum(dim=-1)
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def variance_loss(value: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    if value.shape[0] <= 1:
        return value.new_zeros(())
    std = torch.sqrt(value.var(dim=0, unbiased=False) + 1.0e-4)
    return F.relu(target_std - std).mean()


def covariance_loss(value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] <= 1:
        return value.new_zeros(())
    centered = value - value.mean(dim=0)
    covariance = centered.T @ centered / max(value.shape[0] - 1, 1)
    off_diagonal = covariance.flatten()[:-1].view(covariance.shape[0] - 1, covariance.shape[0] + 1)[:, 1:].flatten()
    return off_diagonal.pow(2).mean()
