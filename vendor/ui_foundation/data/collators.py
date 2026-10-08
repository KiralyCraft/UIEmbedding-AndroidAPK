from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from ui_foundation.config import DataConfig, DistillationConfig, ResolutionBucketConfig, SSLConfig
from ui_foundation.data.records import ScreenRecord
from ui_foundation.data.transforms import BlockMaskGenerator, FullScreenTransform, ResolutionBucketSampler
from ui_foundation.data.xml_targets import ParsedUI, XMLDenseTargetBuilder, XMLTargetParser


class ImageLoader:
    def load(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@dataclass
class SSLBatchCollator:
    data_config: DataConfig
    ssl_config: SSLConfig
    seed: int = 0

    def __post_init__(self) -> None:
        self.image_loader = ImageLoader()
        self.student_transform = FullScreenTransform(self.data_config, strong=True)
        self.teacher_transform = FullScreenTransform(self.data_config, strong=False)
        self.student_buckets = ResolutionBucketSampler(self.data_config.resolution_buckets, self.seed)
        teacher_buckets = self.data_config.teacher_resolution_buckets or self.data_config.resolution_buckets
        self.teacher_buckets = ResolutionBucketSampler(teacher_buckets, self.seed + 1)
        self.xml_targets_enabled = self.data_config.xml_enabled and (
            self.ssl_config.xml_region_weight > 0.0 or self.ssl_config.xml_structure_weight > 0.0
        )
        self.xml_parsing_enabled = self.data_config.xml_enabled and (
            self.xml_targets_enabled or self.data_config.targeted_content_mask_probability > 0.0
        )
        self.xml_parser = (
            XMLTargetParser(
                max_nodes=self.data_config.xml_max_nodes,
                min_box_area_fraction=self.data_config.xml_min_box_area_fraction,
            )
            if self.xml_parsing_enabled
            else None
        )
        self.xml_builder = (
            XMLDenseTargetBuilder(
                patch_size=self.data_config.patch_size,
                attribute_count=self.ssl_config.xml_attribute_count,
                occupancy_grid=self.ssl_config.xml_occupancy_grid,
                count_bins=self.ssl_config.xml_count_bins,
            )
            if self.xml_targets_enabled
            else None
        )
        self.mask_generator = BlockMaskGenerator(
            ratio_min=self.ssl_config.mask_ratio_min,
            ratio_max=self.ssl_config.mask_ratio_max,
            block_min_fraction=self.ssl_config.mask_block_min_fraction,
            block_max_fraction=self.ssl_config.mask_block_max_fraction,
            minimum_masked_patches=self.ssl_config.minimum_masked_patches,
        )

    def __call__(self, records: list[ScreenRecord]) -> dict[str, Any]:
        student_bucket = self.student_buckets.choose()
        teacher_bucket = self.teacher_buckets.choose()
        student_images: list[torch.Tensor] = []
        teacher_images: list[torch.Tensor] = []
        student_valid: list[torch.Tensor] = []
        teacher_valid: list[torch.Tensor] = []
        patch_masks: list[torch.Tensor] = []
        xml_targets: list[dict[str, torch.Tensor]] = []

        for record in records:
            image = self.image_loader.load(record.image_path)
            height, width = image.shape[:2]
            parsed = (
                self.xml_parser.parse(record.xml_path, width, height)
                if self.xml_parser is not None
                else ParsedUI(nodes=(), quality=0.0, raw_node_count=0)
            )
            if record.is_external or parsed.quality < self.data_config.xml_quality_minimum:
                parsed = ParsedUI(nodes=parsed.nodes, quality=0.0, raw_node_count=parsed.raw_node_count)
            student = self.student_transform(image, student_bucket, parsed)
            teacher = self.teacher_transform(image, teacher_bucket, parsed)
            grid_h = student_bucket.height // self.data_config.patch_size
            grid_w = student_bucket.width // self.data_config.patch_size
            patch_mask = self.mask_generator.generate(grid_h, grid_w, student.valid_patch_mask)
            if self.xml_builder is not None:
                xml_targets.append(self.xml_builder.build(parsed, student.geometry))

            student_images.append(student.tensor)
            teacher_images.append(teacher.tensor)
            student_valid.append(student.valid_patch_mask)
            teacher_valid.append(teacher.valid_patch_mask)
            patch_masks.append(patch_mask)

        batch = {
            "student_images": torch.stack(student_images),
            "teacher_images": torch.stack(teacher_images),
            "student_valid_patch_mask": torch.stack(student_valid),
            "teacher_valid_patch_mask": torch.stack(teacher_valid),
            "patch_mask": torch.stack(patch_masks),
            # Grid dimensions are model metadata, not accelerator tensors.
            # Keeping them as tuples avoids a CUDA-to-CPU synchronization in
            # every microbatch.
            "student_grid": (
                student_bucket.height // self.data_config.patch_size,
                student_bucket.width // self.data_config.patch_size,
            ),
            "teacher_grid": (
                teacher_bucket.height // self.data_config.patch_size,
                teacher_bucket.width // self.data_config.patch_size,
            ),
            "record_id": [record.record_id for record in records],
            "package_name": [record.package_name for record in records],
        }
        if self.xml_targets_enabled:
            batch.update(
                {
                    "xml_roles": torch.stack([target["xml_roles"] for target in xml_targets]),
                    "xml_attributes": torch.stack([target["xml_attributes"] for target in xml_targets]),
                    "xml_valid": torch.stack([target["xml_valid"] for target in xml_targets]),
                    "xml_occupancy": torch.stack([target["xml_occupancy"] for target in xml_targets]),
                    "xml_count_bin": torch.stack([target["xml_count_bin"] for target in xml_targets]),
                    "xml_quality": torch.stack([target["xml_quality"] for target in xml_targets]),
                }
            )
        return batch


@dataclass
class DistillationBatchCollator:
    data_config: DataConfig
    distillation_config: DistillationConfig

    def __post_init__(self) -> None:
        self.image_loader = ImageLoader()
        self.teacher_transform = FullScreenTransform(self.data_config, strong=False)
        self.student_transform = FullScreenTransform(self.data_config, strong=True)

    def __call__(self, records: list[ScreenRecord]) -> dict[str, Any]:
        teacher_bucket = self.distillation_config.teacher_input_bucket
        student_bucket = self.distillation_config.student_input_bucket
        teacher_images: list[torch.Tensor] = []
        student_images: list[torch.Tensor] = []
        labels: list[str] = []
        for record in records:
            image = self.image_loader.load(record.image_path)
            teacher = self.teacher_transform(image, teacher_bucket, None)
            student = self.student_transform(image, student_bucket, None)
            teacher_images.append(teacher.tensor)
            student_images.append(student.tensor)
            labels.append(record.structure_key or record.activity_key or record.state_id)
        return {
            "teacher_images": torch.stack(teacher_images),
            "student_images": torch.stack(student_images),
            "record_id": [record.record_id for record in records],
            "package_name": [record.package_name for record in records],
            "label": labels,
        }


@dataclass
class EvaluationBatchCollator:
    data_config: DataConfig
    bucket: ResolutionBucketConfig

    def __post_init__(self) -> None:
        self.image_loader = ImageLoader()
        self.transform = FullScreenTransform(self.data_config, strong=False)

    def __call__(self, records: list[ScreenRecord]) -> dict[str, Any]:
        images: list[torch.Tensor] = []
        for record in records:
            image = self.image_loader.load(record.image_path)
            transformed = self.transform(image, self.bucket, None)
            images.append(transformed.tensor)
        return {
            "images": torch.stack(images),
            "record_id": [record.record_id for record in records],
            "package_name": [record.package_name for record in records],
            "activity_name": [record.activity_name for record in records],
            "structure_id": [record.structure_id for record in records],
            "state_id": [record.state_id for record in records],
            "trace_id": [record.trace_id for record in records],
            "sequence_index": torch.tensor([record.sequence_index for record in records], dtype=torch.long),
            "trace_visits": [record.trace_visits for record in records],
        }
