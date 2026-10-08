from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from ui_foundation.config import DataConfig, ResolutionBucketConfig
from ui_foundation.data.xml_targets import ParsedUI, ResizeGeometry

# DataLoader already parallelizes image work across processes. OpenCV's own
# thread pool would multiply that worker count and oversubscribe the host.
cv2.setNumThreads(1)


@dataclass(frozen=True, slots=True)
class TransformedImage:
    tensor: torch.Tensor
    geometry: ResizeGeometry
    valid_patch_mask: torch.Tensor


class ResolutionBucketSampler:
    def __init__(self, buckets: list[ResolutionBucketConfig], seed: int = 0) -> None:
        if len(buckets) == 0:
            raise ValueError("At least one resolution bucket is required")
        self.buckets = list(buckets)
        self.seed = seed

    def choose(self) -> ResolutionBucketConfig:
        weights = [bucket.weight for bucket in self.buckets]
        # DataLoader seeds Python's process-local RNG per worker. Using it here
        # avoids cloned collator instances replaying the same bucket sequence.
        return random.choices(self.buckets, weights=weights, k=1)[0]


class UIPhotometricAugmenter:
    def __init__(self, config: DataConfig, strong: bool) -> None:
        self.config = config
        self.strong = strong

    def __call__(self, image: np.ndarray) -> np.ndarray:
        result = image.astype(np.float32, copy=True)
        strength = self.config.color_jitter_strength * (1.0 if self.strong else 0.35)
        if strength > 0.0:
            brightness = random.uniform(1.0 - strength, 1.0 + strength)
            contrast = random.uniform(1.0 - strength, 1.0 + strength)
            saturation = random.uniform(1.0 - strength, 1.0 + strength)
            result *= brightness
            grayscale = (
                result[..., 0] * 0.299
                + result[..., 1] * 0.587
                + result[..., 2] * 0.114
            )
            result = result * contrast + grayscale.mean() * (1.0 - contrast)
            grayscale = (
                result[..., 0] * 0.299
                + result[..., 1] * 0.587
                + result[..., 2] * 0.114
            )[..., None]
            result = result * saturation + grayscale * (1.0 - saturation)
        result = np.clip(result, 0.0, 255.0).astype(np.uint8)
        gray_probability = self.config.grayscale_probability * (1.0 if self.strong else 0.25)
        if random.random() < gray_probability:
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        blur_probability = self.config.blur_probability * (1.0 if self.strong else 0.25)
        if random.random() < blur_probability:
            result = cv2.GaussianBlur(result, (0, 0), sigmaX=random.uniform(0.1, 1.2))
        jpeg_probability = self.config.jpeg_probability * (1.0 if self.strong else 0.25)
        if random.random() < jpeg_probability:
            quality = random.randint(65, 95)
            success, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            if success:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is not None:
                    result = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(result)


class FullScreenTransform:
    def __init__(self, config: DataConfig, strong: bool) -> None:
        self.config = config
        self.augment = UIPhotometricAugmenter(config, strong=strong)
        self.strong = strong
        self.mean = tuple(config.normalize_mean)
        self.std = tuple(config.normalize_std)
        self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        self.std_tensor = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)

    def __call__(
        self,
        image: Image.Image | np.ndarray,
        bucket: ResolutionBucketConfig,
        parsed_ui: ParsedUI | None = None,
    ) -> TransformedImage:
        source = self._as_rgb_array(image)
        if self.strong and parsed_ui is not None:
            source = self._maybe_mask_content(source, parsed_ui)
        if self.strong and random.random() < self.config.system_bar_dropout_probability:
            source = self._drop_system_bars(source)
        canvas, geometry = self._fit_pad(source, bucket)
        tensor = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).float().div_(255.0)
        tensor.sub_(self.mean_tensor).div_(self.std_tensor)
        valid_patch_mask = geometry_valid_patch_mask(geometry, self.config.patch_size)
        return TransformedImage(tensor=tensor, geometry=geometry, valid_patch_mask=valid_patch_mask)

    @staticmethod
    def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB HWC image, got shape={image.shape}")
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _fit_pad(self, image: np.ndarray, bucket: ResolutionBucketConfig) -> tuple[np.ndarray, ResizeGeometry]:
        target_width = bucket.width
        target_height = bucket.height
        source_height, source_width = image.shape[:2]
        scale = min(target_width / source_width, target_height / source_height)
        scaled_width = max(1, min(target_width, int(round(source_width * scale))))
        scaled_height = max(1, min(target_height, int(round(source_height * scale))))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (scaled_width, scaled_height), interpolation=interpolation)
        # Color/blur/JPEG augmentation is substantially cheaper after spatial
        # downsampling and leaves the neutral letterbox padding unchanged.
        resized = self.augment(resized)

        jitter_x = 0
        jitter_y = 0
        if self.strong and self.config.small_translation_fraction > 0.0:
            max_x = int(target_width * self.config.small_translation_fraction)
            max_y = int(target_height * self.config.small_translation_fraction)
            jitter_x = random.randint(-max_x, max_x)
            jitter_y = random.randint(-max_y, max_y)
        offset_x = (target_width - scaled_width) // 2 + jitter_x
        offset_y = (target_height - scaled_height) // 2 + jitter_y
        offset_x = min(max(offset_x, 0), target_width - scaled_width)
        offset_y = min(max(offset_y, 0), target_height - scaled_height)

        fill = tuple(int(round(value * 255.0)) for value in self.mean)
        canvas = np.empty((target_height, target_width, 3), dtype=np.uint8)
        canvas[...] = fill
        canvas[offset_y : offset_y + scaled_height, offset_x : offset_x + scaled_width] = resized
        geometry = ResizeGeometry(
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            scaled_width=scaled_width,
            scaled_height=scaled_height,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        return canvas, geometry

    def _drop_system_bars(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        height, width = result.shape[:2]
        band = max(1, int(round(height * 0.035)))
        top_color = result[min(height - 1, band), max(0, width // 2)].copy()
        bottom_color = result[max(0, height - band - 1), max(0, width // 2)].copy()
        result[:band, :] = top_color
        result[height - band :, :] = bottom_color
        return result

    def _maybe_mask_content(self, image: np.ndarray, parsed_ui: ParsedUI) -> np.ndarray:
        if random.random() >= self.config.targeted_content_mask_probability:
            return image
        result = image.copy()
        height, width = result.shape[:2]
        candidates = [node for node in parsed_ui.nodes if node.role in {1, 2}]
        random.shuffle(candidates)
        limit = max(1, min(len(candidates), int(math.sqrt(max(len(candidates), 1)))))
        for node in candidates[:limit]:
            left = int(node.left * width)
            top = int(node.top * height)
            right = int(node.right * width)
            bottom = int(node.bottom * height)
            if right <= left or bottom <= top:
                continue
            sample_x = min(width - 1, max(0, (left + right) // 2))
            sample_y = min(height - 1, max(0, (top + bottom) // 2))
            result[top:bottom, left:right] = result[sample_y, sample_x]
        return result


def geometry_valid_patch_mask(geometry: ResizeGeometry, patch_size: int) -> torch.Tensor:
    grid_h = geometry.target_height // patch_size
    grid_w = geometry.target_width // patch_size
    center_x = (torch.arange(grid_w, dtype=torch.float32) + 0.5) * patch_size
    center_y = (torch.arange(grid_h, dtype=torch.float32) + 0.5) * patch_size
    valid_x = (center_x >= geometry.offset_x) & (
        center_x < geometry.offset_x + geometry.scaled_width
    )
    valid_y = (center_y >= geometry.offset_y) & (
        center_y < geometry.offset_y + geometry.scaled_height
    )
    return (valid_y[:, None] & valid_x[None, :]).flatten()


class BlockMaskGenerator:
    def __init__(
        self,
        ratio_min: float,
        ratio_max: float,
        block_min_fraction: float,
        block_max_fraction: float,
        minimum_masked_patches: int,
    ) -> None:
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.block_min_fraction = block_min_fraction
        self.block_max_fraction = block_max_fraction
        self.minimum_masked_patches = minimum_masked_patches

    def generate(self, grid_h: int, grid_w: int, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_grid = valid_mask.reshape(grid_h, grid_w)
        valid_count = int(valid_grid.sum())
        target = max(self.minimum_masked_patches, int(valid_count * random.uniform(self.ratio_min, self.ratio_max)))
        target = min(target, valid_count)
        result = torch.zeros((grid_h, grid_w), dtype=torch.bool)
        attempts = 0
        while int(result.sum()) < target and attempts < target * 8 + 64:
            attempts += 1
            area_fraction = random.uniform(self.block_min_fraction, self.block_max_fraction)
            block_area = max(1, int(valid_count * area_fraction))
            aspect = math.exp(random.uniform(math.log(0.4), math.log(2.5)))
            block_h = max(1, min(grid_h, int(round(math.sqrt(block_area / aspect)))))
            block_w = max(1, min(grid_w, int(round(math.sqrt(block_area * aspect)))))
            row = random.randint(0, max(0, grid_h - block_h))
            column = random.randint(0, max(0, grid_w - block_w))
            result[row : row + block_h, column : column + block_w] = True
            result &= valid_grid
        if int(result.sum()) > target:
            indices = torch.nonzero(result.flatten(), as_tuple=False).flatten()
            keep = indices[torch.randperm(len(indices))[:target]]
            trimmed = torch.zeros(grid_h * grid_w, dtype=torch.bool)
            trimmed[keep] = True
            return trimmed
        return result.flatten()
