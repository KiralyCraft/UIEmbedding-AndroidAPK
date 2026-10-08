from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, TypeVar

import yaml


@dataclass(slots=True)
class ResolutionBucketConfig:
    width: int = 384
    height: int = 800
    weight: float = 1.0

    @property
    def patch_tokens(self) -> int:
        return (self.width // 16) * (self.height // 16)


@dataclass(slots=True)
class RuntimeConfig:
    output_dir: str = "runs/default"
    cache_dir: str = "data"
    seed: int = 42
    device: str = "auto"
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    in_order: bool = True
    amp: str = "auto"
    compile: bool = False
    deterministic: bool = False
    progress_bars: bool = True
    log_every_steps: int = 25
    save_every_steps: int = 2000
    keep_last_checkpoints: int = 3
    distributed_backend: str = "nccl"
    find_unused_parameters: bool = False


@dataclass(slots=True)
class DataConfig:
    dataset: str = "mogui"
    source: str = "local"
    root: str = ""
    hf_repo_id: str = ""
    hf_revision: str = "main"
    hf_download_dir: str = ""
    hf_allow_patterns: list[str] = field(default_factory=list)
    hf_ignore_patterns: list[str] = field(default_factory=list)
    hf_token_env: str = "HF_TOKEN"
    hf_max_workers: int = 8
    hf_force_download: bool = False
    hf_local_files_only: bool = False
    extract_archives: bool = True
    remove_archives_after_extraction: bool = False
    verify_archives: bool = True
    skip_corrupt_archives: bool = False
    salvage_truncated_archives: bool = False

    manifest: str = ""
    rebuild_manifest: bool = False
    max_records: Optional[int] = None
    shuffle_buffer_size: int = 8192
    include_external: bool = False
    deduplicate_by_state: bool = False
    scan_image_metadata: bool = True
    scan_hashes: bool = False
    scan_xml_package_mismatch: bool = True
    mobileviews_scan_workers: int = 8

    split_seed: int = 2026
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    test_fraction: float = 0.10

    patch_size: int = 16
    resize_mode: str = "fit_pad"
    resolution_buckets: list[ResolutionBucketConfig] = field(
        default_factory=lambda: [ResolutionBucketConfig()]
    )
    teacher_resolution_buckets: list[ResolutionBucketConfig] = field(default_factory=list)
    normalize_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    color_jitter_strength: float = 0.10
    grayscale_probability: float = 0.05
    blur_probability: float = 0.05
    jpeg_probability: float = 0.10
    small_translation_fraction: float = 0.015
    system_bar_dropout_probability: float = 0.10
    targeted_content_mask_probability: float = 0.10

    xml_enabled: bool = True
    xml_max_nodes: int = 256
    xml_min_box_area_fraction: float = 0.00001
    xml_quality_minimum: float = 0.25


@dataclass(slots=True)
class ModelConfig:
    family: str = "dinov3"
    loader: str = "torch_hub"
    hub_model: str = "dinov3_vits16"
    repo_dir: str = ""
    weights: str = ""
    pretrained: bool = True
    checkpoint: str = ""
    checkpoint_component: str = "teacher"
    patch_size: int = 16
    feature_dim: int = 384
    projection_dim: int = 256
    projection_hidden_dim: int = 1024
    predictor_hidden_dim: int = 1024
    embedding_dim: int = 256
    mobilenet_name: str = "mobilenetv4_conv_small.e2400_r224_in1k"
    dropout: float = 0.0
    mock_feature_dim: int = 96


@dataclass(slots=True)
class SSLConfig:
    enabled: bool = True
    global_weight: float = 1.0
    masked_patch_weight: float = 1.0
    cross_resolution_weight: float = 0.10
    xml_region_weight: float = 0.0
    xml_structure_weight: float = 0.0
    variance_weight: float = 0.05
    covariance_weight: float = 0.005

    mask_ratio_min: float = 0.30
    mask_ratio_max: float = 0.50
    mask_block_min_fraction: float = 0.03
    mask_block_max_fraction: float = 0.20
    minimum_masked_patches: int = 8

    teacher_momentum_start: float = 0.996
    teacher_momentum_end: float = 1.0
    teacher_warm_start: bool = True
    target_normalize: bool = True
    patch_loss: str = "cosine"
    global_loss: str = "cosine"

    xml_role_classes: int = 11
    xml_attribute_count: int = 8
    xml_count_bins: int = 8
    xml_occupancy_grid: int = 8


@dataclass(slots=True)
class DistillationConfig:
    enabled: bool = False
    teacher_checkpoint: str = ""
    global_weight: float = 1.0
    relational_weight: float = 0.50
    dense_weight: float = 0.25
    supervised_place_weight: float = 0.0
    temperature: float = 0.07
    teacher_input_bucket: ResolutionBucketConfig = field(
        default_factory=lambda: ResolutionBucketConfig(width=512, height=1040)
    )
    student_input_bucket: ResolutionBucketConfig = field(
        default_factory=lambda: ResolutionBucketConfig(width=384, height=800)
    )


@dataclass(slots=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1.0e-4
    head_learning_rate: float = 3.0e-4
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    exclude_bias_and_norm_from_weight_decay: bool = True


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 1
    max_steps: int = 10000
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    validation_every_steps: int = 1000
    validation_steps: int = 100
    freeze_backbone_steps: int = 0
    resume: str = ""
    init_checkpoint: str = ""
    drop_last: bool = True


@dataclass(slots=True)
class EvaluationConfig:
    enabled: bool = True
    representation: str = "projected"
    split: str = "test"
    batch_size: int = 16
    max_records: Optional[int] = 25000
    label: str = "structure_id"
    within_package: bool = True
    exclude_same_trace_neighbors: int = 1
    recalls: list[int] = field(default_factory=lambda: [1, 5, 10])
    save_embeddings: bool = True
    embedding_output: str = ""


@dataclass(slots=True)
class WandbConfig:
    enabled: bool = True
    project: str = "ui-foundation"
    entity: str = ""
    mode: str = "online"
    run_name: str = ""
    group: str = "F0-F6"
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    run_id: str = ""
    resume: str = "allow"
    save_code: bool = True
    log_visuals: bool = True
    visual_every_steps: int = 1000
    visual_samples: int = 6


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "ui_foundation"
    experiment: str = "F1"
    task: str = "ssl_pretrain"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        valid_tasks = {
            "prepare",
            "frozen_evaluate",
            "ssl_pretrain",
            "distill",
        }
        if self.task not in valid_tasks:
            raise ValueError(f"Unsupported task {self.task!r}; expected one of {sorted(valid_tasks)}")
        if self.evaluation.representation not in {"projected", "raw_cls"}:
            raise ValueError("evaluation.representation must be 'projected' or 'raw_cls'")
        if self.data.dataset not in {"mogui", "mobileviews"}:
            raise ValueError("data.dataset must be 'mogui' or 'mobileviews'")
        if self.data.source not in {"local", "huggingface"}:
            raise ValueError("data.source must be 'local' or 'huggingface'")
        if self.data.source == "local" and self.data.root == "":
            raise ValueError("data.root is required when data.source=local")
        if self.data.source == "huggingface" and self.data.hf_repo_id == "":
            raise ValueError("data.hf_repo_id is required when data.source=huggingface")
        if self.data.patch_size <= 0:
            raise ValueError("data.patch_size must be positive")
        if self.data.resize_mode != "fit_pad":
            raise ValueError("Only data.resize_mode=fit_pad is currently implemented")
        if self.model.family == "dinov3" and self.data.patch_size != self.model.patch_size:
            raise ValueError("DINOv3 data.patch_size and model.patch_size must match")
        for bucket in self.data.resolution_buckets + self.data.teacher_resolution_buckets:
            _validate_bucket(bucket, self.data.patch_size)
        _validate_bucket(self.distillation.teacher_input_bucket, self.data.patch_size)
        _validate_bucket(self.distillation.student_input_bucket, self.data.patch_size)
        if self.training.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if self.training.max_steps <= 0 and self.task in {"ssl_pretrain", "distill"}:
            raise ValueError("training.max_steps must be positive for training tasks")
        if not self.data.xml_enabled and (
            self.ssl.xml_region_weight > 0.0 or self.ssl.xml_structure_weight > 0.0
        ):
            raise ValueError("data.xml_enabled must be true when an XML loss has non-zero weight")
        if not 0.0 <= self.ssl.mask_ratio_min <= self.ssl.mask_ratio_max < 1.0:
            raise ValueError("SSL mask ratios must satisfy 0 <= min <= max < 1")
        if self.model.family == "dinov3" and self.model.loader == "torch_hub":
            if self.model.repo_dir == "":
                raise ValueError("model.repo_dir is required for the DINOv3 torch_hub loader")
            if self.model.pretrained and self.model.weights == "":
                raise ValueError("model.weights is required for pretrained local DINOv3 loading")
        fractions = self.data.train_fraction + self.data.val_fraction + self.data.test_fraction
        if abs(fractions - 1.0) > 1.0e-6:
            raise ValueError("data split fractions must sum to 1.0")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


T = TypeVar("T")
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _validate_bucket(bucket: ResolutionBucketConfig, patch_size: int) -> None:
    if bucket.width <= 0 or bucket.height <= 0:
        raise ValueError("Resolution bucket dimensions must be positive")
    if bucket.width % patch_size != 0 or bucket.height % patch_size != 0:
        raise ValueError(
            f"Resolution bucket {bucket.width}x{bucket.height} must be divisible by patch size {patch_size}"
        )
    if bucket.weight <= 0.0:
        raise ValueError("Resolution bucket weights must be positive")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        unresolved = _ENV_PATTERN.findall(expanded)
        if len(unresolved) > 0:
            names = ", ".join(sorted(set(unresolved)))
            raise ValueError(f"Unresolved environment variables in configuration: {names}")
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in stack + (resolved,))
        raise ValueError(f"Configuration inheritance cycle: {chain}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    extends = raw.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    merged: dict[str, Any] = {}
    for parent in extends:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        merged = _deep_merge(merged, _load_yaml_with_extends(parent_path, stack + (resolved,)))
    return _deep_merge(merged, raw)


def _set_nested_value(mapping: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: MutableMapping[str, Any] = mapping
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, MutableMapping):
            raise ValueError(f"Cannot override {dotted_key!r}; {part!r} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def _apply_overrides(raw: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value")
        key, serialized = override.split("=", 1)
        _set_nested_value(raw, key, yaml.safe_load(serialized))


def _construct_dataclass(cls: type[T], values: Mapping[str, Any] | None) -> T:
    values = dict(values or {})
    allowed = {item.name for item in dataclasses.fields(cls)}
    unknown = sorted(set(values) - allowed)
    if len(unknown) > 0:
        raise ValueError(f"Unknown keys for {cls.__name__}: {unknown}")
    return cls(**values)


def _construct_buckets(values: list[Mapping[str, Any]] | None) -> list[ResolutionBucketConfig]:
    return [_construct_dataclass(ResolutionBucketConfig, item) for item in values or []]


def experiment_config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    values = dict(raw)
    data_values = dict(values.get("data", {}))
    if "resolution_buckets" in data_values:
        data_values["resolution_buckets"] = _construct_buckets(data_values["resolution_buckets"])
    if "teacher_resolution_buckets" in data_values:
        data_values["teacher_resolution_buckets"] = _construct_buckets(data_values["teacher_resolution_buckets"])

    distillation_values = dict(values.get("distillation", {}))
    if "teacher_input_bucket" in distillation_values:
        distillation_values["teacher_input_bucket"] = _construct_dataclass(
            ResolutionBucketConfig, distillation_values["teacher_input_bucket"]
        )
    if "student_input_bucket" in distillation_values:
        distillation_values["student_input_bucket"] = _construct_dataclass(
            ResolutionBucketConfig, distillation_values["student_input_bucket"]
        )

    config = ExperimentConfig(
        name=str(values.get("name", "ui_foundation")),
        experiment=str(values.get("experiment", "F1")),
        task=str(values.get("task", "ssl_pretrain")),
        runtime=_construct_dataclass(RuntimeConfig, values.get("runtime")),
        data=_construct_dataclass(DataConfig, data_values),
        model=_construct_dataclass(ModelConfig, values.get("model")),
        ssl=_construct_dataclass(SSLConfig, values.get("ssl")),
        distillation=_construct_dataclass(DistillationConfig, distillation_values),
        optimizer=_construct_dataclass(OptimizerConfig, values.get("optimizer")),
        training=_construct_dataclass(TrainingConfig, values.get("training")),
        evaluation=_construct_dataclass(EvaluationConfig, values.get("evaluation")),
        wandb=_construct_dataclass(WandbConfig, values.get("wandb")),
    )
    config.validate()
    return config


def load_experiment_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = _load_yaml_with_extends(Path(path))
    _apply_overrides(raw, list(overrides or []))
    raw = _expand_environment(raw)
    return experiment_config_from_dict(raw)
