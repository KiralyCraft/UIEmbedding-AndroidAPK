from __future__ import annotations

import base64
import binascii
import math
import struct
from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Login(StrictModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class RunMetadata(StrictModel):
    id: UUID
    device_id: UUID
    capture_session_id: UUID
    model_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    package_name: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
    app_version: str | None = Field(default=None, max_length=128)
    app_version_code: int | None = Field(default=None, ge=0)
    device_model: str = Field(max_length=128)
    android_sdk: int = Field(ge=30, le=100)
    start_wall_ms: int = Field(ge=946684800000, le=4102444800000)
    start_elapsed_ns: int = Field(ge=0, le=9223372036854775807)
    revision: int = Field(ge=1, le=1000000)
    end_wall_ms: int | None = Field(default=None, ge=946684800000, le=4102444800000)
    end_elapsed_ns: int | None = Field(default=None, ge=0, le=9223372036854775807)
    end_reason: str | None = Field(default=None, max_length=80)
    expected_samples: int | None = Field(default=None, ge=0, le=2147483647)
    recording_policy: Literal["new_frames_only_v1"] = "new_frames_only_v1"

    @model_validator(mode="after")
    def coherent_end(self):
        fields = (self.end_wall_ms, self.end_elapsed_ns, self.end_reason, self.expected_samples)
        if any(v is None for v in fields) and any(v is not None for v in fields):
            raise ValueError("A closed run requires all end fields and expected_samples")
        if self.end_elapsed_ns is not None and self.end_elapsed_ns < self.start_elapsed_ns:
            raise ValueError("Run end precedes start")
        return self


class WindowContext(StrictModel):
    screen_kind: Literal["application", "system_overlay"]
    keyboard_visible: bool
    visible_packages: list[Annotated[str, Field(min_length=1, max_length=255, pattern=r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")]] = Field(max_length=64)


class FrameSample(StrictModel):
    sequence: int = Field(ge=0, le=2147483646)
    wall_ms: int = Field(ge=946684800000, le=4102444800000)
    elapsed_ns: int = Field(ge=0, le=9223372036854775807)
    image_timestamp_ns: int = Field(ge=0, le=9223372036854775807)
    activity: str | None = Field(default=None, max_length=512)
    activity_source: Literal["usage_stats", "unknown"]
    window_class: str | None = Field(default=None, max_length=512)
    window_context: WindowContext | None = None
    label_age_ms: int = Field(ge=0, le=9223372036854775807)
    source_width: int = Field(ge=1, le=32768)
    source_height: int = Field(ge=1, le=32768)
    rotation: int = Field(ge=0, le=3)
    target_fps: float = Field(gt=0, le=120)
    preprocess_ms: float = Field(ge=0, le=60000)
    inference_ms: float = Field(ge=0, le=60000)
    head_ms: float = Field(ge=0, le=60000)
    processing_backend: Literal["litert_gpu_backbone_cpu_head", "litert_opencl_full_encoder"] = "litert_gpu_backbone_cpu_head"
    preprocessing_backend: Literal["opencv_cpu", "opencl"] = "opencv_cpu"
    capture_source: Literal["media_projection", "accessibility"] = "media_projection"
    sampling_mode: Literal["MAXIMUM_DETAIL", "BALANCED", "BATTERY_SAVER"] = "MAXIMUM_DETAIL"
    thermal_status: int = Field(ge=0, le=6)
    embedding_b64: str = Field(min_length=2048, max_length=2048)

    @field_validator("embedding_b64")
    @classmethod
    def check_embedding(cls, value: str) -> str:
        try:
            raw = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid embedding base64") from exc
        if len(raw) != 384 * 4:
            raise ValueError("Expected 384 little-endian float32 values")
        vector = struct.unpack("<384f", raw)
        if all(math.isfinite(x) for x in vector) is False:
            raise ValueError("Embedding contains non-finite values")
        norm = math.sqrt(sum(float(x) * float(x) for x in vector))
        if 0.99 <= norm <= 1.01:
            return value
        raise ValueError("Embedding is not L2 normalized")

    @model_validator(mode="after")
    def activity_consistent(self):
        if (self.activity is None) != (self.activity_source == "unknown"):
            raise ValueError("Activity and its provenance disagree")
        return self


class Ingest(StrictModel):
    run: RunMetadata
    samples: list[FrameSample] = Field(max_length=128)

    @model_validator(mode="after")
    def coherent_batch(self):
        indices = [s.sequence for s in self.samples]
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate sequence within a batch")
        ordered = sorted(self.samples, key=lambda s: s.sequence)
        if any(b.elapsed_ns <= a.elapsed_ns for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Sample monotonic times must increase with sequence")
        for sample in ordered:
            if sample.elapsed_ns < self.run.start_elapsed_ns:
                raise ValueError("Sample precedes run")
            if self.run.end_elapsed_ns is not None and sample.elapsed_ns > self.run.end_elapsed_ns:
                raise ValueError("Sample follows closed run")
            if self.run.expected_samples is not None and sample.sequence >= self.run.expected_samples:
                raise ValueError("Sample exceeds the closed run's sequence range")
        return self


class CandidateResult(StrictModel):
    target_fps: float = Field(gt=0, le=120)
    duration_ms: float = Field(gt=0, le=3600000)
    processed: int = Field(ge=0)
    fresh_frames: int = Field(ge=0)
    p50_ms: float = Field(ge=0, le=60000)
    p95_ms: float = Field(ge=0, le=60000)
    achieved_fps: float = Field(ge=0, le=1000)
    missed_deadlines: int = Field(ge=0)
    sustainable: bool
    attempts: int | None = Field(default=None, ge=0)
    rate_limited: bool = False
    confirmation: bool = False


class BenchmarkReport(StrictModel):
    id: UUID
    device_id: UUID
    model_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    wall_ms: int = Field(ge=946684800000, le=4102444800000)
    backend: Literal["litert_gpu_backbone_cpu_head", "litert_opencl_full_encoder"]
    preprocessing_backend: Literal["opencv_cpu", "opencl"] = "opencv_cpu"
    capture_source: Literal["media_projection", "accessibility"] = "media_projection"
    gpu_parity_cosine: float = Field(ge=-1, le=1.001)
    selected_fps: float = Field(gt=0, le=120)
    thermal_status: int = Field(ge=0, le=6)
    results: list[CandidateResult] = Field(min_length=1, max_length=28)
