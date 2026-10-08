"""Synchronized, warmed inference measurement without dataset I/O or training."""
from __future__ import annotations

import os
import platform
import resource
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    device: str = "cpu"
    precision: str = "fp32"
    cpu_threads: int = 1
    batch_size: int = 1
    warmup: int = 3
    iterations: int = 10
    repeats: int = 3
    mode: str = "model_only"

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Benchmark device must be cpu or cuda")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("Benchmark precision must be fp32 or bf16")
        if self.device == "cpu" and (self.precision != "fp32" or self.mode != "model_only"):
            raise ValueError("CPU protocol is FP32 model_only; tensors already live on the host")
        if self.mode not in {"model_only", "transfer_inclusive"}:
            raise ValueError("Unknown benchmark timing mode")
        if min(self.cpu_threads, self.batch_size, self.iterations, self.repeats) < 1 or self.warmup < 0:
            raise ValueError("Threads, batch, iterations and repeats must be positive; warmup nonnegative")


def summarize_timings(samples_ms: list[list[float]], batch_size: int) -> dict[str, Any]:
    flat = np.asarray([value for repeat in samples_ms for value in repeat], dtype=np.float64)
    if flat.size == 0 or batch_size < 1 or not np.isfinite(flat).all() or (flat <= 0).any():
        raise ValueError("Timing samples must be finite, positive and nonempty")
    means = [statistics.mean(repeat) for repeat in samples_ms]
    return {
        "measured_calls": int(flat.size),
        "batch_latency_mean_ms": float(flat.mean()),
        "batch_latency_median_ms": float(np.median(flat)),
        "batch_latency_p95_ms": float(np.percentile(flat, 95)),
        "amortized_mean_ms_per_image": float(flat.mean() / batch_size),
        "synchronous_images_per_second": float(batch_size * 1000 / flat.mean()),
        "repeat_mean_batch_latency_ms": means,
        "repeat_mean_stddev_ms": statistics.stdev(means) if len(means) > 1 else 0.0,
        "raw_batch_latency_ms": samples_ms,
    }


def current_rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def hardware_metadata(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda, "cpu_logical_count": os.cpu_count(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            info["cpu_model"] = next(line.split(":", 1)[1].strip()
                                     for line in handle if line.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu_model"] = platform.processor()
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update({"gpu": props.name, "gpu_total_memory_bytes": props.total_memory,
                     "gpu_compute_capability": [props.major, props.minor]})
    return info


def measure_inference(
    model: nn.Module, cpu_images: torch.Tensor, settings: BenchmarkSettings,
) -> dict[str, Any]:
    settings.validate()
    if cpu_images.device.type != "cpu" or cpu_images.ndim != 4 or cpu_images.shape[0] < settings.batch_size:
        raise ValueError("Expected enough preprocessed NCHW CPU images")
    device = torch.device(settings.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU fallback is forbidden")
    if settings.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 requested but unsupported")
    if torch.get_num_threads() != settings.cpu_threads:
        raise ValueError("Set the declared CPU thread count before model construction")
    model.eval()
    host = cpu_images[:settings.batch_size].contiguous()
    if device.type == "cuda":
        host = host.pin_memory()
    images = host.to(device) if settings.mode == "model_only" else host
    amp_dtype = torch.bfloat16 if settings.precision == "bf16" else None

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward() -> torch.Tensor:
        selected = images.to(device, non_blocking=True) if settings.mode == "transfer_inclusive" else images
        value = model(selected)
        return value.float().cpu() if settings.mode == "transfer_inclusive" else value

    rss_before = current_rss_bytes()
    with torch.inference_mode(), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        checked = forward()
        synchronize()
        if checked.ndim != 2 or checked.shape[0] != settings.batch_size or not torch.isfinite(checked).all():
            raise RuntimeError("Encoder must return finite batch-by-dimension embeddings")
        norms = checked.float().norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=0.02, rtol=0.02):
            raise RuntimeError("Encoder embeddings are not unit normalized")
        output_dimension = int(checked.shape[1])
        del checked, norms
        for _ in range(settings.warmup):
            forward()
        synchronize()
        gpu_memory: dict[str, int] | None = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
        samples: list[list[float]] = []
        for _ in range(settings.repeats):
            repeat: list[float] = []
            for _ in range(settings.iterations):
                synchronize()
                start = time.perf_counter_ns()
                result = forward()
                synchronize()
                repeat.append((time.perf_counter_ns() - start) / 1e6)
                del result
            samples.append(repeat)
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            gpu_memory = {
                "baseline_allocated_bytes": baseline, "peak_allocated_bytes": peak,
                "peak_increment_over_baseline_bytes": peak - baseline,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
    return {
        "settings": asdict(settings), "input_shape": list(images.shape),
        "output_dimension": output_dimension,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "hardware": hardware_metadata(device), "timings": summarize_timings(samples, settings.batch_size),
        "cuda_allocator_memory": gpu_memory,
        "process_rss_before_measurement_bytes": rss_before,
        "process_rss_after_measurement_bytes": current_rss_bytes(),
        "worker_lifetime_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "memory_caveat": "CPU RSS includes Python, checkpoint construction and input cache; lifetime peak is cumulative across worker cases, not activation-only memory.",
    }


def compare_results(
    results: list[dict[str, Any]], student: str = "F6", references: tuple[str, ...] = ("F4", "F5"),
) -> list[dict[str, Any]]:
    profiles: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for result in results:
        settings = result["settings"]
        key = tuple(settings[name] for name in ("device", "precision", "cpu_threads", "batch_size", "mode"))
        models = profiles.setdefault(key, {})
        if result["model_name"] in models:
            raise ValueError("Duplicate model/profile benchmark result")
        models[result["model_name"]] = result
    comparisons = []
    for key, models in profiles.items():
        if any(name not in models for name in (*references, student)):
            raise ValueError("Every comparison profile must include the student and all references")
        if len({tuple(result["input_shape"]) for result in models.values()}) != 1:
            raise ValueError("Comparison inputs have different shapes")
        small = models[student]["timings"]
        for name, result in models.items():
            if name == student:
                continue
            large = result["timings"]
            comparisons.append({
                "profile": dict(zip(("device", "precision", "cpu_threads", "batch_size", "mode"), key, strict=True)),
                "reference": name, "student": student,
                "median_latency_speedup": large["batch_latency_median_ms"] / small["batch_latency_median_ms"],
                "mean_throughput_speedup": small["synchronous_images_per_second"] / large["synchronous_images_per_second"],
            })
    return comparisons
