from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from ui_foundation.config import ExperimentConfig
from ui_foundation.data.loaders import DataLoaderFactory
from ui_foundation.evaluation.retrieval import evaluate_within_package
from ui_foundation.models.encoder import build_encoder
from ui_foundation.training.application import prepare_data
from ui_foundation.utils.logging import configure_logging
from ui_foundation.utils.runtime import (
    DistributedContext,
    autocast_context,
    move_to_device,
    resolve_amp_dtype,
    seed_everything,
)

LOGGER = logging.getLogger(__name__)


class EvaluationApplication:
    def __init__(self, config: ExperimentConfig, context: DistributedContext) -> None:
        self.config = config
        self.context = context

    def run(self) -> dict[str, Any]:
        output = Path(self.config.runtime.output_dir)
        if self.context.is_main:
            output.mkdir(parents=True, exist_ok=True)
        self.context.barrier()
        configure_logging(output, rank=self.context.rank)
        if self.context.is_main:
            (output / "evaluation_config.json").write_text(
                json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
            )
        seed_everything(self.config.runtime.seed, self.config.runtime.deterministic, self.context.rank)
        manifest_path = prepare_data(self.config, self.context)
        loader = DataLoaderFactory(self.config, self.context).build_evaluation(str(manifest_path))
        encoder = build_encoder(self.config).to(self.context.device).eval()
        amp_dtype = resolve_amp_dtype(self.config.runtime.amp, self.context.device)

        local: dict[str, list[Any]] = {
            "embeddings": [],
            "record_id": [],
            "package_name": [],
            "activity_name": [],
            "structure_id": [],
            "state_id": [],
            "trace_id": [],
            "sequence_index": [],
            "trace_visits": [],
        }
        progress = tqdm(
            loader,
            desc=f"evaluate {self.config.name}",
            unit="batch",
            disable=not (self.context.is_main and self.config.runtime.progress_bars),
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for batch in progress:
                device_batch = move_to_device(batch, self.context.device)
                with autocast_context(self.context.device, amp_dtype):
                    embeddings = encoder(device_batch["images"])
                local["embeddings"].append(embeddings.detach().float().cpu().numpy())
                for key in (
                    "record_id",
                    "package_name",
                    "activity_name",
                    "structure_id",
                    "state_id",
                    "trace_id",
                    "trace_visits",
                ):
                    local[key].extend(batch[key])
                local["sequence_index"].extend(batch["sequence_index"].tolist())

        packed = {
            **{key: value for key, value in local.items() if key != "embeddings"},
            "embeddings": np.concatenate(local["embeddings"], axis=0) if local["embeddings"] else np.empty((0, 0)),
        }
        gathered = self._gather(packed)
        if not self.context.is_main:
            return {}
        merged = merge_payloads(gathered)
        label_key = self.config.evaluation.label
        labels = merged.get(label_key)
        if labels is None:
            raise ValueError(f"Evaluation label {label_key!r} is not available")
        result = evaluate_within_package(
            merged["embeddings"],
            merged["package_name"],
            labels,
            merged["trace_id"],
            merged["sequence_index"],
            recalls=self.config.evaluation.recalls,
            exclude_same_trace_neighbors=self.config.evaluation.exclude_same_trace_neighbors,
            trace_visits=merged["trace_visits"],
            record_ids=merged["record_id"],
        )
        report = {
            "experiment": self.config.experiment,
            "name": self.config.name,
            "checkpoint": self.config.model.checkpoint,
            "representation": (
                "raw_cls" if self.config.model.checkpoint == ""
                else self.config.evaluation.representation
            ),
            "split": self.config.evaluation.split,
            "label": label_key,
            "manifest": str(manifest_path),
            "sample_unit": "full_screenshot",
            "temporal_exclusion": "any_visit_same_continuous_segment",
            "exclude_same_trace_neighbors": self.config.evaluation.exclude_same_trace_neighbors,
            "records_with_trace_visits": sum(bool(visits) for visits in merged["trace_visits"]),
            "records_without_temporal_metadata": sum(
                not visits and index < 0
                for visits, index in zip(merged["trace_visits"], merged["sequence_index"])
            ),
            "records": int(merged["embeddings"].shape[0]),
            "dimensions": int(merged["embeddings"].shape[1]) if merged["embeddings"].ndim == 2 else 0,
            "within_package": result.to_dict(),
        }
        report_path = output / "retrieval_metrics.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if self.config.evaluation.save_embeddings:
            embedding_path = (
                Path(self.config.evaluation.embedding_output)
                if self.config.evaluation.embedding_output != ""
                else output / "embeddings.npz"
            )
            embedding_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                embedding_path,
                embeddings=merged["embeddings"],
                record_id=np.asarray(merged["record_id"], dtype=object),
                package_name=np.asarray(merged["package_name"], dtype=object),
                activity_name=np.asarray(merged["activity_name"], dtype=object),
                structure_id=np.asarray(merged["structure_id"], dtype=object),
                state_id=np.asarray(merged["state_id"], dtype=object),
                trace_id=np.asarray(merged["trace_id"], dtype=object),
                sequence_index=np.asarray(merged["sequence_index"], dtype=np.int64),
                trace_visits_json=np.asarray([json.dumps(visits) for visits in merged["trace_visits"]]),
            )
        LOGGER.info("Retrieval report written to %s", report_path)
        LOGGER.info("Metrics: %s", json.dumps(report["within_package"], sort_keys=True))
        return report

    def _gather(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.context.distributed:
            return [payload]
        result: list[dict[str, Any] | None] = [None] * self.context.world_size if self.context.is_main else []
        dist.gather_object(payload, result if self.context.is_main else None, dst=0)
        return [item for item in result if item is not None]


def merge_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if len(payloads) == 0:
        raise RuntimeError("No evaluation payloads were gathered")
    result: dict[str, Any] = {}
    for key in payloads[0]:
        if key == "embeddings":
            arrays = [payload[key] for payload in payloads if payload[key].size > 0]
            result[key] = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 0))
        else:
            result[key] = [item for payload in payloads for item in payload[key]]
    return result
