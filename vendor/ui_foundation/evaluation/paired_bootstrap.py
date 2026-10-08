from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

COHORT_FIELDS = (
    "record_id",
    "package_name",
    "activity_name",
    "structure_id",
    "state_id",
    "trace_id",
    "sequence_index",
    "trace_visits_json",
)
DEFAULT_RECALLS = (1, 5, 10)
DEFAULT_SENSITIVITIES = {
    "no_temporal_exclusion": -1,
    "same_visit_only": 0,
    "canonical_adjacent": 1,
    "wider_window_2": 2,
}
BASE_METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "map", "mrr")


@dataclass(slots=True)
class EmbeddingArchive:
    path: Path
    embeddings: np.ndarray
    metadata: dict[str, np.ndarray]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(workdir: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workdir, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=workdir, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "working_tree_clean": not status, "status": status}


def load_embedding_archive(path: str | Path) -> EmbeddingArchive:
    resolved = Path(path).resolve()
    # These object arrays are project-generated string metadata, not third-party inputs.
    with np.load(resolved, allow_pickle=True) as payload:
        missing = {"embeddings", *COHORT_FIELDS}.difference(payload.files)
        if missing:
            raise ValueError(f"{resolved} is missing fields: {sorted(missing)}")
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        metadata = {field: np.asarray(payload[field]) for field in COHORT_FIELDS}
    if embeddings.ndim != 2:
        raise ValueError(f"{resolved}: embeddings must have shape [records, dimensions]")
    if any(len(values) != len(embeddings) for values in metadata.values()):
        raise ValueError(f"{resolved}: metadata and embedding lengths do not match")
    return EmbeddingArchive(resolved, embeddings, metadata)


def verify_matched_cohort(first: EmbeddingArchive, second: EmbeddingArchive) -> dict[str, Any]:
    if first.embeddings.shape[0] != second.embeddings.shape[0]:
        raise ValueError("Embedding archives contain different record counts")
    equality = {
        field: bool(np.array_equal(first.metadata[field], second.metadata[field]))
        for field in COHORT_FIELDS
    }
    unequal = [field for field, matches in equality.items() if not matches]
    if unequal:
        raise ValueError(f"Embedding cohorts differ in fields: {unequal}")
    ordered_ids = "".join(f"{value}\n" for value in first.metadata["record_id"])
    return {
        "records": int(first.embeddings.shape[0]),
        "fields_equal": equality,
        "ordered_record_id_sha256": hashlib.sha256(ordered_ids.encode("utf-8")).hexdigest(),
    }


def _normalized(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float32, copy=False)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.maximum(norms, 1.0e-12)


def _empty_sums(recalls: Sequence[int]) -> dict[str, float]:
    return {
        **{f"recall_at_{recall}": 0.0 for recall in recalls},
        "map": 0.0,
        "mrr": 0.0,
    }


def _add_ranking_scores(
    sums: dict[str, float],
    similarities: np.ndarray,
    valid: np.ndarray,
    positives: np.ndarray,
    recalls: Sequence[int],
) -> None:
    candidate_indices = np.flatnonzero(valid)
    order = candidate_indices[
        np.argsort(-similarities[candidate_indices], kind="stable")
    ]
    ranked_positive = positives[order]
    for recall in recalls:
        sums[f"recall_at_{recall}"] += float(bool(ranked_positive[:recall].any()))
    positive_ranks = np.flatnonzero(ranked_positive) + 1
    sums["mrr"] += 1.0 / float(positive_ranks[0])
    cumulative = np.cumsum(ranked_positive)
    precision = cumulative[ranked_positive] / positive_ranks
    sums["map"] += float(np.mean(precision))


def score_package_pair(payload: dict[str, Any]) -> list[dict[str, Any]]:
    package = str(payload["package_name"])
    first = _normalized(payload["first_embeddings"])
    second = _normalized(payload["second_embeddings"])
    labels = [str(value) for value in payload["labels"]]
    trace_ids = [str(value) for value in payload["trace_ids"]]
    record_ids = [str(value) for value in payload["record_ids"]]
    sequence_indices = [int(value) for value in payload["sequence_indices"]]
    visits = [
        tuple(observations)
        if observations
        else ((0, sequence_indices[index]),)
        if sequence_indices[index] >= 0
        else ()
        for index, observations in enumerate(payload["trace_visits"])
    ]
    recalls = tuple(int(value) for value in payload["recalls"])
    sensitivities = dict(payload["sensitivities"])
    query_chunk_size = int(payload.get("query_chunk_size", 512))
    size = len(labels)
    if size < 2:
        return []

    # Fill in the evaluator's exact query chunks. A single full matrix multiply can
    # change float32 tie ordering enough to move a handful of retrieval ranks.
    first_similarities = np.empty((size, size), dtype=np.float32)
    second_similarities = np.empty((size, size), dtype=np.float32)
    for start in range(0, size, query_chunk_size):
        end = min(start + query_chunk_size, size)
        first_similarities[start:end] = first[start:end] @ first.T
        second_similarities[start:end] = second[start:end] @ second.T

    positions: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    image_records: dict[str, list[int]] = defaultdict(list)
    for local_index, observations in enumerate(visits):
        if trace_ids[local_index]:
            for segment, position in observations:
                positions[(trace_ids[local_index], int(segment), int(position))].append(local_index)
        if record_ids[local_index]:
            image_records[record_ids[local_index]].append(local_index)

    results: list[dict[str, Any]] = []
    for sensitivity, radius in sensitivities.items():
        first_sums = _empty_sums(recalls)
        second_sums = _empty_sums(recalls)
        positive_queries = 0
        for query in range(size):
            label = labels[query]
            if label == "":
                continue
            valid = np.ones(size, dtype=bool)
            valid[query] = False
            for duplicate in image_records.get(record_ids[query], []):
                valid[duplicate] = False
            if radius >= 0 and trace_ids[query]:
                for segment, position in visits[query]:
                    for offset in range(-radius, radius + 1):
                        for candidate in positions.get(
                            (trace_ids[query], int(segment), int(position) + offset), []
                        ):
                            valid[candidate] = False
            positives = np.asarray(
                [candidate == label and candidate != "" for candidate in labels], dtype=bool
            ) & valid
            if not bool(positives.any()):
                continue
            positive_queries += 1
            _add_ranking_scores(first_sums, first_similarities[query], valid, positives, recalls)
            _add_ranking_scores(second_sums, second_similarities[query], valid, positives, recalls)
        results.append(
            {
                "sensitivity": sensitivity,
                "temporal_radius": int(radius),
                "package_name": package,
                "record_count": size,
                "query_count": size,
                "positive_query_count": positive_queries,
                "first": {
                    key: value / positive_queries if positive_queries else 0.0
                    for key, value in first_sums.items()
                },
                "second": {
                    key: value / positive_queries if positive_queries else 0.0
                    for key, value in second_sums.items()
                },
            }
        )
    return results


def build_package_payloads(
    first: EmbeddingArchive,
    second: EmbeddingArchive,
    *,
    label_field: str,
    recalls: Sequence[int],
    sensitivities: dict[str, int],
) -> list[dict[str, Any]]:
    metadata = first.metadata
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, package in enumerate(metadata["package_name"]):
        grouped[str(package)].append(index)
    decoded_visits = [
        tuple(tuple(int(item) for item in visit) for visit in json.loads(str(raw)))
        if str(raw) else ()
        for raw in metadata["trace_visits_json"]
    ]
    payloads: list[dict[str, Any]] = []
    for package in sorted(grouped):
        indices = np.asarray(grouped[package], dtype=np.int64)
        payloads.append(
            {
                "package_name": package,
                "first_embeddings": first.embeddings[indices],
                "second_embeddings": second.embeddings[indices],
                "labels": metadata[label_field][indices].tolist(),
                "trace_ids": metadata["trace_id"][indices].tolist(),
                "record_ids": metadata["record_id"][indices].tolist(),
                "sequence_indices": metadata["sequence_index"][indices].tolist(),
                "trace_visits": [decoded_visits[index] for index in indices],
                "recalls": tuple(recalls),
                "sensitivities": sensitivities,
                "query_chunk_size": 512,
            }
        )
    return payloads


def score_all_packages(
    payloads: Sequence[dict[str, Any]], workers: int, progress: bool = True
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    rows: list[dict[str, Any]] = []
    if workers == 1:
        iterator: Iterable[dict[str, Any]] = payloads
        for payload in tqdm(iterator, desc="score packages", disable=not progress):
            rows.extend(score_package_pair(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(score_package_pair, payload) for payload in payloads]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="score packages", disable=not progress
            ):
                rows.extend(future.result())
    return sorted(rows, key=lambda row: (row["sensitivity"], row["package_name"]))


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No packages contain eligible retrieval queries")
    eligible = [row for row in rows if row["positive_query_count"] > 0]
    if not eligible:
        raise ValueError("No packages contain eligible retrieval queries")
    weights = np.asarray([row["positive_query_count"] for row in eligible], dtype=np.float64)
    result: dict[str, Any] = {
        "packages_with_at_least_two_records": len(rows),
        "evaluated_packages": len(eligible),
        "positive_query_count": int(weights.sum()),
        "query_count": int(sum(row["query_count"] for row in rows)),
        "first": {},
        "second": {},
        "delta_first_minus_second_percentage_points": {},
        "package_effect_counts": {},
        "package_effect_concentration": {},
    }
    for metric in BASE_METRICS:
        first_values = np.asarray([row["first"][metric] for row in eligible], dtype=np.float64)
        second_values = np.asarray([row["second"][metric] for row in eligible], dtype=np.float64)
        delta = first_values - second_values
        for model, values in (("first", first_values), ("second", second_values)):
            result[model][f"macro_{metric}"] = float(values.mean())
            result[model][f"query_{metric}"] = float(np.average(values, weights=weights))
        result["delta_first_minus_second_percentage_points"][f"macro_{metric}"] = float(
            delta.mean() * 100.0
        )
        result["delta_first_minus_second_percentage_points"][f"query_{metric}"] = float(
            np.average(delta, weights=weights) * 100.0
        )
        result["package_effect_counts"][metric] = {
            "first_better": int(np.sum(delta > 1.0e-12)),
            "tie": int(np.sum(np.abs(delta) <= 1.0e-12)),
            "second_better": int(np.sum(delta < -1.0e-12)),
        }
        weighted_contribution = weights * delta
        absolute_contribution = np.abs(weighted_contribution)
        order = np.argsort(-absolute_contribution)
        absolute_total = float(absolute_contribution.sum())
        query_total = float(weights.sum())
        result["package_effect_concentration"][metric] = {
            "first_better_query_share": float(weights[delta > 1.0e-12].sum() / query_total),
            "tie_query_share": float(weights[np.abs(delta) <= 1.0e-12].sum() / query_total),
            "second_better_query_share": float(weights[delta < -1.0e-12].sum() / query_total),
            "absolute_query_weighted_contribution_share_top_1": float(
                absolute_contribution[order[:1]].sum() / absolute_total
            ) if absolute_total else 0.0,
            "absolute_query_weighted_contribution_share_top_5": float(
                absolute_contribution[order[:5]].sum() / absolute_total
            ) if absolute_total else 0.0,
            "absolute_query_weighted_contribution_share_top_10": float(
                absolute_contribution[order[:10]].sum() / absolute_total
            ) if absolute_total else 0.0,
            "largest_absolute_contributors": [
                {
                    "package_name": eligible[index]["package_name"],
                    "positive_query_count": int(weights[index]),
                    "delta_percentage_points": float(delta[index] * 100.0),
                    "absolute_contribution_share": float(
                        absolute_contribution[index] / absolute_total
                    ) if absolute_total else 0.0,
                }
                for index in order[:10]
            ],
        }
    return result


def paired_package_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
    chunk_size: int = 1000,
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    eligible = [row for row in rows if row["positive_query_count"] > 0]
    rng = np.random.default_rng(seed)
    package_count = len(eligible)
    weights = np.asarray([row["positive_query_count"] for row in eligible], dtype=np.float64)
    distributions: dict[str, np.ndarray] = {
        name: np.empty(replicates, dtype=np.float64)
        for metric in BASE_METRICS
        for name in (f"macro_{metric}", f"query_{metric}")
    }
    deltas = {
        metric: np.asarray(
            [row["first"][metric] - row["second"][metric] for row in eligible],
            dtype=np.float64,
        )
        for metric in BASE_METRICS
    }
    start = 0
    while start < replicates:
        end = min(start + chunk_size, replicates)
        indices = rng.integers(0, package_count, size=(end - start, package_count))
        sampled_weights = weights[indices]
        denominators = sampled_weights.sum(axis=1)
        for metric, values in deltas.items():
            sampled = values[indices]
            distributions[f"macro_{metric}"][start:end] = sampled.mean(axis=1)
            distributions[f"query_{metric}"][start:end] = (
                (sampled * sampled_weights).sum(axis=1) / denominators
            )
        start = end

    alpha = (1.0 - confidence) / 2.0
    result: dict[str, Any] = {}
    aggregate = aggregate_rows(rows)
    for name, distribution in distributions.items():
        point = aggregate["delta_first_minus_second_percentage_points"][name]
        result[name] = {
            "point_delta_percentage_points": point,
            "confidence_interval_percentage_points": [
                float(np.quantile(distribution, alpha) * 100.0),
                float(np.quantile(distribution, 1.0 - alpha) * 100.0),
            ],
            "bootstrap_standard_error_percentage_points": float(distribution.std(ddof=1) * 100.0),
            "bootstrap_probability_delta_positive": float(np.mean(distribution > 0.0)),
            "bootstrap_probability_delta_nonpositive": float(np.mean(distribution <= 0.0)),
        }
    return result


def _saved_report_metrics(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))["within_package"]
    return {
        **{f"query_recall_at_{key}": float(value) for key, value in report["recall"].items()},
        "query_map": float(report["mAP"]),
        "query_mrr": float(report["MRR"]),
        **{f"macro_recall_at_{key}": float(value) for key, value in report["macro_recall"].items()},
        "macro_map": float(report["macro_mAP"]),
        "macro_mrr": float(report["macro_MRR"]),
    }


def validate_primary_against_reports(
    primary: dict[str, Any], first_report: Path, second_report: Path, tolerance: float = 1.0e-9
) -> dict[str, Any]:
    result: dict[str, Any] = {"tolerance": tolerance, "first": {}, "second": {}}
    maximum = 0.0
    for model, path in (("first", first_report), ("second", second_report)):
        expected = _saved_report_metrics(path)
        differences = {
            metric: abs(primary[model][metric] - value) for metric, value in expected.items()
        }
        result[model] = {
            "report": str(path),
            "maximum_absolute_difference": max(differences.values()),
            "differences": differences,
        }
        maximum = max(maximum, max(differences.values()))
    result["maximum_absolute_difference"] = maximum
    result["passed"] = maximum <= tolerance
    if not result["passed"]:
        raise ValueError(
            f"Recomputed primary metrics differ from saved reports by {maximum}, above {tolerance}"
        )
    return result


def write_per_package_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "sensitivity",
        "temporal_radius",
        "package_name",
        "record_count",
        "query_count",
        "positive_query_count",
    ]
    for metric in BASE_METRICS:
        fields.extend((f"first_{metric}", f"second_{metric}", f"delta_{metric}_percentage_points"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if row["positive_query_count"] == 0:
                continue
            flat = {key: row[key] for key in fields[:6]}
            for metric in BASE_METRICS:
                flat[f"first_{metric}"] = row["first"][metric]
                flat[f"second_{metric}"] = row["second"][metric]
                flat[f"delta_{metric}_percentage_points"] = (
                    row["first"][metric] - row["second"][metric]
                ) * 100.0
            writer.writerow(flat)


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    primary = report["sensitivities"]["canonical_adjacent"]
    first_name = report["models"]["first"]
    second_name = report["models"]["second"]
    lines = [
        f"# Paired {first_name} versus {second_name} package-bootstrap analysis",
        "",
        f"Generated `{report['generated_at_utc']}` from the exact matched {report['cohort']['records']:,}-record cohort.",
        f"{first_name} is the first model and {second_name} is the second; positive deltas favor {first_name}.",
        "",
        "## Primary result: canonical temporal exclusion",
        "",
        f"| Metric | {first_name} | {second_name} | Delta (pp) | Paired 95% package-bootstrap CI (pp) | P(delta > 0) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    display = (
        "macro_recall_at_1",
        "macro_recall_at_5",
        "macro_recall_at_10",
        "macro_map",
        "macro_mrr",
        "query_recall_at_1",
        "query_recall_at_5",
        "query_recall_at_10",
        "query_map",
        "query_mrr",
    )
    for metric in display:
        boot = primary["bootstrap"][metric]
        low, high = boot["confidence_interval_percentage_points"]
        lines.append(
            f"| {metric} | {primary['aggregate']['first'][metric]:.6f} | "
            f"{primary['aggregate']['second'][metric]:.6f} | "
            f"{boot['point_delta_percentage_points']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
            f"{boot['bootstrap_probability_delta_positive']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Chronology sensitivity",
            "",
            "| Exclusion | Packages | Queries | Macro R@1 delta | Query R@1 delta | Macro mAP delta | Query mAP delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in report["sensitivity_order"]:
        item = report["sensitivities"][name]
        delta = item["aggregate"]["delta_first_minus_second_percentage_points"]
        lines.append(
            f"| {name} | {item['aggregate']['evaluated_packages']} | "
            f"{item['aggregate']['positive_query_count']} | {delta['macro_recall_at_1']:+.4f} | "
            f"{delta['query_recall_at_1']:+.4f} | {delta['macro_map']:+.4f} | "
            f"{delta['query_map']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Canonical per-package distribution",
            "",
            f"| Metric | Packages {first_name} / tie / {second_name} | Query share {first_name} / tie / {second_name} | Top-1 / top-5 absolute contribution share |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in BASE_METRICS:
        counts = primary["aggregate"]["package_effect_counts"][metric]
        concentration = primary["aggregate"]["package_effect_concentration"][metric]
        lines.append(
            f"| {metric} | {counts['first_better']} / {counts['tie']} / {counts['second_better']} | "
            f"{concentration['first_better_query_share']:.3f} / {concentration['tie_query_share']:.3f} / "
            f"{concentration['second_better_query_share']:.3f} | "
            f"{concentration['absolute_query_weighted_contribution_share_top_1']:.3f} / "
            f"{concentration['absolute_query_weighted_contribution_share_top_5']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The percentile interval resamples packages as paired clusters and measures uncertainty for this fixed evaluation cohort. It does not measure training-seed variance, repair test-set reuse, or establish the effect on a separately trained run.",
            "The per-package CSV is the audit trail for package wins, ties, losses, and concentration of effects.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    *,
    first_path: Path,
    second_path: Path,
    output_dir: Path,
    first_name: str,
    second_name: str,
    label_field: str,
    workers: int,
    replicates: int,
    seed: int,
    sensitivities: dict[str, int] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    first = load_embedding_archive(first_path)
    second = load_embedding_archive(second_path)
    cohort = verify_matched_cohort(first, second)
    selected_sensitivities = sensitivities or DEFAULT_SENSITIVITIES
    payloads = build_package_payloads(
        first,
        second,
        label_field=label_field,
        recalls=DEFAULT_RECALLS,
        sensitivities=selected_sensitivities,
    )
    rows = score_all_packages(payloads, workers=workers)
    by_sensitivity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sensitivity[row["sensitivity"]].append(row)
    sensitivity_results: dict[str, Any] = {}
    for offset, name in enumerate(selected_sensitivities):
        selected = by_sensitivity[name]
        sensitivity_results[name] = {
            "temporal_radius": selected_sensitivities[name],
            "aggregate": aggregate_rows(selected),
            "bootstrap": paired_package_bootstrap(
                selected, replicates=replicates, seed=seed + offset
            ),
        }

    primary = sensitivity_results["canonical_adjacent"]["aggregate"]
    first_report = first.path.with_name("retrieval_metrics.json")
    second_report = second.path.with_name("retrieval_metrics.json")
    validation = validate_primary_against_reports(primary, first_report, second_report)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": git_provenance(Path.cwd()),
        "models": {"first": first_name, "second": second_name},
        "inputs": {
            "first_embeddings": str(first.path),
            "first_sha256": sha256_file(first.path),
            "second_embeddings": str(second.path),
            "second_sha256": sha256_file(second.path),
            "label_field": label_field,
        },
        "cohort": cohort,
        "method": {
            "unit": "package cluster",
            "paired": True,
            "bootstrap_replicates": replicates,
            "confidence": 0.95,
            "interval": "percentile",
            "seed_base": seed,
            "workers": workers,
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "recalls": list(DEFAULT_RECALLS),
        },
        "sensitivity_order": list(selected_sensitivities),
        "sensitivities": sensitivity_results,
        "primary_saved_report_validation": validation,
    }
    write_per_package_csv(output_dir / "per_package_effects.csv", rows)
    (output_dir / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(output_dir / "REPORT.md", report)
    return report
