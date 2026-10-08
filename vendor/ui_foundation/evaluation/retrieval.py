from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(slots=True)
class RetrievalResult:
    query_count: int
    positive_query_count: int
    recalls: dict[int, float]
    mean_average_precision: float
    mean_reciprocal_rank: float
    macro_recalls: dict[int, float]
    macro_mean_average_precision: float
    macro_mean_reciprocal_rank: float
    evaluated_packages: int

    def to_dict(self) -> dict:
        return {
            "query_count": self.query_count,
            "positive_query_count": self.positive_query_count,
            "evaluated_packages": self.evaluated_packages,
            "recall": {str(key): value for key, value in self.recalls.items()},
            "mAP": self.mean_average_precision,
            "MRR": self.mean_reciprocal_rank,
            "macro_recall": {str(key): value for key, value in self.macro_recalls.items()},
            "macro_mAP": self.macro_mean_average_precision,
            "macro_MRR": self.macro_mean_reciprocal_rank,
        }


def evaluate_within_package(
    embeddings: np.ndarray,
    packages: Sequence[str],
    labels: Sequence[str],
    trace_ids: Sequence[str],
    sequence_indices: Sequence[int],
    recalls: Iterable[int] = (1, 5, 10),
    exclude_same_trace_neighbors: int = 1,
    query_chunk_size: int = 512,
    trace_visits: Sequence[Sequence[Sequence[int]]] | None = None,
    record_ids: Sequence[str] | None = None,
) -> RetrievalResult:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [records, dimensions]")
    metadata = [packages, labels, trace_ids, sequence_indices]
    if trace_visits is not None:
        metadata.append(trace_visits)
    if record_ids is not None:
        metadata.append(record_ids)
    if any(len(values) != embeddings.shape[0] for values in metadata):
        raise ValueError("Metadata and embedding lengths do not match")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    normalized = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized = normalized / np.maximum(norms, 1.0e-12)
    requested_recalls = sorted(set(int(value) for value in recalls if int(value) > 0))
    total_hits = {value: 0 for value in requested_recalls}
    all_average_precisions: list[float] = []
    all_reciprocal_ranks: list[float] = []
    package_summaries: list[tuple[dict[int, float], float, float]] = []
    query_count = 0
    positive_queries = 0

    grouped: dict[str, list[int]] = {}
    for index, package in enumerate(packages):
        grouped.setdefault(package, []).append(index)

    for indices in grouped.values():
        if len(indices) < 2:
            continue
        local = normalized[np.asarray(indices)]
        local_labels = [labels[index] for index in indices]
        local_traces = [trace_ids[index] for index in indices]
        local_sequences = [int(sequence_indices[index]) for index in indices]
        observations = [
            tuple(tuple(visit) for visit in trace_visits[index])
            if trace_visits is not None and len(trace_visits[index]) > 0
            else ((0, local_sequences[local_index]),) if local_sequences[local_index] >= 0 else ()
            for local_index, index in enumerate(indices)
        ]
        # Native graph records carry all real visits. Index them once per app;
        # filename numbers and graph-node ordering are not temporal evidence.
        positions: dict[tuple[str, int, int], list[int]] = {}
        image_records: dict[str, list[int]] = {}
        for local_index, visits in enumerate(observations):
            if local_traces[local_index]:
                for segment, position in visits:
                    positions.setdefault((local_traces[local_index], segment, position), []).append(local_index)
            if record_ids is not None and record_ids[indices[local_index]]:
                image_records.setdefault(record_ids[indices[local_index]], []).append(local_index)
        package_hits = {value: 0 for value in requested_recalls}
        package_aps: list[float] = []
        package_rrs: list[float] = []
        package_positive_queries = 0
        for start in range(0, len(indices), query_chunk_size):
            end = min(start + query_chunk_size, len(indices))
            similarities = local[start:end] @ local.T
            for row, local_query in enumerate(range(start, end)):
                query_count += 1
                label = local_labels[local_query]
                if label == "":
                    continue
                valid = np.ones(len(indices), dtype=bool)
                valid[local_query] = False
                if record_ids is not None:
                    for duplicate in image_records.get(record_ids[indices[local_query]], []):
                        valid[duplicate] = False
                if exclude_same_trace_neighbors >= 0 and local_traces[local_query] != "":
                    for segment, position in observations[local_query]:
                        for offset in range(-exclude_same_trace_neighbors, exclude_same_trace_neighbors + 1):
                            key = (local_traces[local_query], segment, position + offset)
                            for candidate in positions.get(key, []):
                                valid[candidate] = False
                positives = np.asarray(
                    [candidate_label == label and candidate_label != "" for candidate_label in local_labels],
                    dtype=bool,
                ) & valid
                if not bool(positives.any()):
                    continue
                positive_queries += 1
                package_positive_queries += 1
                candidate_indices = np.flatnonzero(valid)
                order = candidate_indices[np.argsort(-similarities[row, candidate_indices], kind="stable")]
                ranked_positive = positives[order]
                for recall in requested_recalls:
                    if bool(ranked_positive[:recall].any()):
                        total_hits[recall] += 1
                        package_hits[recall] += 1
                positive_ranks = np.flatnonzero(ranked_positive) + 1
                rr = 1.0 / float(positive_ranks[0])
                cumulative = np.cumsum(ranked_positive)
                precision = cumulative[ranked_positive] / positive_ranks
                ap = float(np.mean(precision))
                all_reciprocal_ranks.append(rr)
                all_average_precisions.append(ap)
                package_rrs.append(rr)
                package_aps.append(ap)
        if package_positive_queries > 0:
            package_summaries.append(
                (
                    {key: value / package_positive_queries for key, value in package_hits.items()},
                    float(np.mean(package_aps)),
                    float(np.mean(package_rrs)),
                )
            )

    denominator = max(positive_queries, 1)
    macro_recalls = {
        key: float(np.mean([summary[0][key] for summary in package_summaries]))
        if package_summaries
        else 0.0
        for key in requested_recalls
    }
    return RetrievalResult(
        query_count=query_count,
        positive_query_count=positive_queries,
        recalls={key: value / denominator for key, value in total_hits.items()},
        mean_average_precision=float(np.mean(all_average_precisions)) if all_average_precisions else 0.0,
        mean_reciprocal_rank=float(np.mean(all_reciprocal_ranks)) if all_reciprocal_ranks else 0.0,
        macro_recalls=macro_recalls,
        macro_mean_average_precision=float(np.mean([item[1] for item in package_summaries])) if package_summaries else 0.0,
        macro_mean_reciprocal_rank=float(np.mean([item[2] for item in package_summaries])) if package_summaries else 0.0,
        evaluated_packages=len(package_summaries),
    )
