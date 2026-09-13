from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data import Record


@dataclass(frozen=True)
class RankedCandidate:
    record: Record
    score: float


def rank_candidates(
    query: Record,
    query_feature: np.ndarray,
    gallery: Sequence[Record],
    gallery_features: np.ndarray,
) -> list[RankedCandidate]:
    eligible = [index for index, candidate in enumerate(gallery) if candidate.camera_id != query.camera_id]
    if not eligible:
        return []
    scores = gallery_features[eligible] @ query_feature
    order = np.argsort(-scores, kind="stable")
    return [RankedCandidate(gallery[eligible[index]], float(scores[index])) for index in order]


def average_precision(matches: Sequence[bool]) -> float:
    relevant = sum(matches)
    if relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, matched in enumerate(matches, start=1):
        if matched:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / relevant


def evaluate(
    queries: Sequence[Record],
    query_features: np.ndarray,
    gallery: Sequence[Record],
    gallery_features: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    recall1: list[float] = []
    recall5: list[float] = []
    aps: list[float] = []
    rankings: list[dict[str, object]] = []
    skipped = 0
    for query, query_feature in zip(queries, query_features):
        ranked = rank_candidates(query, query_feature, gallery, gallery_features)
        matches = [candidate.record.vehicle_id == query.vehicle_id for candidate in ranked]
        if not any(matches):
            skipped += 1
            continue
        recall1.append(float(any(matches[:1])))
        recall5.append(float(any(matches[:5])))
        aps.append(average_precision(matches))
        rankings.append(
            {
                "query": str(query.image_path),
                "vehicle_id": query.vehicle_id,
                "top5": [
                    {
                        "image_path": str(candidate.record.image_path),
                        "vehicle_id": candidate.record.vehicle_id,
                        "camera_id": candidate.record.camera_id,
                        "score": round(candidate.score, 6),
                        "correct": candidate.record.vehicle_id == query.vehicle_id,
                    }
                    for candidate in ranked[:5]
                ],
            }
        )
    evaluated = len(recall1)
    metrics = {
        "protocol": "cross-camera query-to-gallery",
        "evaluated_queries": evaluated,
        "skipped_queries_without_positive": skipped,
        "recall_at_1": float(np.mean(recall1)) if evaluated else None,
        "recall_at_5": float(np.mean(recall5)) if evaluated else None,
        "mAP": float(np.mean(aps)) if evaluated else None,
    }
    return metrics, rankings
