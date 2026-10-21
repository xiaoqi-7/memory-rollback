"""Scoring and selection for MemAudit-style memory auditing."""

from __future__ import annotations

from typing import Any, Dict, List


def score_memories(
    *,
    candidate_memory_ids: List[str],
    counterfactual_results: List[Dict[str, Any]],
    anomaly_scores: Dict[str, float],
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    influence_by_id = {
        row["memory_id"]: float(row.get("influence_score", 0.0))
        for row in counterfactual_results
    }
    scores: Dict[str, Dict[str, float]] = {}
    for memory_id in candidate_memory_ids:
        influence = influence_by_id.get(memory_id, 0.0)
        anomaly = float(anomaly_scores.get(memory_id, 0.0))
        detox = alpha * influence + (1.0 - alpha) * anomaly
        scores[memory_id] = {
            "influence_score": influence,
            "anomaly_score": anomaly,
            "detoxification_score": detox,
        }
    return scores


def select_delete_memory_ids(
    *,
    scores: Dict[str, Dict[str, float]],
    candidate_memory_ids: List[str],
    faulty_memory_ids: List[str],
    threshold: float,
    mode: str = "threshold",
) -> List[str]:
    selected = [
        memory_id
        for memory_id in candidate_memory_ids
        if scores.get(memory_id, {}).get("detoxification_score", 0.0) >= threshold
    ]
    if selected or mode == "threshold":
        return selected

    if mode != "threshold_or_top_k":
        return []

    top_k = max(1, len(faulty_memory_ids))
    ranked = sorted(
        candidate_memory_ids,
        key=lambda memory_id: (
            scores.get(memory_id, {}).get("detoxification_score", 0.0),
            scores.get(memory_id, {}).get("influence_score", 0.0),
            scores.get(memory_id, {}).get("anomaly_score", 0.0),
        ),
        reverse=True,
    )
    return [
        memory_id
        for memory_id in ranked[:top_k]
        if scores.get(memory_id, {}).get("detoxification_score", 0.0) > 0.0
    ]
