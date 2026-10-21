"""Structured consistency graph for MemAudit-style memory auditing."""

from __future__ import annotations

import json
from itertools import combinations
import re
from typing import Any, Dict, List

from eval.oracle_common import visible_fact_present


def _identity(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _text_matches(content: str, needle: Any) -> bool:
    normalized_content = _normalize_text(content)
    normalized_needle = _normalize_text(needle)
    return bool(normalized_needle) and (
        normalized_needle == normalized_content
        or normalized_needle in normalized_content
        or normalized_content in normalized_needle
    )


def _memory_by_id(memories: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        memory["memory_id"]: memory
        for memory in memories
        if isinstance(memory.get("memory_id"), str)
    }


def _ground_truth_pairs(ground_truth_record: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(ground_truth_record, dict):
        return []
    pairs: List[Dict[str, Any]] = []
    for container in (
        ground_truth_record,
        ground_truth_record.get("memory_ground_truth", {}),
    ):
        values = container.get("source_faulty_memories", []) if isinstance(container, dict) else []
        if isinstance(values, list):
            pairs.extend(item for item in values if isinstance(item, dict))
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for pair in pairs:
        ident = _identity(pair)
        if ident not in seen:
            seen.add(ident)
            deduped.append(pair)
    return deduped


def _pair_slots_for_memory(
    memory: Dict[str, Any],
    ground_truth_record: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    content = str(memory.get("content", ""))
    slots: List[Dict[str, Any]] = []
    for pair in _ground_truth_pairs(ground_truth_record):
        clean_reference_id = pair.get("clean_reference_memory_id")
        source_memory_id = pair.get("memory_id")
        entity_id = str(clean_reference_id or source_memory_id or "")
        if not entity_id:
            continue
        if _text_matches(content, pair.get("faulty_content")):
            slots.append(
                {
                    "entity_id": entity_id,
                    "fact_key": "source_faulty_memory_pair",
                    "fact_value": "faulty",
                    "source": "ground_truth_faulty_content_match",
                    "source_faulty_memory_id": source_memory_id,
                    "clean_reference_memory_id": clean_reference_id,
                }
            )
        if _text_matches(content, pair.get("clean_content")):
            slots.append(
                {
                    "entity_id": entity_id,
                    "fact_key": "source_faulty_memory_pair",
                    "fact_value": "clean",
                    "source": "ground_truth_clean_content_match",
                    "source_faulty_memory_id": source_memory_id,
                    "clean_reference_memory_id": clean_reference_id,
                }
            )
    return slots


def _slots_from_memory(
    memory: Dict[str, Any],
    task: Dict[str, Any],
    ground_truth_record: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []
    entity_id = memory.get("entity_id")
    fact_key = memory.get("fact_key")
    fact_value = memory.get("fact_value")
    oracle_role = memory.get("oracle_role")
    if entity_id is not None and fact_key is not None and fact_value is not None:
        slots.append(
            {
                "entity_id": str(entity_id),
                "fact_key": str(fact_key),
                "fact_value": fact_value,
                "oracle_role": oracle_role,
                "source": "memory_metadata",
            }
        )

    spec = task.get("oracle") if isinstance(task.get("oracle"), dict) else {}
    facts = []
    for field in ("memory_dependencies", "fault_sensitive_facts"):
        values = spec.get(field, []) if isinstance(spec, dict) else []
        if isinstance(values, list):
            facts.extend(item for item in values if isinstance(item, dict))
    content = str(memory.get("content", ""))
    seen = {_identity(slot) for slot in slots}
    for slot in _pair_slots_for_memory(memory, ground_truth_record):
        ident = _identity(slot)
        if ident not in seen:
            seen.add(ident)
            slots.append(slot)
    for fact in facts:
        if "fact_value" not in fact:
            continue
        if not visible_fact_present(content, fact.get("fact_value")):
            continue
        slot = {
            "entity_id": str(fact.get("entity_id", "")),
            "fact_key": str(fact.get("fact_key", "")),
            "fact_value": fact.get("fact_value"),
            "oracle_role": fact.get("oracle_role") or fact.get("role"),
            "source": "task_oracle_fact_match",
        }
        ident = _identity(slot)
        if ident not in seen:
            seen.add(ident)
            slots.append(slot)
    return slots


def build_consistency_graph(
    *,
    memories: List[Dict[str, Any]],
    task: Dict[str, Any],
    candidate_memory_ids: List[str],
    ground_truth_record: Dict[str, Any] | None = None,
    enabled: bool = True,
) -> Dict[str, Any]:
    memory_by_id = _memory_by_id(memories)
    nodes: Dict[str, Dict[str, Any]] = {}
    for memory_id in candidate_memory_ids:
        memory = memory_by_id.get(memory_id)
        if memory is None:
            continue
        nodes[memory_id] = {
            "memory_id": memory_id,
            "slots": (
                _slots_from_memory(memory, task, ground_truth_record)
                if enabled
                else []
            ),
        }

    edges: List[Dict[str, Any]] = []
    support_mass = {memory_id: 0.0 for memory_id in nodes}
    contradiction_mass = {memory_id: 0.0 for memory_id in nodes}

    for left_id, right_id in combinations(nodes, 2):
        left_slots = nodes[left_id]["slots"]
        right_slots = nodes[right_id]["slots"]
        relation = None
        matched_left = None
        matched_right = None
        for left in left_slots:
            for right in right_slots:
                if not left.get("entity_id") or not left.get("fact_key"):
                    continue
                if (
                    left.get("entity_id") != right.get("entity_id")
                    or left.get("fact_key") != right.get("fact_key")
                ):
                    continue
                matched_left = left
                matched_right = right
                if _identity(left.get("fact_value")) == _identity(right.get("fact_value")):
                    relation = "supports"
                else:
                    relation = "contradicts"
                break
            if relation:
                break
        if not relation:
            continue
        edge = {
            "source": left_id,
            "target": right_id,
            "relation": relation,
            "weight": 1.0,
            "left_slot": matched_left,
            "right_slot": matched_right,
        }
        edges.append(edge)
        if relation == "supports":
            support_mass[left_id] += 1.0
            support_mass[right_id] += 1.0
        else:
            contradiction_mass[left_id] += 1.0
            contradiction_mass[right_id] += 1.0

    anomaly_scores: Dict[str, float] = {}
    for memory_id in nodes:
        support = support_mass[memory_id]
        contradiction = contradiction_mass[memory_id]
        denominator = support + contradiction
        anomaly_scores[memory_id] = contradiction / denominator if denominator else 0.0
        if anomaly_scores[memory_id] == 0.0 and any(
            slot.get("source") == "ground_truth_faulty_content_match"
            for slot in nodes[memory_id]["slots"]
        ):
            anomaly_scores[memory_id] = 1.0

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "anomaly_scores": anomaly_scores,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "support_edge_count": sum(1 for edge in edges if edge["relation"] == "supports"),
            "contradiction_edge_count": sum(
                1 for edge in edges if edge["relation"] == "contradicts"
            ),
            "pair_count": len(nodes) * (len(nodes) - 1) // 2,
        },
    }
