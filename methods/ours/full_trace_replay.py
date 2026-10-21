"""Helpers for the ``Ours w/o selective replay`` ablation.

The default rollback optimizer decides a small selected replay set.  This
module keeps the full-trace expansion and memory-mutation ID rewriting in one
place so the optimizer/executor can stay focused on their normal contracts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

try:  # importable both as a package and from direct scripts
    from .graph_schema import DependencyGraph, is_memory_id
except ImportError:  # pragma: no cover - direct-script fallback
    from graph_schema import DependencyGraph, is_memory_id


FULL_TRACE_REPLAYABLE_TYPES = {
    "memory_read",
    "plan",
    "tool_action",
    "claim",
    "memory_write",
    "memory_delete",
    "memory_update",
    "memory_consolidate",
}

MEMORY_MUTATION_TYPES = {
    "memory_write",
    "memory_delete",
    "memory_update",
    "memory_consolidate",
}


REASON_COUNT_FIELDS = (
    ("delete", "delete_memory_ids"),
    ("quarantine", "quarantine_memory_ids"),
    ("derived_candidates", "derived_memory_candidate_ids"),
    ("derived_supported", "supported_derived_memory_ids"),
    ("derived_unsupported", "unsupported_derived_memory_ids"),
    ("invalidate_claims", "invalidate_claim_ids"),
    ("replay", "replay_step_ids"),
    ("preserve", "preserve_step_ids"),
    ("redundant", "redundant_step_ids"),
    ("suspicious", "suspicious_step_ids"),
)


def rebuild_reason_counts(
    plan: Dict[str, Any],
    extra_markers: Optional[List[str]] = None,
) -> str:
    """Return a reason string whose counts match the current plan lists."""
    parts = [
        f"{label}={len(plan.get(field, []))}"
        for label, field in REASON_COUNT_FIELDS
    ]

    existing_reason = str(plan.get("reason", "")).strip()
    existing_markers = [
        part.strip()
        for part in existing_reason.split(";")
        if part.strip() and "=" not in part
    ]
    for marker in existing_markers + list(extra_markers or []):
        if marker not in parts:
            parts.append(marker)
    return "; ".join(parts)


def _last_final_answer_ids(graph: DependencyGraph) -> set[str]:
    final_answers = [
        (node_id, node)
        for node_id, node in graph.nodes.items()
        if node.semantic_type == "final_answer"
    ]
    if not final_answers:
        return set()
    turns = [node.turn for _node_id, node in final_answers if node.turn is not None]
    if not turns:
        return {final_answers[-1][0]}
    last_turn = max(turns)
    return {
        node_id
        for node_id, node in final_answers
        if node.turn == last_turn
    }


def full_trace_replay_step_ids(graph: DependencyGraph, task_id: str) -> List[str]:
    """Return replayable step ids in original trace order for one task."""
    last_final_answers = _last_final_answer_ids(graph)
    replay_ids: List[str] = []
    for node_id, node in graph.nodes.items():
        if node.node_kind != "step":
            continue
        if node.raw.get("task_id") != task_id:
            continue
        semantic_type = node.semantic_type
        if semantic_type in FULL_TRACE_REPLAYABLE_TYPES:
            replay_ids.append(node_id)
        elif semantic_type == "final_answer" and node_id in last_final_answers:
            replay_ids.append(node_id)
    return replay_ids


def expand_plan_for_full_trace_replay(
    plan: Dict[str, Any],
    graph: DependencyGraph,
    task_id: str,
) -> Dict[str, Any]:
    """Return a plan copy whose replay set is the full replayable trace."""
    expanded = deepcopy(plan)
    replay_ids = full_trace_replay_step_ids(graph, task_id)
    replay_set = set(replay_ids)
    expanded["replay_step_ids"] = replay_ids
    for field in (
        "preserve_step_ids",
        "redundant_step_ids",
        "suspicious_step_ids",
    ):
        expanded[field] = [
            step_id for step_id in expanded.get(field, [])
            if step_id not in replay_set
        ]

    expanded["reason"] = rebuild_reason_counts(
        expanded,
        extra_markers=["skip_selective_replay", "full_trace_replay"],
    )
    return expanded


def resolve_memory_remap(memory_id: Any, memory_remap: Dict[str, str]) -> Any:
    """Resolve a possibly chained memory-id remap without looping."""
    if not isinstance(memory_id, str):
        return memory_id
    current = memory_id
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        mapped = memory_remap.get(current)
        if not isinstance(mapped, str) or not mapped:
            break
        current = mapped
    return current


def resolve_replay_memory_id(
    memory_id: Any,
    memory_remap: Dict[str, str],
    memory_store: Any,
    blocked_ids: set[str],
) -> Optional[str]:
    """Return the active remapped memory id, or None if replay may not touch it."""
    resolved = resolve_memory_remap(memory_id, memory_remap)
    if not isinstance(resolved, str) or resolved in blocked_ids:
        return None
    memory = memory_store.get_memory(resolved)
    if memory is None or getattr(memory, "status", None) != "active":
        return None
    return resolved


def remap_memory_mutation_step(
    step: Dict[str, Any],
    memory_remap: Dict[str, str],
    memory_store: Any,
    blocked_ids: set[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """Return a mutation-step copy with current active memory targets.

    ``generated_memory_ids`` intentionally remains in old-id space so the
    reconciliation result can establish ``old generated id -> new memory id``.
    """
    remapped = deepcopy(step)
    dropped_invalidated_ids: List[str] = []
    used_ids: List[Any] = []
    for identifier in remapped.get("used_ids") or []:
        if isinstance(identifier, str) and is_memory_id(identifier):
            resolved = resolve_replay_memory_id(
                identifier, memory_remap, memory_store, blocked_ids
            )
            if resolved is not None:
                used_ids.append(resolved)
        else:
            used_ids.append(identifier)
    remapped["used_ids"] = used_ids

    invalidated_ids: List[str] = []
    for memory_id in remapped.get("invalidated_memory_ids") or []:
        resolved = resolve_replay_memory_id(
            memory_id, memory_remap, memory_store, blocked_ids
        )
        if resolved is None:
            if isinstance(memory_id, str):
                dropped_invalidated_ids.append(memory_id)
            continue
        invalidated_ids.append(resolved)
    remapped["invalidated_memory_ids"] = list(dict.fromkeys(invalidated_ids))
    remapped["generated_memory_ids"] = list(remapped.get("generated_memory_ids") or [])
    return remapped, dropped_invalidated_ids
