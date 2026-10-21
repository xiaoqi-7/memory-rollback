"""Reconstruct active memory ids immediately before the final user turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_DEACTIVATING_STEP_TYPES = {
    "memory_delete",
    "memory_update",
    "memory_consolidate",
}
_CREATING_STEP_TYPES = {
    "memory_write",
    "memory_update",
    "memory_consolidate",
}
_TERMINAL_STATUSES = {"deleted", "quarantined", "superseded", "inactive"}


@dataclass(frozen=True)
class ActiveMemoryState:
    active_memory_ids: list[str]
    unrecognized_memory_lifecycle_event_count: int


def _memory_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [
        value for value in values
        if isinstance(value, str) and value.startswith("m_")
    ]


def _last_user_turn_start_index(steps: Sequence[Mapping[str, Any]]) -> int:
    for index in range(len(steps) - 1, -1, -1):
        if steps[index].get("step_type") == "memory_read":
            return index
    raise ValueError("trace has no memory_read step for final user turn")


def compute_active_memory_state_before_last_user_turn(
    memories: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> ActiveMemoryState:
    """Apply memory lifecycle events before the final turn in trace order."""
    cutoff = _last_user_turn_start_index(steps)
    produced_ids = {
        memory_id
        for step in steps
        for memory_id in _memory_ids(step.get("generated_memory_ids"))
    }
    original_order = [
        memory["memory_id"]
        for memory in memories
        if isinstance(memory.get("memory_id"), str)
    ]
    active = {
        memory["memory_id"]
        for memory in memories
        if isinstance(memory.get("memory_id"), str)
        and memory.get("memory_id") not in produced_ids
        and memory.get("status") not in _TERMINAL_STATUSES
    }
    trace_created_order: list[str] = []
    unrecognized = 0

    for step in steps[:cutoff]:
        step_type = step.get("step_type")
        generated = _memory_ids(step.get("generated_memory_ids"))
        invalidated = _memory_ids(step.get("invalidated_memory_ids"))

        if invalidated:
            if step_type in _DEACTIVATING_STEP_TYPES:
                active.difference_update(invalidated)
            else:
                unrecognized += 1

        if generated:
            if step_type in _CREATING_STEP_TYPES:
                for memory_id in generated:
                    active.add(memory_id)
                    if memory_id not in trace_created_order:
                        trace_created_order.append(memory_id)
            else:
                unrecognized += 1

        status = step.get("status") or step.get("memory_status")
        target_ids = _memory_ids(step.get("memory_ids")) + _memory_ids(
            step.get("target_memory_ids")
        )
        if target_ids and isinstance(status, str):
            if status in _TERMINAL_STATUSES:
                active.difference_update(target_ids)
            elif status == "active":
                for memory_id in target_ids:
                    active.add(memory_id)
                    if memory_id not in trace_created_order:
                        trace_created_order.append(memory_id)
            else:
                unrecognized += 1

    ordered: list[str] = []
    for memory_id in original_order:
        if memory_id in active and memory_id not in trace_created_order:
            ordered.append(memory_id)
    for memory_id in trace_created_order:
        if memory_id in active:
            ordered.append(memory_id)

    return ActiveMemoryState(
        active_memory_ids=ordered,
        unrecognized_memory_lifecycle_event_count=unrecognized,
    )


def compute_active_memory_ids_before_last_user_turn(
    memories: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> list[str]:
    return compute_active_memory_state_before_last_user_turn(
        memories,
        steps,
    ).active_memory_ids
