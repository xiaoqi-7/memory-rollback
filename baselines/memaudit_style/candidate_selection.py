"""Candidate-memory selection for the MemAudit-style baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from baselines.common_tools.trace_utils import step_ids
from baselines.delete_retrieved_memories.trace_active_state import (
    compute_active_memory_state_before_last_user_turn,
)


EXCLUDED_STEP_TYPES = {"memory_read"}


@dataclass
class CandidateSelection:
    candidate_memory_ids: List[str]
    final_turn_used_memory_ids: List[str]
    active_memory_ids_before_last_user_turn: List[str]
    fallback_used: bool
    candidate_source_step_ids: List[str]


def _memory_ids(memories: List[Dict[str, Any]]) -> set[str]:
    return {
        memory["memory_id"]
        for memory in memories
        if isinstance(memory.get("memory_id"), str)
    }


def select_candidate_memories(
    *,
    memories: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    old_final_turn_steps: List[Dict[str, Any]],
    faulty_memory_ids: List[str],
) -> CandidateSelection:
    memory_ids = _memory_ids(memories)
    active_state = compute_active_memory_state_before_last_user_turn(memories, steps)
    active_before_last = active_state.active_memory_ids
    active_before_last_set = set(active_before_last)

    candidate_source_steps = [
        step
        for step in old_final_turn_steps
        if step.get("step_type") not in EXCLUDED_STEP_TYPES
    ]
    seen_used: set[str] = set()
    final_turn_used_memory_ids: List[str] = []
    for step in candidate_source_steps:
        for used_id in step.get("used_ids", []) or []:
            if used_id in memory_ids and used_id not in seen_used:
                seen_used.add(used_id)
                final_turn_used_memory_ids.append(used_id)

    candidate_memory_ids = [
        memory_id
        for memory_id in active_before_last
        if memory_id in seen_used
    ]
    fallback_used = False
    if not candidate_memory_ids:
        fallback = [
            memory_id
            for memory_id in faulty_memory_ids
            if memory_id in active_before_last_set
        ]
        if fallback:
            fallback_used = True
            candidate_memory_ids = list(dict.fromkeys(fallback))

    return CandidateSelection(
        candidate_memory_ids=candidate_memory_ids,
        final_turn_used_memory_ids=final_turn_used_memory_ids,
        active_memory_ids_before_last_user_turn=active_before_last,
        fallback_used=fallback_used,
        candidate_source_step_ids=step_ids(candidate_source_steps),
    )
