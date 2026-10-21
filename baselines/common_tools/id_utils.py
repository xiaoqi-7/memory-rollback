"""Memory id utilities shared by baselines."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple


_MEMORY_ID_RE = re.compile(r"^m_(?:[fn])?(\d+)$")


def extract_memory_number(memory_id: str) -> int:
    match = _MEMORY_ID_RE.match(memory_id or "")
    return int(match.group(1)) if match else 0


def max_memory_number(memories: List[Dict[str, Any]]) -> int:
    return max(
        (extract_memory_number(memory.get("memory_id", "")) for memory in memories),
        default=0,
    )


def _remap_list(values: Any, memory_id_map: Dict[str, str]) -> Any:
    if not isinstance(values, list):
        return values
    return [memory_id_map.get(value, value) for value in values]


def rename_new_memories_to_mn(
    original_memories: List[Dict[str, Any]],
    repaired_memories: List[Dict[str, Any]],
    new_trace_steps: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, str]]:
    """Rename newly-created memory ids to m_nXXX without touching old steps."""
    original_ids = {
        memory.get("memory_id")
        for memory in original_memories
        if isinstance(memory.get("memory_id"), str)
    }
    next_slot = max_memory_number(original_memories) + 1
    memory_id_map: Dict[str, str] = {}

    renamed_memories = copy.deepcopy(repaired_memories)
    new_memory_ids: List[str] = []
    for memory in renamed_memories:
        memory_id = memory.get("memory_id")
        if not isinstance(memory_id, str) or memory_id in original_ids:
            continue
        renamed = f"m_n{next_slot:03d}"
        next_slot += 1
        memory_id_map[memory_id] = renamed
        memory["memory_id"] = renamed
        new_memory_ids.append(renamed)

    for memory in renamed_memories:
        if memory.get("supersedes") in memory_id_map:
            memory["supersedes"] = memory_id_map[memory["supersedes"]]
        memory["derived_from"] = _remap_list(memory.get("derived_from", []), memory_id_map)

    renamed_steps = copy.deepcopy(new_trace_steps)
    for step in renamed_steps:
        step["generated_memory_ids"] = _remap_list(
            step.get("generated_memory_ids", []), memory_id_map
        )
        step["invalidated_memory_ids"] = _remap_list(
            step.get("invalidated_memory_ids", []), memory_id_map
        )
        step["used_ids"] = _remap_list(step.get("used_ids", []), memory_id_map)

    return renamed_memories, renamed_steps, new_memory_ids, memory_id_map
