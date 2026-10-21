"""Trace helpers shared by baselines."""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple


_STEP_ID_RE = re.compile(r"^s_(?:n)?(\d+)$")


def extract_step_number(step_id: str) -> int:
    match = _STEP_ID_RE.match(step_id or "")
    return int(match.group(1)) if match else 0


def max_step_number(steps: List[Dict[str, Any]]) -> int:
    return max(
        (extract_step_number(step.get("step_id", "")) for step in steps),
        default=0,
    )


def _remap_step_refs(values: Any, step_id_map: Dict[str, str]) -> Any:
    if not isinstance(values, list):
        return values
    return [step_id_map.get(value, value) for value in values]


def rename_new_steps_to_sn(
    original_steps: List[Dict[str, Any]],
    new_steps: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    """Rename replacement step ids to s_nXXX and update only new-step refs."""
    next_slot = max_step_number(original_steps) + 1
    step_id_map: Dict[str, str] = {}
    new_step_ids: List[str] = []

    for step in new_steps:
        old_step_id = step.get("step_id")
        if not isinstance(old_step_id, str) or not old_step_id:
            continue
        renamed_step_id = f"s_n{next_slot:03d}"
        next_slot += 1
        step_id_map[old_step_id] = renamed_step_id
        new_step_ids.append(renamed_step_id)

    renamed_steps = copy.deepcopy(new_steps)
    for step in renamed_steps:
        step_id = step.get("step_id")
        if isinstance(step_id, str) and step_id in step_id_map:
            step["step_id"] = step_id_map[step_id]
        step["prev_step_ids"] = _remap_step_refs(
            step.get("prev_step_ids", []), step_id_map
        )
        step["next_step_ids"] = _remap_step_refs(
            step.get("next_step_ids", []), step_id_map
        )
        step["used_ids"] = _remap_step_refs(step.get("used_ids", []), step_id_map)

    return renamed_steps, step_id_map, new_step_ids


def find_final_turn_boundary(
    steps: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    final_index = None
    for index in range(len(steps) - 1, -1, -1):
        if steps[index].get("step_type") == "final_answer":
            final_index = index
            break
    if final_index is None:
        raise ValueError("trace has no final_answer step")

    start_index = None
    for index in range(final_index, -1, -1):
        if steps[index].get("step_type") == "memory_read":
            start_index = index
            break
    if start_index is None:
        raise ValueError("trace final turn has no memory_read step")

    return steps[:start_index], steps[start_index:]


def step_ids(steps: List[Dict[str, Any]]) -> List[str]:
    return [
        step["step_id"]
        for step in steps
        if isinstance(step.get("step_id"), str) and step.get("step_id")
    ]


def align_step_id_remap_by_type(
    old_steps: List[Dict[str, Any]],
    new_steps: List[Dict[str, Any]],
) -> Dict[str, str]:
    old_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    new_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for step in old_steps:
        old_by_type[step.get("step_type")].append(step)
    for step in new_steps:
        new_by_type[step.get("step_type")].append(step)

    remap: Dict[str, str] = {}
    for step_type, old_group in old_by_type.items():
        for old_step, new_step in zip(old_group, new_by_type.get(step_type, [])):
            old_id = old_step.get("step_id")
            new_id = new_step.get("step_id")
            if isinstance(old_id, str) and isinstance(new_id, str):
                remap[old_id] = new_id
    return remap


def _tool_identity(step: Dict[str, Any]) -> tuple:
    return (
        step.get("tool_name"),
        json.dumps(step.get("tool_args") or {}, sort_keys=True, default=str),
    )


def classify_tool_calls(
    old_final_turn_steps: List[Dict[str, Any]],
    new_final_turn_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    old_actions = [
        step for step in old_final_turn_steps if step.get("step_type") == "tool_action"
    ]
    new_actions = [
        step for step in new_final_turn_steps if step.get("step_type") == "tool_action"
    ]
    unmatched_old_by_identity: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for step in old_actions:
        unmatched_old_by_identity[_tool_identity(step)].append(step)

    replayed_old_ids: List[str] = []
    synthetic_new_ids: List[str] = []
    for new_step in new_actions:
        identity = _tool_identity(new_step)
        candidates = unmatched_old_by_identity.get(identity) or []
        if candidates:
            old_step = candidates.pop(0)
            replayed_old_ids.append(old_step["step_id"])
        else:
            synthetic_new_ids.append(new_step["step_id"])

    return {
        "replayed_old_tool_action_ids": replayed_old_ids,
        "synthetic_new_tool_action_ids": synthetic_new_ids,
        "replayed_tool_call_count": len(replayed_old_ids),
        "synthetic_tool_call_count": len(synthetic_new_ids),
    }
