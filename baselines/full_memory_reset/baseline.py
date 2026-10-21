"""Full-memory-reset baseline."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from agent.run_agent import AgentRunner
from baselines.common_tools.id_utils import rename_new_memories_to_mn
from baselines.common_tools.trace_utils import (
    align_step_id_remap_by_type,
    classify_tool_calls,
    find_final_turn_boundary,
    rename_new_steps_to_sn,
    step_ids,
)
from baselines.full_memory_reset.lifecycle import run_full_memory_reset_lifecycle


METHOD = "full_memory_reset"


def _drop_old_final_turn_links(
    old_prefix_steps: List[Dict[str, Any]],
    old_final_turn_ids: set[str],
) -> List[Dict[str, Any]]:
    prefix = copy.deepcopy(old_prefix_steps)
    for step in prefix:
        step["next_step_ids"] = [
            sid for sid in (step.get("next_step_ids") or [])
            if sid not in old_final_turn_ids
        ]
    return prefix


def _build_plan(
    *,
    task_id: str,
    faulty_memory_ids: List[str],
    original_active_memory_ids: List[str],
    old_prefix_step_ids: List[str],
    old_final_turn_step_ids: List[str],
    case_id: str | None,
) -> Dict[str, Any]:
    plan = {
        "method": METHOD,
        "task_id": task_id,
        "faulty_memory_ids": list(faulty_memory_ids),
        "delete_memory_ids": list(original_active_memory_ids),
        "quarantine_memory_ids": [],
        "invalidate_claim_ids": [],
        "replay_step_ids": list(old_final_turn_step_ids),
        "preserve_step_ids": list(old_prefix_step_ids),
        "redundant_step_ids": [],
        "suspicious_step_ids": [],
        "reason": (
            "Deleted all original active memories and reran the final user turn "
            "through the full baseline lifecycle."
        ),
    }
    if case_id is not None:
        plan["case_id"] = case_id
    return plan


def run_full_memory_reset(
    task_id: str,
    steps: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    faulty_memory_ids: List[str],
    case_id: str | None = None,
    *,
    domain: str | None = None,
    session: List[str] | None = None,
    agent_config: str | None = None,
) -> Dict[str, Any]:
    if session is None:
        raise ValueError("full_memory_reset requires session")
    if agent_config is None:
        raise ValueError("full_memory_reset requires agent_config")

    runner = AgentRunner(agent_config, mode="clean")
    old_prefix_steps, old_final_turn_steps = find_final_turn_boundary(steps)
    old_final_turn_ids = set(step_ids(old_final_turn_steps))
    old_prefix_steps = _drop_old_final_turn_links(old_prefix_steps, old_final_turn_ids)
    old_prefix_step_ids = step_ids(old_prefix_steps)
    old_final_turn_step_ids = step_ids(old_final_turn_steps)
    original_active_memory_ids = [
        memory["memory_id"]
        for memory in memories
        if memory.get("status") == "active"
        and isinstance(memory.get("memory_id"), str)
    ]

    lifecycle_result = run_full_memory_reset_lifecycle(
        task_id=task_id,
        session=session,
        original_memories=memories,
        old_prefix_steps=old_prefix_steps,
        tools=runner.tools,
        config=runner.config,
    )

    raw_new_steps = lifecycle_result.new_final_turn_steps
    renamed_steps, _lifecycle_step_id_map, new_step_ids = rename_new_steps_to_sn(
        steps,
        raw_new_steps,
    )
    repaired_memories, renamed_new_steps, new_memory_ids, _memory_id_map = (
        rename_new_memories_to_mn(
            memories,
            lifecycle_result.repaired_memories,
            renamed_steps,
        )
    )
    repaired_trace_steps = old_prefix_steps + renamed_new_steps
    new_final_turn_step_ids = new_step_ids
    step_id_remap = align_step_id_remap_by_type(
        old_final_turn_steps,
        renamed_new_steps,
    )
    tool_classification = classify_tool_calls(old_final_turn_steps, renamed_new_steps)

    plan = _build_plan(
        task_id=task_id,
        faulty_memory_ids=faulty_memory_ids,
        original_active_memory_ids=original_active_memory_ids,
        old_prefix_step_ids=old_prefix_step_ids,
        old_final_turn_step_ids=old_final_turn_step_ids,
        case_id=case_id,
    )
    replay_result = {
        "task_id": task_id,
        "step_id_remap": step_id_remap,
        "memory_id_remap": {},
        "generated_replacement_step_ids": new_final_turn_step_ids,
        "new_memory_ids": new_memory_ids,
        "invalidated_claim_ids": [],
        "llm_replay_call_count": lifecycle_result.tracker.llm_call_count,
        "llm_replay_calls_by_operation": (
            lifecycle_result.tracker.calls_by_operation_payload()
        ),
        "llm_token_usage": lifecycle_result.tracker.token_usage_payload(),
        "summary": {
            "replayed": len(old_final_turn_step_ids),
            "pass_through": len(old_prefix_step_ids),
            "superseded_skipped": 0,
        },
    }

    replayed_old_tool_ids = tool_classification["replayed_old_tool_action_ids"]
    synthetic_new_tool_ids = tool_classification["synthetic_new_tool_action_ids"]
    cost_overrides = {
        "planned_replay_step_ids": old_final_turn_step_ids,
        "replayed_target_step_ids": old_final_turn_step_ids,
        "replayed_target_step_count": len(old_final_turn_step_ids),
        "replayed_tool_action_ids": replayed_old_tool_ids,
        "replayed_tool_call_count": len(replayed_old_tool_ids),
        "synthetic_tool_action_step_ids": synthetic_new_tool_ids,
        "synthetic_tool_call_count": len(synthetic_new_tool_ids),
        "total_repair_tool_call_count": (
            len(replayed_old_tool_ids) + len(synthetic_new_tool_ids)
        ),
        "generated_replacement_step_ids": new_final_turn_step_ids,
        "generated_replacement_step_count": len(new_final_turn_step_ids),
        "llm_replay_call_count": lifecycle_result.tracker.llm_call_count,
        "llm_replay_calls_by_operation": (
            lifecycle_result.tracker.calls_by_operation_payload()
        ),
        "total_llm_call_count": lifecycle_result.tracker.llm_call_count,
        "llm_token_usage": lifecycle_result.tracker.token_usage_payload(),
        "step_id_remap": step_id_remap,
    }

    return {
        "plan": plan,
        "repaired_trace_steps": repaired_trace_steps,
        "repaired_memories": repaired_memories,
        "replay_result": replay_result,
        "runtime_config": {"domain": domain or runner.config.get("domain")},
        "cost_overrides": cost_overrides,
    }


def run(
    task: Dict[str, Any],
    faulty_trace: List[Dict[str, Any]],
    memory_store: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return run_full_memory_reset(
        task["task_id"],
        faulty_trace,
        memory_store,
        task.get("faulty_memory_ids", []),
        case_id=task.get("case_id"),
        domain=task.get("domain"),
        session=task.get("session"),
        agent_config=task.get("agent_config"),
    )
