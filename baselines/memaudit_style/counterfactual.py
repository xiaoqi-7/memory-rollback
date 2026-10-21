"""Counterfactual replay helpers for MemAudit-style memory auditing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from baselines.common_tools.id_utils import rename_new_memories_to_mn
from baselines.common_tools.trace_utils import (
    align_step_id_remap_by_type,
    classify_tool_calls,
    rename_new_steps_to_sn,
)
from baselines.full_memory_reset.lifecycle import run_full_memory_reset_lifecycle
from baselines.memaudit_style.oracle_adapter import (
    judge_recovery,
    score_oracle_trace,
)


@dataclass
class ReplayCase:
    repaired_trace_steps: List[Dict[str, Any]]
    repaired_memories: List[Dict[str, Any]]
    new_step_ids: List[str]
    new_memory_ids: List[str]
    step_id_remap: Dict[str, str]
    llm_call_count: int
    calls_by_operation: Dict[str, Any]
    token_usage: Dict[str, Any]
    replayed_tool_action_ids: List[str]
    synthetic_tool_action_step_ids: List[str]
    total_repair_tool_call_count: int


def _materialize_replay_case(
    *,
    original_steps: List[Dict[str, Any]],
    original_memories: List[Dict[str, Any]],
    old_prefix_steps: List[Dict[str, Any]],
    old_final_turn_steps: List[Dict[str, Any]],
    lifecycle_result: Any,
) -> ReplayCase:
    renamed_steps, _lifecycle_step_id_map, new_step_ids = rename_new_steps_to_sn(
        original_steps,
        lifecycle_result.new_final_turn_steps,
    )
    repaired_memories, renamed_new_steps, new_memory_ids, _memory_id_map = (
        rename_new_memories_to_mn(
            original_memories,
            lifecycle_result.repaired_memories,
            renamed_steps,
        )
    )
    repaired_trace_steps = old_prefix_steps + renamed_new_steps
    step_id_remap = align_step_id_remap_by_type(
        old_final_turn_steps,
        renamed_new_steps,
    )
    tool_classification = classify_tool_calls(old_final_turn_steps, renamed_new_steps)
    replayed_tool_ids = tool_classification["replayed_old_tool_action_ids"]
    synthetic_tool_ids = tool_classification["synthetic_new_tool_action_ids"]
    return ReplayCase(
        repaired_trace_steps=repaired_trace_steps,
        repaired_memories=repaired_memories,
        new_step_ids=new_step_ids,
        new_memory_ids=new_memory_ids,
        step_id_remap=step_id_remap,
        llm_call_count=lifecycle_result.tracker.llm_call_count,
        calls_by_operation=lifecycle_result.tracker.calls_by_operation_payload(),
        token_usage=lifecycle_result.tracker.token_usage_payload(),
        replayed_tool_action_ids=replayed_tool_ids,
        synthetic_tool_action_step_ids=synthetic_tool_ids,
        total_repair_tool_call_count=len(replayed_tool_ids) + len(synthetic_tool_ids),
    )


def run_replay_case(
    *,
    task_id: str,
    session: List[str],
    original_steps: List[Dict[str, Any]],
    original_memories: List[Dict[str, Any]],
    old_prefix_steps: List[Dict[str, Any]],
    old_final_turn_steps: List[Dict[str, Any]],
    tools: Dict[str, Any],
    config: Dict[str, Any],
    delete_memory_ids: List[str],
    modified_by: str,
) -> ReplayCase:
    lifecycle_result = run_full_memory_reset_lifecycle(
        task_id=task_id,
        session=session,
        original_memories=original_memories,
        old_prefix_steps=old_prefix_steps,
        tools=tools,
        config=config,
        delete_memory_ids=delete_memory_ids,
        modified_by=modified_by,
    )
    return _materialize_replay_case(
        original_steps=original_steps,
        original_memories=original_memories,
        old_prefix_steps=old_prefix_steps,
        old_final_turn_steps=old_final_turn_steps,
        lifecycle_result=lifecycle_result,
    )


def run_counterfactual_audit(
    *,
    task_id: str,
    domain: str,
    task: Dict[str, Any],
    session: List[str],
    original_steps: List[Dict[str, Any]],
    original_memories: List[Dict[str, Any]],
    old_prefix_steps: List[Dict[str, Any]],
    old_final_turn_steps: List[Dict[str, Any]],
    tools: Dict[str, Any],
    config: Dict[str, Any],
    candidate_memory_ids: List[str],
) -> List[Dict[str, Any]]:
    original_score_details = score_oracle_trace(
        task=task,
        trace_steps=original_steps,
    )
    original_oracle_score = float(original_score_details.get("oracle_score", 0.0))
    results: List[Dict[str, Any]] = []
    for memory_id in candidate_memory_ids:
        replay_case = run_replay_case(
            task_id=task_id,
            session=session,
            original_steps=original_steps,
            original_memories=original_memories,
            old_prefix_steps=old_prefix_steps,
            old_final_turn_steps=old_final_turn_steps,
            tools=tools,
            config=config,
            delete_memory_ids=[memory_id],
            modified_by="memaudit_style_counterfactual",
        )
        oracle_result = judge_recovery(
            domain=domain,
            task=task,
            memories=replay_case.repaired_memories,
            trace_steps=replay_case.repaired_trace_steps,
        )
        counterfactual_score_details = score_oracle_trace(
            task=task,
            trace_steps=replay_case.repaired_trace_steps,
        )
        counterfactual_oracle_score = float(
            counterfactual_score_details.get("oracle_score", 0.0)
        )
        oracle_score_delta = counterfactual_oracle_score - original_oracle_score
        if original_oracle_score >= 1.0:
            influence_score = 0.0
        else:
            influence_score = max(0.0, oracle_score_delta) / (
                1.0 - original_oracle_score
            )
        results.append(
            {
                "memory_id": memory_id,
                "oracle": oracle_result,
                "influence_score": influence_score,
                "original_oracle_score": original_oracle_score,
                "counterfactual_oracle_score": counterfactual_oracle_score,
                "oracle_score_delta": oracle_score_delta,
                "oracle_score_improvement": influence_score,
                "oracle_score_details": {
                    "original": original_score_details,
                    "counterfactual": counterfactual_score_details,
                },
                "llm_call_count": replay_case.llm_call_count,
                "llm_calls_by_operation": replay_case.calls_by_operation,
                "llm_token_usage": replay_case.token_usage,
                "replayed_target_step_count": len(old_final_turn_steps),
                "replayed_tool_action_ids": replay_case.replayed_tool_action_ids,
                "synthetic_tool_action_step_ids": (
                    replay_case.synthetic_tool_action_step_ids
                ),
                "total_repair_tool_call_count": (
                    replay_case.total_repair_tool_call_count
                ),
                "generated_replacement_step_ids": replay_case.new_step_ids,
                "new_memory_ids": replay_case.new_memory_ids,
            }
        )
    return results
