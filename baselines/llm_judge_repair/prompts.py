"""Prompt construction for the LLM-judge rollback-plan baseline."""

from __future__ import annotations

import json
from typing import Any, Dict, List


SYSTEM_PROMPT = """You generate rollback plans for a memory-repair executor.
Return strict JSON only: no markdown, no commentary."""


EXECUTOR_CONTRACT = {
    "delete_memory_ids": "memory ids to mark deleted",
    "quarantine_memory_ids": "memory ids to mark quarantined",
    "invalidate_claim_ids": "claim step ids to invalidate",
    "replay_step_ids": (
        "existing trace step ids that executor should replay; choose a boundary "
        "early enough to regenerate any factual evidence no longer supported by "
        "remaining active memory"
    ),
    "preserve_step_ids": "existing trace step ids to preserve",
    "redundant_step_ids": "existing trace step ids made obsolete",
    "suspicious_step_ids": "uncertain trace step ids",
}


def _compact_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    fields = (
        "memory_id",
        "content",
        "status",
        "source",
        "user_id",
        "created_at",
        "last_modified_at",
        "last_modified_by",
        "fault_type",
        "derived_from",
        "supersedes",
    )
    return {field: memory.get(field) for field in fields if field in memory}


def _compact_step(step: Dict[str, Any]) -> Dict[str, Any]:
    fields = (
        "step_id",
        "step_type",
        "content",
        "used_ids",
        "generated_memory_ids",
        "invalidated_memory_ids",
        "tool_name",
        "tool_args",
    )
    return {field: step.get(field) for field in fields if field in step}


def build_prompt_payload(
    *,
    task_id: str,
    case_id: str,
    faulty_memory_ids: List[str],
    memories: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    session: List[str],
) -> Dict[str, Any]:
    final_answers = [
        step for step in steps if step.get("step_type") == "final_answer"
    ]
    last_final_answer = final_answers[-1] if final_answers else None
    all_memory_ids = [
        memory["memory_id"]
        for memory in memories
        if isinstance(memory.get("memory_id"), str)
    ]
    active_memory_ids = [
        memory["memory_id"]
        for memory in memories
        if memory.get("status") == "active"
        and isinstance(memory.get("memory_id"), str)
    ]
    all_step_ids = [
        step["step_id"] for step in steps if isinstance(step.get("step_id"), str)
    ]
    claim_step_ids = [
        step["step_id"]
        for step in steps
        if step.get("step_type") == "claim"
        and isinstance(step.get("step_id"), str)
    ]
    tool_action_step_ids = [
        step["step_id"]
        for step in steps
        if step.get("step_type") == "tool_action"
        and isinstance(step.get("step_id"), str)
    ]
    tool_observation_step_ids = [
        step["step_id"]
        for step in steps
        if step.get("step_type") == "tool_observation"
        and isinstance(step.get("step_id"), str)
    ]
    memory_read_step_ids = [
        step["step_id"]
        for step in steps
        if step.get("step_type") == "memory_read"
        and isinstance(step.get("step_id"), str)
    ]
    final_answer_step_ids = [
        step["step_id"]
        for step in steps
        if step.get("step_type") == "final_answer"
        and isinstance(step.get("step_id"), str)
    ]
    return {
        "task_id": task_id,
        "case_id": case_id,
        "session_turns": list(session or []),
        "faulty_memory_ids": list(faulty_memory_ids),
        "required_delete_memory_ids": list(faulty_memory_ids),
        "memories": [_compact_memory(memory) for memory in memories],
        "faulty_trace_steps": [_compact_step(step) for step in steps],
        "wrong_final_answer": (
            {
                "step_id": last_final_answer.get("step_id"),
                "content": last_final_answer.get("content"),
            }
            if last_final_answer
            else None
        ),
        "executor_plan_contract": EXECUTOR_CONTRACT,
        "available_valid_ids": {
            "all_memory_ids": all_memory_ids,
            "active_memory_ids": active_memory_ids,
            "claim_step_ids": claim_step_ids,
            "all_trace_step_ids": all_step_ids,
            "tool_action_step_ids": tool_action_step_ids,
            "tool_observation_step_ids": tool_observation_step_ids,
            "memory_read_step_ids": memory_read_step_ids,
            "final_answer_step_ids": final_answer_step_ids,
        },
        "output_schema": {
            "delete_memory_ids": ["..."],
            "quarantine_memory_ids": ["..."],
            "invalidate_claim_ids": ["..."],
            "replay_step_ids": ["..."],
            "preserve_step_ids": ["..."],
            "redundant_step_ids": ["..."],
            "suspicious_step_ids": ["..."],
            "reason": "...",
        },
        "rules": [
            "Return a JSON object only.",
            "Do not invent ids.",
            "Memory ids must come from all_memory_ids.",
            "delete_memory_ids must include every id in required_delete_memory_ids.",
            "Do not quarantine source faulty memories instead of deleting them.",
            "You may additionally delete or quarantine derived memories if needed.",
            "Claim ids must come from claim_step_ids.",
            "Step ids must come from all_trace_step_ids.",
            "invalidate_claim_ids must be a subset of replay_step_ids union suspicious_step_ids.",
            "Step disposition lists must be mutually exclusive.",
            "If a claim or final answer depends on faulty memory, include the affected claim/final_answer chain in replay_step_ids.",
            "memory_read steps only retrieve current memory state; they do not create new factual evidence.",
            "If deleting or quarantining a memory removes the only concrete evidence for a factual answer, replaying memory_read or final_answer alone is insufficient.",
            "If the corrected final answer requires external/tool-derived facts and no remaining active memory directly supports those facts, include enough upstream steps in replay_step_ids for the executor to regenerate or reacquire evidence.",
            "When a relevant tool_action/tool_observation pair exists upstream and provides the factual evidence needed by the final answer, include the smallest necessary plan/tool_action/tool_observation/final_answer region, unless the executor can safely replan from a later step.",
            "If the original faulty final answer used a poisoned factual value, do not preserve that factual answer unless a remaining reliable memory or replayed tool observation supports it.",
            "Prefer minimal repair: do not delete unrelated benign memories or replay unrelated earlier turns.",
            "Prefer minimal repair, but not at the expense of leaving the final answer unsupported.",
            "If unsure, replay affected final-turn reasoning/final_answer rather than clearing all memory.",
            "The replay boundary should be early enough that replacement steps do not cite deleted or quarantined memories.",
            "If replaying from an early boundary, include a dependency-complete replay region from that boundary through the affected final_answer. Do not select sparse alternating steps across the trace.",
            "Omitted intermediate steps can leave replacement steps citing stale, deleted, quarantined, or inactive memories; include any intermediate memory_write, memory_update, memory_delete, memory_read, claim, plan, tool_action, tool_observation, and final_answer steps needed to keep references valid.",
            "If the intended repair only affects the last user turn, prefer a compact final-turn replay region over replaying scattered steps from the whole trace.",
            "When replaying affected claim steps, include those claim step ids in invalidate_claim_ids unless the claim is explicitly preserved unchanged.",
            "Include the final_answer step in replay_step_ids if the final answer may change.",
            "In reason, mention: faulty memory -> affected claim/final_answer, whether remaining memories are sufficient evidence, and why the replay boundary is enough.",
        ],
    }


def build_user_prompt(payload: Dict[str, Any], validation_error: str | None = None) -> str:
    if validation_error:
        payload = dict(payload)
        payload["previous_validation_error"] = validation_error
        payload["retry_instruction"] = (
            "Fix the validation error and return one strict JSON object."
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)
