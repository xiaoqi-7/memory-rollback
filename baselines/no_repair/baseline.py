"""No-repair baseline implementation."""

from __future__ import annotations

from typing import Any, Dict, List


METHOD = "no_repair"


def build_noop_plan(
    *,
    method: str,
    task_id: str,
    faulty_memory_ids: List[str],
    original_trace_steps: List[Dict[str, Any]],
    case_id: str | None = None,
) -> Dict[str, Any]:
    plan = {
        "method": method,
        "task_id": task_id,
        "faulty_memory_ids": list(faulty_memory_ids),
        "delete_memory_ids": [],
        "quarantine_memory_ids": [],
        "invalidate_claim_ids": [],
        "replay_step_ids": [],
        "preserve_step_ids": [
            step["step_id"]
            for step in original_trace_steps
            if isinstance(step.get("step_id"), str)
        ],
        "redundant_step_ids": [],
        "suspicious_step_ids": [],
        "reason": "No repair applied; original faulty output retained.",
    }
    if case_id is not None:
        plan["case_id"] = case_id
    return plan


def build_noop_replay_result(
    *,
    task_id: str,
    original_trace_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "step_id_remap": {},
        "memory_id_remap": {},
        "generated_replacement_step_ids": [],
        "new_memory_ids": [],
        "invalidated_claim_ids": [],
        "llm_replay_call_count": 0,
        "llm_replay_calls_by_operation": {},
        "llm_token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "usage_available_call_count": 0,
            "usage_unavailable_call_count": 0,
            "by_operation": {},
        },
        "summary": {
            "replayed": 0,
            "pass_through": len(original_trace_steps),
            "superseded_skipped": 0,
        },
    }


def run_no_repair(
    task_id: str,
    steps: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    faulty_memory_ids: List[str],
    case_id: str | None = None,
    *,
    domain: str | None = None,
    **_: Any,
) -> Dict[str, Any]:
    plan = build_noop_plan(
        method=METHOD,
        task_id=task_id,
        faulty_memory_ids=faulty_memory_ids,
        original_trace_steps=steps,
        case_id=case_id,
    )
    replay_result = build_noop_replay_result(
        task_id=task_id,
        original_trace_steps=steps,
    )
    return {
        "plan": plan,
        "repaired_trace_steps": steps,
        "repaired_memories": memories,
        "replay_result": replay_result,
        "runtime_config": {"domain": domain} if domain is not None else None,
        "cost_overrides": None,
    }


def run(
    task: Dict[str, Any],
    faulty_trace: List[Dict[str, Any]],
    memory_store: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return run_no_repair(
        task["task_id"],
        faulty_trace,
        memory_store,
        task.get("faulty_memory_ids", []),
        case_id=task.get("case_id"),
        domain=task.get("domain"),
    )
