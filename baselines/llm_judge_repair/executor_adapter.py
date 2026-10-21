"""Adapter from LLM-judge plans to the existing rollback executor."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List

from methods.ours.build_dependency_graph import build_dependency_graph
from methods.ours.rollback_executor import build_repaired_result_payload
from methods.ours.run_ours import _make_replay_context_observer, run_rollback_executor


JUDGE_ARTIFACT_FILENAMES = {
    "repaired_result": "repaired_result.json",
    "repaired_memories": "repaired_memories.jsonl",
    "repaired_tracesteps": "repaired_tracesteps.jsonl",
    "rollback_plan": "rollback_plan.json",
    "llm_judge_prompt": "llm_judge_prompt.json",
    "llm_judge_raw_response": "llm_judge_raw_response.json",
    "llm_judge_normalized_plan": "llm_judge_normalized_plan.json",
}


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, ensure_ascii=False)
            fh.write("\n")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _noop_replay_result(
    *,
    task_id: str,
    steps: List[Dict[str, Any]],
    decision_calls_by_operation: Dict[str, int],
    decision_token_usage: Dict[str, Any],
    repair_failed: bool,
    error: str | None,
    extra_summary: Dict[str, Any],
) -> Dict[str, Any]:
    summary = {
        "replayed": 0,
        "pass_through": len(steps),
        "superseded_skipped": 0,
        "repair_failed": repair_failed,
    }
    if error:
        summary["error"] = error
    summary.update(extra_summary)
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
        "summary": summary,
    }


def _summary_counts(plan: Dict[str, Any], judge_attempt_count: int) -> Dict[str, Any]:
    return {
        "repair_failed": False,
        "judge_attempt_count": judge_attempt_count,
        "normalized_delete_memory_count": len(plan.get("delete_memory_ids", [])),
        "normalized_quarantine_memory_count": len(
            plan.get("quarantine_memory_ids", [])
        ),
        "normalized_invalidate_claim_count": len(
            plan.get("invalidate_claim_ids", [])
        ),
        "normalized_replay_step_count": len(plan.get("replay_step_ids", [])),
    }


def _write_repaired_result(
    *,
    output_dir: str,
    plan: Dict[str, Any],
    steps: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    repaired_steps: List[Dict[str, Any]],
    repaired_memories: List[Dict[str, Any]],
    replay_result: Dict[str, Any],
    decision_calls_by_operation: Dict[str, int],
    decision_token_usage: Dict[str, Any],
    runtime_config: Dict[str, Any] | None,
) -> Dict[str, Any]:
    payload = build_repaired_result_payload(
        plan,
        steps,
        memories,
        repaired_steps,
        repaired_memories,
        replay_result,
        llm_repair_decision_calls_by_operation=decision_calls_by_operation,
        llm_repair_decision_token_usage=decision_token_usage,
        runtime_config=runtime_config,
        artifact_filenames=JUDGE_ARTIFACT_FILENAMES,
    )
    internal_summary = replay_result.get("summary", {})
    if isinstance(internal_summary, dict):
        for key, value in internal_summary.items():
            if key not in payload["summary"]:
                payload["summary"][key] = value
    _write_json(os.path.join(output_dir, "repaired_result.json"), payload)
    return payload


def execute_llm_judge_plan(
    *,
    task_id: str,
    plan: Dict[str, Any],
    steps: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    session: List[str],
    output_dir: str,
    agent_config: str,
    runtime_config: Dict[str, Any] | None,
    decision_calls_by_operation: Dict[str, int],
    decision_token_usage: Dict[str, Any],
    judge_attempt_count: int,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, "rollback_plan.json"), plan)

    summary_counts = _summary_counts(plan, judge_attempt_count)
    try:
        graph = build_dependency_graph(task_id, steps, memories, session)
        replay_context_path = os.path.join(output_dir, "replay_context.json")
        replay_result = run_rollback_executor(
            graph,
            plan,
            steps,
            memories,
            session,
            output_dir,
            agent_config_path=agent_config,
            llm_repair_decision_calls_by_operation=decision_calls_by_operation,
            replay_context_observer=_make_replay_context_observer(
                replay_context_path
            ),
        )
        repaired_steps = _load_jsonl(os.path.join(output_dir, "repaired_tracesteps.jsonl"))
        repaired_memories = _load_jsonl(os.path.join(output_dir, "repaired_memories.jsonl"))
        replay_result.setdefault("summary", {}).update(summary_counts)
        payload = _write_repaired_result(
            output_dir=output_dir,
            plan=plan,
            steps=steps,
            memories=memories,
            repaired_steps=repaired_steps,
            repaired_memories=repaired_memories,
            replay_result=replay_result,
            decision_calls_by_operation=decision_calls_by_operation,
            decision_token_usage=decision_token_usage,
            runtime_config=runtime_config,
        )
        return {
            "executor_succeeded": True,
            "replay_result": replay_result,
            "repaired_trace_steps": repaired_steps,
            "repaired_memories": repaired_memories,
            "payload": payload,
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        failed_summary = dict(summary_counts)
        failed_summary["repair_failed"] = True
        replay_result = _noop_replay_result(
            task_id=task_id,
            steps=steps,
            decision_calls_by_operation=decision_calls_by_operation,
            decision_token_usage=decision_token_usage,
            repair_failed=True,
            error=error,
            extra_summary=failed_summary,
        )
        repaired_steps = copy.deepcopy(steps)
        repaired_memories = copy.deepcopy(memories)
        _write_jsonl(os.path.join(output_dir, "repaired_tracesteps.jsonl"), repaired_steps)
        _write_jsonl(os.path.join(output_dir, "repaired_memories.jsonl"), repaired_memories)
        payload = _write_repaired_result(
            output_dir=output_dir,
            plan=plan,
            steps=steps,
            memories=memories,
            repaired_steps=repaired_steps,
            repaired_memories=repaired_memories,
            replay_result=replay_result,
            decision_calls_by_operation=decision_calls_by_operation,
            decision_token_usage=decision_token_usage,
            runtime_config=runtime_config,
        )
        return {
            "executor_succeeded": False,
            "executor_error": error,
            "replay_result": replay_result,
            "repaired_trace_steps": repaired_steps,
            "repaired_memories": repaired_memories,
            "payload": payload,
        }


def write_failed_judge_outputs(
    *,
    task_id: str,
    plan: Dict[str, Any],
    steps: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    output_dir: str,
    runtime_config: Dict[str, Any] | None,
    decision_calls_by_operation: Dict[str, int],
    decision_token_usage: Dict[str, Any],
    judge_attempt_count: int,
    error: str,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, "rollback_plan.json"), plan)
    repaired_steps = copy.deepcopy(steps)
    repaired_memories = copy.deepcopy(memories)
    _write_jsonl(os.path.join(output_dir, "repaired_tracesteps.jsonl"), repaired_steps)
    _write_jsonl(os.path.join(output_dir, "repaired_memories.jsonl"), repaired_memories)
    summary = _summary_counts(plan, judge_attempt_count)
    summary["repair_failed"] = True
    replay_result = _noop_replay_result(
        task_id=task_id,
        steps=steps,
        decision_calls_by_operation=decision_calls_by_operation,
        decision_token_usage=decision_token_usage,
        repair_failed=True,
        error=error,
        extra_summary=summary,
    )
    payload = _write_repaired_result(
        output_dir=output_dir,
        plan=plan,
        steps=steps,
        memories=memories,
        repaired_steps=repaired_steps,
        repaired_memories=repaired_memories,
        replay_result=replay_result,
        decision_calls_by_operation=decision_calls_by_operation,
        decision_token_usage=decision_token_usage,
        runtime_config=runtime_config,
    )
    return {
        "executor_succeeded": False,
        "executor_error": error,
        "replay_result": replay_result,
        "repaired_trace_steps": repaired_steps,
        "repaired_memories": repaired_memories,
        "payload": payload,
    }
