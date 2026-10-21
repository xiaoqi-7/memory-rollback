"""LLM-judge rollback-plan baseline."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import yaml

from baselines.llm_judge_repair.executor_adapter import (
    execute_llm_judge_plan,
    write_failed_judge_outputs,
)
from baselines.llm_judge_repair.judge import generate_llm_judge_plan
from baselines.llm_judge_repair.plan_schema import METHOD


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _load_agent_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _noop_plan(
    *,
    task_id: str,
    case_id: str | None,
    faulty_memory_ids: List[str],
    steps: List[Dict[str, Any]],
    reason: str,
) -> Dict[str, Any]:
    plan = {
        "method": METHOD,
        "task_id": task_id,
        "faulty_memory_ids": list(faulty_memory_ids),
        "delete_memory_ids": [],
        "quarantine_memory_ids": [],
        "invalidate_claim_ids": [],
        "replay_step_ids": [],
        "preserve_step_ids": [
            step["step_id"]
            for step in steps
            if isinstance(step.get("step_id"), str)
        ],
        "redundant_step_ids": [],
        "suspicious_step_ids": [],
        "reason": reason,
    }
    if case_id is not None:
        plan["case_id"] = case_id
    return plan


def run_llm_judge_repair(
    task_id: str,
    steps: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    faulty_memory_ids: List[str],
    case_id: str | None = None,
    *,
    domain: str | None = None,
    session: List[str] | None = None,
    agent_config: str | None = None,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    if session is None:
        raise ValueError("llm_judge_repair requires session")
    if agent_config is None:
        raise ValueError("llm_judge_repair requires agent_config")
    if output_dir is None:
        raise ValueError("llm_judge_repair requires output_dir")

    config = _load_agent_config(agent_config)
    runtime_config = {"domain": domain or config.get("domain")}
    judge_result = generate_llm_judge_plan(
        task_id=task_id,
        case_id=case_id or "",
        faulty_memory_ids=faulty_memory_ids,
        memories=memories,
        steps=steps,
        session=session,
        model=config.get("model"),
    )
    tracker = judge_result["tracker"]
    decision_calls = tracker.calls_by_operation_payload()
    decision_usage = tracker.token_usage_payload()
    attempts = judge_result["raw_response_artifact"].get("attempts", [])

    os.makedirs(output_dir, exist_ok=True)
    _write_json(
        os.path.join(output_dir, "llm_judge_prompt.json"),
        judge_result["prompt_artifact"],
    )
    _write_json(
        os.path.join(output_dir, "llm_judge_raw_response.json"),
        judge_result["raw_response_artifact"],
    )

    if judge_result.get("repair_failed"):
        plan = _noop_plan(
            task_id=task_id,
            case_id=case_id,
            faulty_memory_ids=faulty_memory_ids,
            steps=steps,
            reason=(
                "LLM judge failed to produce a valid rollback plan; "
                "no-op plan emitted for evaluator-facing output."
            ),
        )
        normalized_artifact = {
            "repair_failed": True,
            "error": judge_result.get("error"),
            "normalized_plan": plan,
        }
        _write_json(
            os.path.join(output_dir, "llm_judge_normalized_plan.json"),
            normalized_artifact,
        )
        result = write_failed_judge_outputs(
            task_id=task_id,
            plan=plan,
            steps=steps,
            memories=memories,
            output_dir=output_dir,
            runtime_config=runtime_config,
            decision_calls_by_operation=decision_calls,
            decision_token_usage=decision_usage,
            judge_attempt_count=len(attempts),
            error=str(judge_result.get("error") or "judge failed"),
        )
    else:
        plan = judge_result["plan"]
        _write_json(
            os.path.join(output_dir, "llm_judge_normalized_plan.json"),
            {"repair_failed": False, "normalized_plan": plan},
        )
        result = execute_llm_judge_plan(
            task_id=task_id,
            plan=plan,
            steps=steps,
            memories=memories,
            session=session,
            output_dir=output_dir,
            agent_config=agent_config,
            runtime_config=runtime_config,
            decision_calls_by_operation=decision_calls,
            decision_token_usage=decision_usage,
            judge_attempt_count=len(attempts),
        )

    return {
        "outputs_written": True,
        "payload": result["payload"],
        "plan": plan,
        "replay_result": result["replay_result"],
        "repaired_trace_steps": result["repaired_trace_steps"],
        "repaired_memories": result["repaired_memories"],
        "runtime_config": runtime_config,
        "executor_succeeded": result.get("executor_succeeded", False),
        "executor_error": result.get("executor_error"),
    }


def run(
    task: Dict[str, Any],
    faulty_trace: List[Dict[str, Any]],
    memory_store: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return run_llm_judge_repair(
        task["task_id"],
        faulty_trace,
        memory_store,
        task.get("faulty_memory_ids", []),
        case_id=task.get("case_id"),
        domain=task.get("domain"),
        session=task.get("session"),
        agent_config=task.get("agent_config"),
        output_dir=task.get("output_dir"),
    )
