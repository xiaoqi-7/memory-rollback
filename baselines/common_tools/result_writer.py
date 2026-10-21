"""Shared evaluator-facing output helpers for baselines."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

from methods.ours.rollback_executor import build_repaired_result_payload


DEFAULT_BASELINE_OUTPUT_TEMPLATE = (
    "outputs/baselines/{method}/{domain}/{fault_types}/{task_id}/{faulty_memory_ids}"
)

DEFAULT_ARTIFACT_FILENAMES = {
    "repaired_result": "repaired_result.json",
    "repaired_memories": "repaired_memories.jsonl",
    "repaired_tracesteps": "repaired_tracesteps.jsonl",
}


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, ensure_ascii=False)
            fh.write("\n")


def build_repaired_result_only_payload(
    *,
    plan: Dict[str, Any],
    original_trace_steps: List[Dict[str, Any]],
    original_memories: List[Dict[str, Any]],
    replay_result: Dict[str, Any],
    repaired_trace_steps: Optional[List[Dict[str, Any]]] = None,
    repaired_memories: Optional[List[Dict[str, Any]]] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    cost_overrides: Optional[Dict[str, Any]] = None,
    artifact_filenames: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build repaired_result.json payload for baseline case artifacts."""
    filenames = dict(DEFAULT_ARTIFACT_FILENAMES)
    if artifact_filenames:
        filenames.update(artifact_filenames)
    payload = build_repaired_result_payload(
        plan,
        original_trace_steps,
        original_memories,
        copy.deepcopy(
            repaired_trace_steps
            if repaired_trace_steps is not None
            else original_trace_steps
        ),
        copy.deepcopy(
            repaired_memories if repaired_memories is not None else original_memories
        ),
        replay_result,
        llm_repair_decision_calls_by_operation={},
        llm_repair_decision_token_usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "usage_available_call_count": 0,
            "usage_unavailable_call_count": 0,
            "by_operation": {},
        },
        runtime_config=runtime_config,
        artifact_filenames=filenames,
    )
    if cost_overrides:
        payload["replay_cost"].update(cost_overrides)
    internal_summary = replay_result.get("summary", {})
    if isinstance(internal_summary, dict):
        for key, value in internal_summary.items():
            if key not in payload.get("summary", {}):
                payload["summary"][key] = value
    json.dumps(payload, ensure_ascii=False)
    payload["artifacts"] = filenames
    return payload


def write_case_outputs(
    *,
    output_dir: str,
    plan: Dict[str, Any],
    original_trace_steps: List[Dict[str, Any]],
    original_memories: List[Dict[str, Any]],
    replay_result: Dict[str, Any],
    repaired_trace_steps: Optional[List[Dict[str, Any]]] = None,
    repaired_memories: Optional[List[Dict[str, Any]]] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    cost_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write repaired_result.json plus evaluator sidecar JSONL artifacts."""
    final_trace_steps = (
        repaired_trace_steps if repaired_trace_steps is not None else original_trace_steps
    )
    final_memories = repaired_memories if repaired_memories is not None else original_memories
    payload = build_repaired_result_only_payload(
        plan=plan,
        original_trace_steps=original_trace_steps,
        original_memories=original_memories,
        replay_result=replay_result,
        repaired_trace_steps=final_trace_steps,
        repaired_memories=final_memories,
        runtime_config=runtime_config,
        cost_overrides=cost_overrides,
        artifact_filenames=DEFAULT_ARTIFACT_FILENAMES,
    )
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, DEFAULT_ARTIFACT_FILENAMES["repaired_result"])
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    _write_jsonl(
        os.path.join(output_dir, DEFAULT_ARTIFACT_FILENAMES["repaired_memories"]),
        copy.deepcopy(final_memories),
    )
    _write_jsonl(
        os.path.join(output_dir, DEFAULT_ARTIFACT_FILENAMES["repaired_tracesteps"]),
        copy.deepcopy(final_trace_steps),
    )
    return payload


def write_repaired_result_only(
    *,
    output_dir: str,
    plan: Dict[str, Any],
    original_trace_steps: List[Dict[str, Any]],
    original_memories: List[Dict[str, Any]],
    replay_result: Dict[str, Any],
    repaired_trace_steps: Optional[List[Dict[str, Any]]] = None,
    repaired_memories: Optional[List[Dict[str, Any]]] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    cost_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper; writes the full baseline case artifact set."""
    return write_case_outputs(
        output_dir=output_dir,
        plan=plan,
        original_trace_steps=original_trace_steps,
        original_memories=original_memories,
        replay_result=replay_result,
        repaired_trace_steps=repaired_trace_steps,
        repaired_memories=repaired_memories,
        runtime_config=runtime_config,
        cost_overrides=cost_overrides,
    )
