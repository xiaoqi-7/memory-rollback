"""MemAudit-style memory auditing baseline."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List

import yaml

from agent.run_agent import AgentRunner
from baselines.common_tools.result_writer import write_case_outputs
from baselines.common_tools.trace_utils import find_final_turn_boundary, step_ids
from baselines.full_memory_reset.baseline import _drop_old_final_turn_links
from baselines.memaudit_style.artifacts import write_json, write_jsonl
from baselines.memaudit_style.candidate_selection import select_candidate_memories
from baselines.memaudit_style.consistency_graph import build_consistency_graph
from baselines.memaudit_style.counterfactual import (
    run_counterfactual_audit,
    run_replay_case,
)
from baselines.memaudit_style.oracle_adapter import (
    load_ground_truth_record,
    load_task_record,
    score_oracle_trace,
)
from baselines.memaudit_style.scoring import (
    score_memories,
    select_delete_memory_ids,
)


METHOD = "memaudit_style"
CONFIG_PATH = "configs/baselines.yaml"
EXTRA_ARTIFACT_FILENAMES = {
    "memaudit_audit_report": "memaudit_audit_report.json",
    "counterfactual_results": "counterfactual_results.jsonl",
    "memory_consistency_graph": "memory_consistency_graph.json",
    "memaudit_config_resolved": "memaudit_config_resolved.json",
}


def _load_config() -> Dict[str, Any]:
    defaults = {
        "alpha": 0.5,
        "selection": {"mode": "threshold", "threshold": 0.5},
        "consistency_graph": {"enabled": True},
        "oracle": {"use_deterministic_oracle": True},
    }
    if not os.path.exists(CONFIG_PATH):
        return defaults
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    config = data.get(METHOD, {}) if isinstance(data, dict) else {}
    merged = copy.deepcopy(defaults)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def _token_usage_zero() -> Dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "usage_available_call_count": 0,
        "usage_unavailable_call_count": 0,
        "by_operation": {},
    }


def _merge_token_usage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = _token_usage_zero()
    for usage in rows:
        if not isinstance(usage, dict):
            continue
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "usage_available_call_count",
            "usage_unavailable_call_count",
        ):
            merged[key] += int(usage.get(key, 0) or 0)
        for operation, payload in (usage.get("by_operation") or {}).items():
            target = merged["by_operation"].setdefault(
                operation,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                },
            )
            if isinstance(payload, dict):
                for key in ("input_tokens", "output_tokens", "total_tokens", "call_count"):
                    target[key] += int(payload.get(key, 0) or 0)
    return merged


def _merge_calls_by_operation(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for operation, count in row.items():
            merged[operation] = merged.get(operation, 0) + int(count or 0)
    return merged


def _patch_extra_artifacts(output_dir: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("artifacts", {}).update(EXTRA_ARTIFACT_FILENAMES)
    with open(os.path.join(output_dir, "repaired_result.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return payload


def _zero_cost_overrides() -> Dict[str, Any]:
    return {
        "planned_replay_step_ids": [],
        "replayed_target_step_ids": [],
        "replayed_target_step_count": 0,
        "replayed_tool_action_ids": [],
        "replayed_tool_call_count": 0,
        "synthetic_tool_action_step_ids": [],
        "synthetic_tool_call_count": 0,
        "total_repair_tool_call_count": 0,
        "generated_replacement_step_ids": [],
        "generated_replacement_step_count": 0,
        "llm_replay_call_count": 0,
        "llm_replay_calls_by_operation": {},
        "total_llm_call_count": 0,
        "llm_token_usage": _token_usage_zero(),
        "step_id_remap": {},
        "memaudit_candidate_memory_count": 0,
        "memaudit_counterfactual_replay_count": 0,
        "memaudit_counterfactual_replayed_step_count": 0,
        "memaudit_oracle_judgment_count": 0,
        "memaudit_consistency_pair_count": 0,
        "memaudit_audit_llm_call_count": 0,
        "memaudit_final_repair_llm_call_count": 0,
        "memaudit_counterfactual_tool_call_count": 0,
        "memaudit_final_repair_tool_call_count": 0,
    }


def _build_plan(
    *,
    task_id: str,
    faulty_memory_ids: List[str],
    selected_delete_memory_ids: List[str],
    old_prefix_step_ids: List[str],
    old_final_turn_step_ids: List[str],
    case_id: str | None,
) -> Dict[str, Any]:
    plan = {
        "method": METHOD,
        "task_id": task_id,
        "faulty_memory_ids": list(faulty_memory_ids),
        "delete_memory_ids": list(selected_delete_memory_ids),
        "quarantine_memory_ids": [],
        "invalidate_claim_ids": [],
        "replay_step_ids": list(old_final_turn_step_ids),
        "preserve_step_ids": list(old_prefix_step_ids),
        "redundant_step_ids": [],
        "suspicious_step_ids": [],
        "reason": (
            "Audited final-turn candidate memories with counterfactual deletion, "
            "scored them with oracle recovery and structured consistency signals, "
            "then deleted selected high-detoxification memories and replayed the "
            "final user turn."
        ),
    }
    if case_id is not None:
        plan["case_id"] = case_id
    return plan


def _build_noop_plan(
    *,
    task_id: str,
    faulty_memory_ids: List[str],
    preserved_step_ids: List[str],
    case_id: str | None,
) -> Dict[str, Any]:
    plan = _build_plan(
        task_id=task_id,
        faulty_memory_ids=faulty_memory_ids,
        selected_delete_memory_ids=[],
        old_prefix_step_ids=preserved_step_ids,
        old_final_turn_step_ids=[],
        case_id=case_id,
    )
    plan["reason"] = (
        "MemAudit-style audit found no candidate memories before the final "
        "turn; emitted true no-op output without replay."
    )
    return plan


def run_memaudit_style(
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
        raise ValueError("memaudit_style requires session")
    if agent_config is None:
        raise ValueError("memaudit_style requires agent_config")
    if output_dir is None:
        raise ValueError("memaudit_style requires output_dir")

    config = _load_config()
    runner = AgentRunner(agent_config, mode="clean")
    runtime_domain = domain or runner.config.get("domain")
    if not isinstance(runtime_domain, str) or not runtime_domain:
        raise ValueError("memaudit_style requires domain")

    old_prefix_steps, old_final_turn_steps = find_final_turn_boundary(steps)
    old_final_turn_ids = set(step_ids(old_final_turn_steps))
    old_prefix_steps = _drop_old_final_turn_links(old_prefix_steps, old_final_turn_ids)
    old_prefix_step_ids = step_ids(old_prefix_steps)
    old_final_turn_step_ids = step_ids(old_final_turn_steps)

    task = load_task_record(runner.config, task_id)
    original_oracle_score_details = score_oracle_trace(
        task=task,
        trace_steps=steps,
    )
    original_oracle_score = float(
        original_oracle_score_details.get("oracle_score", 0.0)
    )
    ground_truth_record = load_ground_truth_record(
        config=runner.config,
        case_id=case_id,
        task_id=task_id,
    )
    selection = select_candidate_memories(
        memories=memories,
        steps=steps,
        old_final_turn_steps=old_final_turn_steps,
        faulty_memory_ids=faulty_memory_ids,
    )
    if not selection.candidate_memory_ids:
        graph = build_consistency_graph(
            memories=memories,
            task=task,
            candidate_memory_ids=[],
            ground_truth_record=ground_truth_record,
            enabled=bool(config.get("consistency_graph", {}).get("enabled", True)),
        )
        summary = {
            "replayed": 0,
            "pass_through": len(steps),
            "superseded_skipped": 0,
            "memaudit_noop_reason": "no_candidates_found",
            "memaudit_candidate_memory_count": 0,
            "memaudit_counterfactual_replay_count": 0,
            "memaudit_counterfactual_replayed_step_count": 0,
            "memaudit_oracle_judgment_count": 0,
            "memaudit_consistency_pair_count": 0,
            "memaudit_consistency_graph_edge_count": 0,
            "memaudit_selected_delete_memory_count": 0,
            "memaudit_candidate_memory_ids": [],
            "memaudit_selected_delete_memory_ids": [],
            "memaudit_original_oracle_score": original_oracle_score,
        }
        plan = _build_noop_plan(
            task_id=task_id,
            faulty_memory_ids=faulty_memory_ids,
            preserved_step_ids=step_ids(steps),
            case_id=case_id,
        )
        replay_result = {
            "task_id": task_id,
            "step_id_remap": {},
            "memory_id_remap": {},
            "generated_replacement_step_ids": [],
            "new_memory_ids": [],
            "invalidated_claim_ids": [],
            "llm_replay_call_count": 0,
            "llm_replay_calls_by_operation": {},
            "llm_token_usage": _token_usage_zero(),
            "summary": summary,
        }
        audit_report = {
            "method": METHOD,
            "case_id": case_id,
            "task_id": task_id,
            "candidate_selection": selection.__dict__,
            "scores": {},
            "original_oracle_score": original_oracle_score,
            "original_oracle_score_details": original_oracle_score_details,
            "selected_delete_memory_ids": [],
            "selection_policy": config.get("selection", {}),
            "alpha": config.get("alpha", 0.7),
            "summary": summary,
        }
        write_json(
            os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["memaudit_audit_report"]),
            audit_report,
        )
        write_jsonl(
            os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["counterfactual_results"]),
            [],
        )
        write_json(
            os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["memory_consistency_graph"]),
            graph,
        )
        write_json(
            os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["memaudit_config_resolved"]),
            config,
        )
        payload = write_case_outputs(
            output_dir=output_dir,
            plan=plan,
            original_trace_steps=steps,
            original_memories=memories,
            replay_result=replay_result,
            repaired_trace_steps=steps,
            repaired_memories=memories,
            runtime_config={"domain": runtime_domain},
            cost_overrides=_zero_cost_overrides(),
        )
        payload = _patch_extra_artifacts(output_dir, payload)
        return {
            "outputs_written": True,
            "payload": payload,
            "plan": plan,
            "repaired_trace_steps": steps,
            "repaired_memories": memories,
            "replay_result": replay_result,
            "runtime_config": {"domain": runtime_domain},
            "cost_overrides": _zero_cost_overrides(),
        }

    counterfactual_results = run_counterfactual_audit(
        task_id=task_id,
        domain=runtime_domain,
        task=task,
        session=session,
        original_steps=steps,
        original_memories=memories,
        old_prefix_steps=old_prefix_steps,
        old_final_turn_steps=old_final_turn_steps,
        tools=runner.tools,
        config=runner.config,
        candidate_memory_ids=selection.candidate_memory_ids,
    )
    graph = build_consistency_graph(
        memories=memories,
        task=task,
        candidate_memory_ids=selection.candidate_memory_ids,
        ground_truth_record=ground_truth_record,
        enabled=bool(config.get("consistency_graph", {}).get("enabled", True)),
    )
    scores = score_memories(
        candidate_memory_ids=selection.candidate_memory_ids,
        counterfactual_results=counterfactual_results,
        anomaly_scores=graph["anomaly_scores"],
        alpha=float(config.get("alpha", 0.7)),
    )
    selected_delete_memory_ids = select_delete_memory_ids(
        scores=scores,
        candidate_memory_ids=selection.candidate_memory_ids,
        faulty_memory_ids=faulty_memory_ids,
        threshold=float(config.get("selection", {}).get("threshold", 0.75)),
        mode=str(config.get("selection", {}).get("mode", "threshold")),
    )

    final_replay = run_replay_case(
        task_id=task_id,
        session=session,
        original_steps=steps,
        original_memories=memories,
        old_prefix_steps=old_prefix_steps,
        old_final_turn_steps=old_final_turn_steps,
        tools=runner.tools,
        config=runner.config,
        delete_memory_ids=selected_delete_memory_ids,
        modified_by=METHOD,
    )

    audit_llm_call_count = sum(
        int(row.get("llm_call_count", 0) or 0)
        for row in counterfactual_results
    )
    audit_tool_call_count = sum(
        int(row.get("total_repair_tool_call_count", 0) or 0)
        for row in counterfactual_results
    )
    audit_token_usage = _merge_token_usage(
        [row.get("llm_token_usage", {}) for row in counterfactual_results]
    )
    token_usage = _merge_token_usage([audit_token_usage, final_replay.token_usage])
    llm_calls_by_operation = _merge_calls_by_operation(
        [row.get("llm_calls_by_operation", {}) for row in counterfactual_results]
        + [final_replay.calls_by_operation]
    )

    plan = _build_plan(
        task_id=task_id,
        faulty_memory_ids=faulty_memory_ids,
        selected_delete_memory_ids=selected_delete_memory_ids,
        old_prefix_step_ids=old_prefix_step_ids,
        old_final_turn_step_ids=old_final_turn_step_ids,
        case_id=case_id,
    )
    summary = {
        "replayed": len(old_final_turn_step_ids),
        "pass_through": len(old_prefix_step_ids),
        "superseded_skipped": 0,
        "memaudit_candidate_memory_count": len(selection.candidate_memory_ids),
        "memaudit_counterfactual_replay_count": len(counterfactual_results),
        "memaudit_counterfactual_replayed_step_count": (
            len(counterfactual_results) * len(old_final_turn_step_ids)
        ),
        "memaudit_oracle_judgment_count": len(counterfactual_results),
        "memaudit_consistency_pair_count": graph["summary"]["pair_count"],
        "memaudit_consistency_graph_edge_count": graph["summary"]["edge_count"],
        "memaudit_selected_delete_memory_count": len(selected_delete_memory_ids),
        "memaudit_candidate_memory_ids": list(selection.candidate_memory_ids),
        "memaudit_selected_delete_memory_ids": list(selected_delete_memory_ids),
    }
    replay_result = {
        "task_id": task_id,
        "step_id_remap": final_replay.step_id_remap,
        "memory_id_remap": {},
        "generated_replacement_step_ids": final_replay.new_step_ids,
        "new_memory_ids": final_replay.new_memory_ids,
        "invalidated_claim_ids": [],
        "llm_replay_call_count": audit_llm_call_count + final_replay.llm_call_count,
        "llm_replay_calls_by_operation": llm_calls_by_operation,
        "llm_token_usage": token_usage,
        "summary": summary,
    }
    cost_overrides = {
        "planned_replay_step_ids": old_final_turn_step_ids,
        "replayed_target_step_ids": old_final_turn_step_ids,
        "replayed_target_step_count": len(old_final_turn_step_ids),
        "replayed_tool_action_ids": final_replay.replayed_tool_action_ids,
        "replayed_tool_call_count": len(final_replay.replayed_tool_action_ids),
        "synthetic_tool_action_step_ids": final_replay.synthetic_tool_action_step_ids,
        "synthetic_tool_call_count": len(final_replay.synthetic_tool_action_step_ids),
        "total_repair_tool_call_count": final_replay.total_repair_tool_call_count,
        "generated_replacement_step_ids": final_replay.new_step_ids,
        "generated_replacement_step_count": len(final_replay.new_step_ids),
        "llm_replay_call_count": audit_llm_call_count + final_replay.llm_call_count,
        "llm_replay_calls_by_operation": llm_calls_by_operation,
        "total_llm_call_count": audit_llm_call_count + final_replay.llm_call_count,
        "llm_token_usage": token_usage,
        "step_id_remap": final_replay.step_id_remap,
        "memaudit_candidate_memory_count": len(selection.candidate_memory_ids),
        "memaudit_counterfactual_replay_count": len(counterfactual_results),
        "memaudit_counterfactual_replayed_step_count": (
            len(counterfactual_results) * len(old_final_turn_step_ids)
        ),
        "memaudit_oracle_judgment_count": len(counterfactual_results),
        "memaudit_consistency_pair_count": graph["summary"]["pair_count"],
        "memaudit_audit_llm_call_count": audit_llm_call_count,
        "memaudit_final_repair_llm_call_count": final_replay.llm_call_count,
        "memaudit_counterfactual_tool_call_count": audit_tool_call_count,
        "memaudit_final_repair_tool_call_count": (
            final_replay.total_repair_tool_call_count
        ),
        "memaudit_original_oracle_score": original_oracle_score,
    }

    audit_report = {
        "method": METHOD,
        "case_id": case_id,
        "task_id": task_id,
        "candidate_selection": selection.__dict__,
        "scores": scores,
        "original_oracle_score": original_oracle_score,
        "original_oracle_score_details": original_oracle_score_details,
        "selected_delete_memory_ids": selected_delete_memory_ids,
        "selection_policy": config.get("selection", {}),
        "alpha": config.get("alpha", 0.7),
        "summary": summary,
    }
    write_json(
        os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["memaudit_audit_report"]),
        audit_report,
    )
    write_jsonl(
        os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["counterfactual_results"]),
        counterfactual_results,
    )
    write_json(
        os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["memory_consistency_graph"]),
        graph,
    )
    write_json(
        os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["memaudit_config_resolved"]),
        config,
    )

    print(
        "memaudit_style: "
        f"candidates={len(selection.candidate_memory_ids)} "
        f"selected_delete={len(selected_delete_memory_ids)} "
        f"audit_llm={audit_llm_call_count} "
        f"final_llm={final_replay.llm_call_count}"
    )

    payload = write_case_outputs(
        output_dir=output_dir,
        plan=plan,
        original_trace_steps=steps,
        original_memories=memories,
        replay_result=replay_result,
        repaired_trace_steps=final_replay.repaired_trace_steps,
        repaired_memories=final_replay.repaired_memories,
        runtime_config={"domain": runtime_domain},
        cost_overrides=cost_overrides,
    )
    payload = _patch_extra_artifacts(output_dir, payload)

    return {
        "outputs_written": True,
        "payload": payload,
        "plan": plan,
        "repaired_trace_steps": final_replay.repaired_trace_steps,
        "repaired_memories": final_replay.repaired_memories,
        "replay_result": replay_result,
        "runtime_config": {"domain": runtime_domain},
        "cost_overrides": cost_overrides,
    }
