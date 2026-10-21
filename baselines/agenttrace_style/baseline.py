"""AgentTrace-style root-cause repair baseline."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List

import yaml

from agent.run_agent import AgentRunner
from baselines.agenttrace_style.artifacts import write_json, write_jsonl
from baselines.agenttrace_style.attribution import rank_root_cause_steps
from baselines.common_tools.trace_utils import find_final_turn_boundary, step_ids
from baselines.memaudit_style.oracle_adapter import load_task_record, score_oracle_trace
from methods.ours.run_ours import run_rollback_executor

METHOD = "agenttrace_style"
CONFIG_PATH = "configs/baselines.yaml"
EXTRA_ARTIFACT_FILENAMES = {
    "agenttrace_root_cause_candidates": "agenttrace_root_cause_candidates.jsonl",
    "agenttrace_selected_root_cause": "agenttrace_selected_root_cause.json",
    "agenttrace_causal_graph_summary": "agenttrace_causal_graph_summary.json",
    "agenttrace_replay_policy": "agenttrace_replay_policy.json",
}


def _load_config() -> Dict[str, Any]:
    defaults = {
        "candidate_step_types": ["memory_read", "claim", "plan", "tool_action", "final_answer"],
        "max_candidates": 12,
        "ranking": {
            "backward_relevance_weight": 0.40,
            "affected_membership_weight": 0.25,
            "faulty_memory_signal_weight": 0.20,
            "downstream_reach_weight": 0.10,
            "step_type_prior_weight": 0.05,
        },
        "memory_read_policy": "replay_only",
        "allow_persistent_memory_deletion": False,
        "replay_scope": "fault_source_plus_selected_step_to_trace_end",
        "oracle": {"use_deterministic_oracle": True},
        "fallback": {"mode": "last_final_answer"},
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


def _patch_extra_artifacts(
    output_dir: str,
    *,
    extra_replay_cost: Dict[str, Any],
    extra_summary: Dict[str, Any],
) -> Dict[str, Any]:
    result_path = os.path.join(output_dir, "repaired_result.json")
    with open(result_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    payload.setdefault("artifacts", {}).update(EXTRA_ARTIFACT_FILENAMES)
    payload.setdefault("replay_cost", {}).update(extra_replay_cost)
    payload.setdefault("summary", {}).update(extra_summary)
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return payload


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _replay_suffix_step_ids(steps: List[Dict[str, Any]], selected_step_id: str) -> List[str]:
    ids = step_ids(steps)
    if selected_step_id in ids:
        return ids[ids.index(selected_step_id):]
    return ids[-1:] if ids else []


def _merge_step_ids_in_trace_order(
    steps: List[Dict[str, Any]],
    *groups: List[str],
) -> List[str]:
    wanted = {
        step_id
        for group in groups
        for step_id in group
        if isinstance(step_id, str)
    }
    return [
        step["step_id"]
        for step in steps
        if isinstance(step.get("step_id"), str) and step["step_id"] in wanted
    ]


def _preserve_before_replay(steps: List[Dict[str, Any]], replay_step_ids: List[str]) -> List[str]:
    if not replay_step_ids:
        return step_ids(steps)
    replay_set = set(replay_step_ids)
    first_index = None
    for index, step in enumerate(steps):
        if step.get("step_id") in replay_set:
            first_index = index
            break
    if first_index is None:
        return step_ids(steps)
    return step_ids(steps[:first_index])


def _fault_source_step_ids(
    provenance: Dict[str, Dict[str, Any]],
    steps: List[Dict[str, Any]],
) -> List[str]:
    order = {
        step["step_id"]: index
        for index, step in enumerate(steps)
        if isinstance(step.get("step_id"), str)
    }
    source_ids = []
    for item in provenance.values():
        if not isinstance(item, dict):
            continue
        producing_step_id = item.get("producing_step_id")
        if isinstance(producing_step_id, str) and producing_step_id in order:
            source_ids.append(producing_step_id)
        stale_window = item.get("stale_window")
        if isinstance(stale_window, dict):
            mutation_step_id = stale_window.get("mutation_step_id")
            if isinstance(mutation_step_id, str) and mutation_step_id in order:
                source_ids.append(mutation_step_id)
    return sorted(dict.fromkeys(source_ids), key=lambda step_id: order[step_id])


def _claim_step_ids(steps: List[Dict[str, Any]], step_ids_in_plan: List[str]) -> List[str]:
    replay_set = set(step_ids_in_plan)
    return [
        step["step_id"]
        for step in steps
        if step.get("step_id") in replay_set and step.get("step_type") == "claim"
    ]


def _build_plan(
    *,
    task_id: str,
    faulty_memory_ids: List[str],
    invalidate_claim_ids: List[str],
    preserve_step_ids: List[str],
    replay_step_ids: List[str],
    case_id: str | None,
) -> Dict[str, Any]:
    plan = {
        "method": METHOD,
        "task_id": task_id,
        "faulty_memory_ids": list(faulty_memory_ids),
        "delete_memory_ids": [],
        "quarantine_memory_ids": [],
        "invalidate_claim_ids": list(invalidate_claim_ids),
        "replay_step_ids": list(replay_step_ids),
        "preserve_step_ids": list(preserve_step_ids),
        "redundant_step_ids": [],
        "suspicious_step_ids": [],
        "reason": (
            "AgentTrace-style baseline traced the final wrong answer back to a "
            "ranked root-cause trace step, included faulty provenance source "
            "steps in the replay plan, and replayed downstream steps with the "
            "rollback executor without persistent memory cleanup."
        ),
    }
    if case_id is not None:
        plan["case_id"] = case_id
    return plan


def _write_rollback_plan(output_dir: str, plan: Dict[str, Any]) -> None:
    with open(os.path.join(output_dir, "rollback_plan.json"), "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, ensure_ascii=False)


def run_agenttrace_style(
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
        raise ValueError("agenttrace_style requires session")
    if agent_config is None:
        raise ValueError("agenttrace_style requires agent_config")
    if output_dir is None:
        raise ValueError("agenttrace_style requires output_dir")

    config = _load_config()
    runner = AgentRunner(agent_config, mode="clean")
    runtime_domain = domain or runner.config.get("domain")
    _, old_final_turn_steps = find_final_turn_boundary(steps)

    task = load_task_record(runner.config, task_id)
    original_score = score_oracle_trace(task=task, trace_steps=steps)
    attribution = rank_root_cause_steps(
        task_id=task_id,
        steps=steps,
        memories=memories,
        session=session,
        faulty_memory_ids=faulty_memory_ids,
        final_turn_steps=old_final_turn_steps,
        candidate_step_types=list(config.get("candidate_step_types", [])),
        ranking_weights=dict(config.get("ranking", {})),
        max_candidates=int(config.get("max_candidates", 12) or 12),
        fallback_mode=str(config.get("fallback", {}).get("mode", "last_final_answer")),
    )
    selected = attribution.selected or {
        "step_id": step_ids(old_final_turn_steps)[-1],
        "step_type": "final_answer",
        "score": 0.0,
        "fallback": "last_final_answer",
    }
    selected_step_id = str(selected.get("step_id"))
    selected_step_type = str(selected.get("step_type"))
    fault_source_step_ids = _fault_source_step_ids(attribution.provenance, steps)
    root_cause_replay_step_ids = _replay_suffix_step_ids(steps, selected_step_id)
    replay_step_ids = _merge_step_ids_in_trace_order(
        steps,
        fault_source_step_ids,
        root_cause_replay_step_ids,
    )
    preserve_step_ids = _preserve_before_replay(steps, replay_step_ids)
    invalidate_claim_ids = _claim_step_ids(steps, replay_step_ids)
    replay_start_step_id = replay_step_ids[0] if replay_step_ids else None
    replay_start_adjusted = replay_start_step_id != selected_step_id

    plan = _build_plan(
        task_id=task_id,
        faulty_memory_ids=faulty_memory_ids,
        invalidate_claim_ids=invalidate_claim_ids,
        preserve_step_ids=preserve_step_ids,
        replay_step_ids=replay_step_ids,
        case_id=case_id,
    )

    os.makedirs(output_dir, exist_ok=True)
    _write_rollback_plan(output_dir, plan)
    write_jsonl(
        os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["agenttrace_root_cause_candidates"]),
        attribution.candidates,
    )
    write_json(
        os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["agenttrace_selected_root_cause"]),
        selected,
    )
    write_json(
        os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["agenttrace_causal_graph_summary"]),
        {
            **attribution.causal_graph_summary,
            "non_chain_affected_step_ids": attribution.non_chain_affected_step_ids,
            "backward_reached_step_ids": attribution.backward_reached_step_ids,
            "provenance": attribution.provenance,
            "original_oracle_score": original_score,
            "candidate_scope": "full_trace_backward_reachable_fault_dependent_region",
            "executor": "methods.ours.rollback_executor",
            "uses_rollback_optimizer": False,
        },
    )
    write_json(
        os.path.join(output_dir, EXTRA_ARTIFACT_FILENAMES["agenttrace_replay_policy"]),
        {
            "memory_read_policy": "replay_only",
            "allow_persistent_memory_deletion": False,
            "persistent_memory_cleanup": False,
            "trace_local_filter_enabled": False,
            "trace_local_filtered_memory_ids": [],
            "replay_scope": config.get("replay_scope"),
            "selected_step_id": selected_step_id,
            "selected_step_type": selected_step_type,
            "fault_source_step_ids": fault_source_step_ids,
            "root_cause_replay_step_ids": root_cause_replay_step_ids,
            "replay_start_step_id": replay_start_step_id,
            "replay_start_adjusted": replay_start_adjusted,
            "replay_start_reason": (
                "earliest_fault_source_step"
                if replay_start_step_id in fault_source_step_ids
                and replay_start_step_id != selected_step_id
                else "selected_root_cause_step"
            ),
            "planned_replay_step_ids": replay_step_ids,
            "preserve_step_ids": preserve_step_ids,
            "executor": "methods.ours.rollback_executor",
            "uses_rollback_optimizer": False,
        },
    )

    replay_result = run_rollback_executor(
        attribution.graph,
        plan,
        steps,
        memories,
        session,
        output_dir,
        agent_config_path=agent_config,
        llm_repair_decision_calls_by_operation={},
    )
    repaired_trace_steps = _read_jsonl(os.path.join(output_dir, "repaired_tracesteps.jsonl"))
    repaired_memories = _read_jsonl(os.path.join(output_dir, "repaired_memories.jsonl"))
    repaired_score = score_oracle_trace(task=task, trace_steps=repaired_trace_steps)
    extra_fields = {
        "agenttrace_candidate_step_count": len(attribution.candidates),
        "agenttrace_selected_step_id": selected_step_id,
        "agenttrace_selected_step_type": selected_step_type,
        "agenttrace_selected_step_score": float(selected.get("score", 0.0) or 0.0),
        "agenttrace_fault_source_step_ids": fault_source_step_ids,
        "agenttrace_root_cause_replay_step_count": len(root_cause_replay_step_ids),
        "agenttrace_replay_start_step_id": replay_start_step_id,
        "agenttrace_replay_start_adjusted": replay_start_adjusted,
        "agenttrace_trace_local_filter_enabled": False,
        "agenttrace_persistent_memory_cleanup": False,
        "agenttrace_original_oracle_score": original_score.get("oracle_score"),
        "agenttrace_repaired_oracle_score": repaired_score.get("oracle_score"),
    }
    payload = _patch_extra_artifacts(
        output_dir,
        extra_replay_cost=extra_fields,
        extra_summary=extra_fields,
    )
    print(
        "agenttrace_style: "
        f"candidates={len(attribution.candidates)} "
        f"selected={selected_step_id}:{selected_step_type} "
        f"replay_start={replay_start_step_id} "
        f"llm={replay_result.get('llm_replay_call_count', 0)}"
    )
    return {
        "outputs_written": True,
        "payload": payload,
        "plan": plan,
        "repaired_trace_steps": repaired_trace_steps,
        "repaired_memories": repaired_memories,
        "replay_result": replay_result,
        "runtime_config": {"domain": runtime_domain},
        "cost_overrides": {},
    }
