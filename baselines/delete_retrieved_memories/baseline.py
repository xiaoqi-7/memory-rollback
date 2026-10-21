"""Delete-retrieved-memories baseline."""

from __future__ import annotations

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
from baselines.delete_retrieved_memories.subgraph_memory import (
    extract_memory_ids_from_subgraph_steps,
    filter_non_chain_affected_steps,
)
from baselines.delete_retrieved_memories.trace_active_state import (
    compute_active_memory_state_before_last_user_turn,
)
from baselines.full_memory_reset.baseline import _drop_old_final_turn_links
from baselines.full_memory_reset.lifecycle import run_full_memory_reset_lifecycle
from methods.ours.affected_subgraph import compute_affected_subgraph
from methods.ours.build_dependency_graph import build_dependency_graph
from methods.ours.fault_provenance import (
    find_producing_step,
    find_stale_window,
    walk_ancestor_chain,
)


METHOD = "delete_retrieved_memories"
_CANDIDATE_EXCLUDED_STEP_TYPES = ("memory_read",)


def _fault_provenance(
    graph: Any,
    faulty_memory_ids: List[str],
    steps: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    provenance: Dict[str, Dict[str, Any]] = {}
    for memory_id in faulty_memory_ids:
        try:
            provenance[memory_id] = {
                "producing_step_id": find_producing_step(graph, memory_id),
                "stale_window": find_stale_window(graph, memory_id, steps),
                "ancestor_chain": walk_ancestor_chain(graph, memory_id),
            }
        except KeyError:
            print(
                f"  ! warning: faulty memory {memory_id!r} is not in dependency "
                "graph; skipping it for affected-subgraph seeding"
            )
    return provenance


def _affected_steps(
    graph: Any,
    steps: List[Dict[str, Any]],
    affected_nodes: Dict[str, Any],
) -> List[Dict[str, Any]]:
    affected_ids = set(affected_nodes.get("affected_node_ids", []))
    step_by_id = {
        step.get("step_id"): step
        for step in steps
        if isinstance(step.get("step_id"), str)
    }
    return [
        step_by_id[node_id]
        for node_id in affected_nodes.get("affected_node_ids", [])
        if node_id in step_by_id
        and getattr(graph.node(node_id), "node_kind", None) == "step"
        and node_id in affected_ids
    ]


def _build_plan(
    *,
    task_id: str,
    faulty_memory_ids: List[str],
    planned_delete_memory_ids: List[str],
    old_prefix_step_ids: List[str],
    old_final_turn_step_ids: List[str],
    case_id: str | None,
) -> Dict[str, Any]:
    plan = {
        "method": METHOD,
        "task_id": task_id,
        "faulty_memory_ids": list(faulty_memory_ids),
        "delete_memory_ids": list(planned_delete_memory_ids),
        "quarantine_memory_ids": [],
        "invalidate_claim_ids": [],
        "replay_step_ids": list(old_final_turn_step_ids),
        "preserve_step_ids": list(old_prefix_step_ids),
        "redundant_step_ids": [],
        "suspicious_step_ids": [],
        "reason": (
            "Deleted active memories retrieved or referenced inside the affected "
            "subgraph, then reran the final user turn through the baseline "
            "lifecycle."
        ),
    }
    if case_id is not None:
        plan["case_id"] = case_id
    return plan


def run_delete_retrieved_memories(
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
        raise ValueError("delete_retrieved_memories requires session")
    if agent_config is None:
        raise ValueError("delete_retrieved_memories requires agent_config")

    runner = AgentRunner(agent_config, mode="clean")
    old_prefix_steps, old_final_turn_steps = find_final_turn_boundary(steps)
    old_final_turn_ids = set(step_ids(old_final_turn_steps))
    old_prefix_steps = _drop_old_final_turn_links(old_prefix_steps, old_final_turn_ids)
    old_prefix_step_ids = step_ids(old_prefix_steps)
    old_final_turn_step_ids = step_ids(old_final_turn_steps)

    graph = build_dependency_graph(task_id, steps, memories, session)
    provenance = _fault_provenance(graph, faulty_memory_ids, steps)
    affected_nodes = compute_affected_subgraph(
        graph,
        faulty_memory_ids,
        provenance,
        steps,
    )
    affected_steps = _affected_steps(graph, steps, affected_nodes)
    non_chain_affected_steps = filter_non_chain_affected_steps(
        graph,
        affected_nodes,
        steps,
        faulty_memory_ids=faulty_memory_ids,
        provenance=provenance,
    )
    excluded_memory_read_step_count = sum(
        1
        for step in non_chain_affected_steps
        if step.get("step_type") in _CANDIDATE_EXCLUDED_STEP_TYPES
    )
    candidate_source_steps = [
        step
        for step in non_chain_affected_steps
        if step.get("step_type") not in _CANDIDATE_EXCLUDED_STEP_TYPES
    ]
    candidate_source_step_ids = step_ids(candidate_source_steps)
    retrieved_candidates = extract_memory_ids_from_subgraph_steps(
        non_chain_affected_steps,
        exclude_step_types=_CANDIDATE_EXCLUDED_STEP_TYPES,
    )
    active_state = compute_active_memory_state_before_last_user_turn(memories, steps)
    active_before_last = active_state.active_memory_ids
    active_before_last_set = set(active_before_last)
    retrieved_candidate_set = set(retrieved_candidates)
    planned_delete_memory_ids = [
        memory_id for memory_id in active_before_last
        if memory_id in retrieved_candidate_set
    ]

    lifecycle_result = run_full_memory_reset_lifecycle(
        task_id=task_id,
        session=session,
        original_memories=memories,
        old_prefix_steps=old_prefix_steps,
        tools=runner.tools,
        config=runner.config,
        delete_memory_ids=planned_delete_memory_ids,
        modified_by=METHOD,
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
    step_id_remap = align_step_id_remap_by_type(
        old_final_turn_steps,
        renamed_new_steps,
    )
    tool_classification = classify_tool_calls(old_final_turn_steps, renamed_new_steps)

    plan = _build_plan(
        task_id=task_id,
        faulty_memory_ids=faulty_memory_ids,
        planned_delete_memory_ids=planned_delete_memory_ids,
        old_prefix_step_ids=old_prefix_step_ids,
        old_final_turn_step_ids=old_final_turn_step_ids,
        case_id=case_id,
    )
    summary = {
        "replayed": len(old_final_turn_step_ids),
        "pass_through": len(old_prefix_step_ids),
        "superseded_skipped": 0,
        "affected_subgraph_step_count": len(affected_steps),
        "non_chain_affected_step_count": len(non_chain_affected_steps),
        "excluded_memory_read_step_count": excluded_memory_read_step_count,
        "candidate_step_count": len(candidate_source_steps),
        "candidate_source_step_ids": list(candidate_source_step_ids),
        "retrieved_memory_candidate_count": len(retrieved_candidates),
        "retrieved_memory_candidate_ids": list(retrieved_candidates),
        "active_memory_ids_before_last_user_turn": list(active_before_last),
        "planned_delete_memory_ids": list(planned_delete_memory_ids),
        "active_memory_before_last_user_turn_count": len(active_before_last),
        "planned_delete_memory_count": len(planned_delete_memory_ids),
        "unrecognized_memory_lifecycle_event_count": (
            active_state.unrecognized_memory_lifecycle_event_count
        ),
    }
    replay_result = {
        "task_id": task_id,
        "step_id_remap": step_id_remap,
        "memory_id_remap": {},
        "generated_replacement_step_ids": new_step_ids,
        "new_memory_ids": new_memory_ids,
        "invalidated_claim_ids": [],
        "llm_replay_call_count": lifecycle_result.tracker.llm_call_count,
        "llm_replay_calls_by_operation": (
            lifecycle_result.tracker.calls_by_operation_payload()
        ),
        "llm_token_usage": lifecycle_result.tracker.token_usage_payload(),
        "summary": summary,
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
        "generated_replacement_step_ids": new_step_ids,
        "generated_replacement_step_count": len(new_step_ids),
        "llm_replay_call_count": lifecycle_result.tracker.llm_call_count,
        "llm_replay_calls_by_operation": (
            lifecycle_result.tracker.calls_by_operation_payload()
        ),
        "total_llm_call_count": lifecycle_result.tracker.llm_call_count,
        "llm_token_usage": lifecycle_result.tracker.token_usage_payload(),
        "step_id_remap": step_id_remap,
    }

    print(
        "delete_retrieved_memories: "
        f"affected_steps={len(affected_steps)} "
        f"non_chain_affected_steps={len(non_chain_affected_steps)} "
        f"excluded_memory_read_steps={excluded_memory_read_step_count} "
        f"candidate_source_steps={len(candidate_source_steps)} "
        f"candidate_memories={len(retrieved_candidates)} "
        f"active_before_last={len(active_before_last_set)} "
        f"planned_delete={len(planned_delete_memory_ids)} "
        "unrecognized_lifecycle_events="
        f"{active_state.unrecognized_memory_lifecycle_event_count}"
    )

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
    return run_delete_retrieved_memories(
        task["task_id"],
        faulty_trace,
        memory_store,
        task.get("faulty_memory_ids", []),
        case_id=task.get("case_id"),
        domain=task.get("domain"),
        session=task.get("session"),
        agent_config=task.get("agent_config"),
    )
