"""Root-cause tracing for AgentTrace-style baseline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Set

from baselines.common_tools.trace_utils import step_ids
from baselines.delete_retrieved_memories.subgraph_memory import filter_non_chain_affected_steps
from methods.ours.affected_subgraph import compute_affected_subgraph
from methods.ours.build_dependency_graph import build_dependency_graph
from methods.ours.fault_provenance import (
    find_producing_step,
    find_stale_window,
    walk_ancestor_chain,
)

_CHAIN_EDGE_TYPE = "chain"
_DEFAULT_STEP_PRIOR = {
    "memory_read": 0.95,
    "claim": 0.85,
    "plan": 0.75,
    "tool_action": 0.65,
    "final_answer": 0.40,
}


@dataclass
class AttributionResult:
    graph: Any
    provenance: Dict[str, Dict[str, Any]]
    affected_nodes: Dict[str, Any]
    non_chain_affected_step_ids: List[str]
    backward_reached_step_ids: List[str]
    candidates: List[Dict[str, Any]]
    selected: Dict[str, Any]
    causal_graph_summary: Dict[str, Any]


def fault_provenance(graph: Any, faulty_memory_ids: Sequence[str], steps: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for memory_id in faulty_memory_ids:
        try:
            out[memory_id] = {
                "producing_step_id": find_producing_step(graph, memory_id),
                "stale_window": find_stale_window(graph, memory_id, list(steps)),
                "ancestor_chain": walk_ancestor_chain(graph, memory_id),
            }
        except KeyError:
            out[memory_id] = {
                "producing_step_id": None,
                "stale_window": None,
                "ancestor_chain": [],
                "missing_from_graph": True,
            }
    return out


def _non_chain_edge(labels: Any) -> bool:
    return bool(set(labels or []) - {_CHAIN_EDGE_TYPE})


def _backward_trace(graph: Any, final_step_id: str) -> Dict[str, int]:
    distances: Dict[str, int] = {final_step_id: 0}
    queue: deque[str] = deque([final_step_id])
    while queue:
        node_id = queue.popleft()
        for predecessor, labels in graph.predecessor_edges(node_id).items():
            if predecessor in distances:
                continue
            # AgentTrace follows the execution causal graph. Keep chain edges for
            # backward localization, but score non-chain dependency separately.
            distances[predecessor] = distances[node_id] + 1
            queue.append(predecessor)
    return distances


def _sequential_suffix_ids(steps: Sequence[Mapping[str, Any]], seed_ids: Set[str], final_step_id: str) -> Set[str]:
    index_by_id = {step.get("step_id"): idx for idx, step in enumerate(steps)}
    final_idx = index_by_id.get(final_step_id)
    if final_idx is None:
        return set()
    seed_indices = [index_by_id[sid] for sid in seed_ids if sid in index_by_id and index_by_id[sid] <= final_idx]
    if not seed_indices:
        return {final_step_id}
    start_idx = min(seed_indices)
    return {
        str(step.get("step_id"))
        for step in steps[start_idx : final_idx + 1]
        if isinstance(step.get("step_id"), str)
    }


def _faulty_memory_signal(step: Mapping[str, Any], faulty_memory_ids: Set[str], provenance: Mapping[str, Any]) -> float:
    step_id = step.get("step_id")
    used = set(str(item) for item in step.get("used_ids", []) or [])
    invalidated = set(str(item) for item in step.get("invalidated_memory_ids", []) or [])
    generated = set(str(item) for item in step.get("generated_memory_ids", []) or [])
    if used & faulty_memory_ids or invalidated & faulty_memory_ids or generated & faulty_memory_ids:
        return 1.0
    for item in provenance.values():
        if not isinstance(item, Mapping):
            continue
        if item.get("producing_step_id") == step_id:
            return 1.0
        stale = item.get("stale_window")
        if isinstance(stale, Mapping) and stale.get("mutation_step_id") == step_id:
            return 1.0
    return 0.0


def _downstream_reach(graph: Any, step_id: str, final_step_id: str) -> float:
    if step_id == final_step_id:
        return 0.0
    visited = {step_id}
    queue: deque[str] = deque([step_id])
    reached_steps = 0
    reaches_final = False
    while queue:
        node_id = queue.popleft()
        for nxt, labels in graph.successor_edges(node_id).items():
            if nxt in visited:
                continue
            visited.add(nxt)
            node = graph.node(nxt)
            if getattr(node, "node_kind", None) == "step":
                reached_steps += 1
            if nxt == final_step_id:
                reaches_final = True
            queue.append(nxt)
    base = min(1.0, reached_steps / 8.0)
    return max(base, 0.5 if reaches_final else 0.0)


def rank_root_cause_steps(
    *,
    task_id: str,
    steps: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    session: List[str],
    faulty_memory_ids: List[str],
    final_turn_steps: List[Dict[str, Any]],
    candidate_step_types: List[str],
    ranking_weights: Dict[str, float],
    max_candidates: int,
    fallback_mode: str,
) -> AttributionResult:
    graph = build_dependency_graph(task_id, steps, memories, session)
    provenance = fault_provenance(graph, faulty_memory_ids, steps)
    affected_nodes = compute_affected_subgraph(graph, faulty_memory_ids, provenance, steps)
    non_chain_steps = filter_non_chain_affected_steps(
        graph,
        affected_nodes,
        steps,
        faulty_memory_ids=faulty_memory_ids,
        provenance=provenance,
    )
    non_chain_affected_step_ids = step_ids(list(non_chain_steps))
    affected_ids = {
        node_id for node_id in affected_nodes.get("affected_node_ids", [])
        if isinstance(node_id, str)
    }
    final_step = next((step for step in reversed(steps) if step.get("step_type") == "final_answer"), None)
    if final_step is None or not isinstance(final_step.get("step_id"), str):
        raise ValueError("trace has no final_answer step")
    final_step_id = final_step["step_id"]
    backward_distances = _backward_trace(graph, final_step_id)
    backward_step_ids = [
        node_id for node_id in backward_distances
        if getattr(graph.node(node_id), "node_kind", None) == "step"
    ]
    suffix_seed_ids = set(non_chain_affected_step_ids) | {
        item.get("producing_step_id")
        for item in provenance.values()
        if isinstance(item, Mapping) and isinstance(item.get("producing_step_id"), str)
    }
    suffix_ids = _sequential_suffix_ids(steps, {sid for sid in suffix_seed_ids if isinstance(sid, str)}, final_step_id)
    final_turn_ids = set(step_ids(final_turn_steps))
    faulty_set = set(faulty_memory_ids)
    candidate_types = set(candidate_step_types)
    max_distance = max(backward_distances.values() or [1])
    weights = {
        "backward_relevance": float(ranking_weights.get("backward_relevance_weight", 0.40)),
        "affected_membership": float(ranking_weights.get("affected_membership_weight", 0.25)),
        "faulty_memory_signal": float(ranking_weights.get("faulty_memory_signal_weight", 0.20)),
        "downstream_reach": float(ranking_weights.get("downstream_reach_weight", 0.10)),
        "step_type_prior": float(ranking_weights.get("step_type_prior_weight", 0.05)),
    }
    candidates: List[Dict[str, Any]] = []
    for step in steps:
        step_id = step.get("step_id")
        step_type = step.get("step_type")
        if not isinstance(step_id, str) or step_type not in candidate_types:
            continue
        if step_id not in backward_distances:
            continue
        in_affected = step_id in affected_ids or step_id in non_chain_affected_step_ids
        fault_signal = _faulty_memory_signal(step, faulty_set, provenance)
        in_suffix = step_id in suffix_ids
        if not (in_affected or fault_signal > 0 or in_suffix or step_id == final_step_id):
            continue
        backward = 1.0 - (backward_distances[step_id] / max(1, max_distance))
        downstream = _downstream_reach(graph, step_id, final_step_id)
        prior = _DEFAULT_STEP_PRIOR.get(str(step_type), 0.5)
        components = {
            "backward_relevance": backward,
            "affected_membership": 1.0 if in_affected else 0.0,
            "faulty_memory_signal": fault_signal,
            "downstream_reach": downstream,
            "step_type_prior": prior,
        }
        score = sum(components[name] * weights[name] for name in components)
        candidates.append({
            "step_id": step_id,
            "step_type": step_type,
            "turn": step.get("turn"),
            "content": step.get("content"),
            "score": score,
            "components": components,
            "weights": weights,
            "backward_distance_to_error": backward_distances[step_id],
            "in_final_turn": step_id in final_turn_ids,
            "in_affected_subgraph": in_affected,
            "in_fault_dependent_suffix": in_suffix,
            "faulty_memory_signal": fault_signal,
        })
    candidates.sort(key=lambda row: (row["score"], row["faulty_memory_signal"], row["components"]["affected_membership"]), reverse=True)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    if candidates:
        selected = dict(candidates[0])
    elif fallback_mode == "last_final_answer":
        selected = {
            "step_id": final_step_id,
            "step_type": "final_answer",
            "turn": final_step.get("turn"),
            "content": final_step.get("content"),
            "score": 0.0,
            "components": {},
            "weights": weights,
            "fallback": "last_final_answer",
        }
    else:
        selected = {}
    summary = {
        "node_count": len(graph.nodes),
        "affected_node_count": len(affected_ids),
        "non_chain_affected_step_count": len(non_chain_affected_step_ids),
        "backward_reached_step_count": len(backward_step_ids),
        "candidate_step_count": len(candidates),
        "candidate_scope": "full_trace_backward_reachable_fault_dependent_region",
        "final_answer_step_id": final_step_id,
        "final_turn_step_ids": list(final_turn_ids),
        "fault_dependent_suffix_step_count": len(suffix_ids),
    }
    return AttributionResult(
        graph=graph,
        provenance=provenance,
        affected_nodes=affected_nodes,
        non_chain_affected_step_ids=non_chain_affected_step_ids,
        backward_reached_step_ids=backward_step_ids,
        candidates=candidates,
        selected=selected,
        causal_graph_summary=summary,
    )
