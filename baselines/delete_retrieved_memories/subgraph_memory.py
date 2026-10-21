"""Affected-subgraph memory-id extraction for delete-retrieved baseline."""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Collection, Mapping, Sequence


_MEMORY_ID_RE = re.compile(r"^m_(?:f|n)?\d+$")
_MEMORY_KEY_PARTS = (
    "memory",
    "memories",
    "memory_id",
    "memory_ids",
    "retrieved",
    "retrieval",
    "context",
    "source_memory",
    "target_memory",
    "used_ids",
)
_CHAIN_EDGE_TYPE = "chain"


def _is_memory_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.lower()
    return any(part in normalized for part in _MEMORY_KEY_PARTS)


def _collect_from_value(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        if _MEMORY_ID_RE.match(value):
            out.append(value)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_memory_key(key):
                _collect_from_value(child, out)
            elif isinstance(child, (Mapping, list, tuple)):
                _collect_from_relevant_containers(child, out)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _collect_from_value(child, out)


def _collect_from_relevant_containers(value: Any, out: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_memory_key(key):
                _collect_from_value(child, out)
            elif isinstance(child, (Mapping, list, tuple)):
                _collect_from_relevant_containers(child, out)
    elif isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, (Mapping, list, tuple)):
                _collect_from_relevant_containers(child, out)


def extract_memory_ids_from_subgraph_steps(
    steps: Sequence[Mapping[str, Any]],
    *,
    exclude_step_types: Collection[str] = ("memory_read",),
) -> list[str]:
    """Return memory ids from memory/retrieval/context fields in step records.

    ``memory_read`` records are excluded by default because their fields describe
    retrieval context returned to the agent, not memories actually relied on by
    downstream reasoning, answers, or memory operations.
    """
    excluded = set(exclude_step_types)
    found: list[str] = []
    for step in steps:
        if step.get("step_type") in excluded:
            continue
        _collect_from_relevant_containers(step, found)
    return list(dict.fromkeys(found))


def _resolved_seed_causes(affected_nodes: Mapping[str, Any]) -> list[str]:
    causes: list[str] = []
    for resolved in affected_nodes.get("resolved_seed_ids", []) or []:
        if isinstance(resolved, str):
            causes.append(resolved)
        elif isinstance(resolved, Mapping):
            memory_id = resolved.get("memory_id")
            if isinstance(memory_id, str):
                causes.append(memory_id)
    return list(dict.fromkeys(causes))


def _non_chain_successors(graph: Any, node_id: str) -> list[str]:
    if not hasattr(graph, "successor_edges"):
        return [
            dst for dst in sorted(graph.successors(node_id))
            if _CHAIN_EDGE_TYPE not in graph.edge_type(node_id, dst)
        ]
    return [
        dst for dst, labels in sorted(graph.successor_edges(node_id).items())
        if set(labels) - {_CHAIN_EDGE_TYPE}
    ]


def _non_chain_descendants(graph: Any, seeds: Sequence[str]) -> set[str]:
    visited = set(seeds)
    reached: set[str] = set()
    queue: deque[str] = deque(seeds)
    while queue:
        node_id = queue.popleft()
        if graph.node(node_id) is None:
            continue
        for nxt in _non_chain_successors(graph, node_id):
            if nxt not in visited:
                visited.add(nxt)
                reached.add(nxt)
                queue.append(nxt)
    return reached


def _is_step_node(graph: Any, node_id: str) -> bool:
    node = graph.node(node_id)
    return getattr(node, "node_kind", None) == "step"


def _source_fault_related_step_ids(
    graph: Any,
    faulty_memory_ids: Sequence[str],
    provenance: Mapping[str, Any],
    affected_step_ids: set[str],
) -> set[str]:
    related: set[str] = set()
    for memory_id in faulty_memory_ids:
        item = provenance.get(memory_id, {})
        if isinstance(item, Mapping):
            producing_step_id = item.get("producing_step_id")
            if isinstance(producing_step_id, str) and _is_step_node(
                graph, producing_step_id
            ):
                related.add(producing_step_id)

        if graph.node(memory_id) is None:
            continue
        for src, labels in graph.predecessor_edges(memory_id).items():
            if set(labels) - {_CHAIN_EDGE_TYPE} and _is_step_node(graph, src):
                related.add(src)
        for dst, labels in graph.successor_edges(memory_id).items():
            if (
                set(labels) - {_CHAIN_EDGE_TYPE}
                and _is_step_node(graph, dst)
                and dst in affected_step_ids
            ):
                related.add(dst)
    return related


def filter_non_chain_affected_steps(
    graph: Any,
    affected_nodes: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    *,
    faulty_memory_ids: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> list[Mapping[str, Any]]:
    """Return affected step records whose inclusion is backed by non-chain edges.

    ``compute_affected_subgraph`` is still the source of the affected node set,
    but delete-retrieved only scans steps reachable through dependency-bearing
    edges. Pure ``chain`` adjacency is execution order, not memory support, so a
    step reached only by chain is intentionally excluded. If a pair carries both
    ``chain`` and another label, the non-chain label keeps it eligible.
    """
    affected_ids = {
        node_id for node_id in affected_nodes.get("affected_node_ids", [])
        if isinstance(node_id, str)
    }
    seeds = [
        seed for seed in _resolved_seed_causes(affected_nodes)
        if graph.node(seed) is not None
    ]
    non_chain_reached = _non_chain_descendants(graph, seeds) & affected_ids
    affected_step_ids = {
        node_id for node_id in affected_ids
        if _is_step_node(graph, node_id)
    }

    relevant_ids = set(non_chain_reached) | set(seeds) | set(faulty_memory_ids)
    kept_step_ids = {
        node_id for node_id in non_chain_reached
        if node_id in affected_step_ids
    }
    for step_id in affected_step_ids:
        predecessor_ids = graph.predecessor_edges(step_id)
        successor_ids = graph.successor_edges(step_id)
        if any(
            src in relevant_ids and set(labels) - {_CHAIN_EDGE_TYPE}
            for src, labels in predecessor_ids.items()
        ) or any(
            dst in relevant_ids and set(labels) - {_CHAIN_EDGE_TYPE}
            for dst, labels in successor_ids.items()
        ):
            kept_step_ids.add(step_id)

    kept_step_ids.update(
        _source_fault_related_step_ids(
            graph,
            faulty_memory_ids,
            provenance or {},
            affected_step_ids,
        )
    )

    return [
        step for step in steps
        if isinstance(step.get("step_id"), str)
        and step["step_id"] in kept_step_ids
    ]
