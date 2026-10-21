# methods/ours/build_dependency_graph.py
# Build a dependency graph from execution traces and memory records.
# This graph captures memory-to-claim and claim-to-action relationships.
#
# The graph is fault-agnostic: it encodes pure data-flow structure (which
# step feeds which step, which step produced/consumed which memory) so a
# downstream module (affected_subgraph.py) can run forward BFS from a
# corrupted node to find everything transitively affected. No BFS, no
# fault-type branching, and no file output live here by design.

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import deque
from typing import Any, Dict, List, Set

try:  # importable both as ``methods.ours.build_dependency_graph`` and as a script
    from .graph_schema import (
        DependencyGraph,
        GraphNode,
        EDGE_PASSES,
        CONSUMING_STEP_TYPES,
        CONSUMPTION_STEP_TYPE_TO_EDGE,
        PRODUCING_STEP_TYPES,
        USER_INPUT_ID_PREFIX,
        is_memory_id as _is_memory_id,
        is_step_id as _is_step_id,
        user_input_node_id,
    )
except ImportError:  # pragma: no cover - direct-script fallback
    from graph_schema import (
        DependencyGraph,
        GraphNode,
        EDGE_PASSES,
        CONSUMING_STEP_TYPES,
        CONSUMPTION_STEP_TYPE_TO_EDGE,
        PRODUCING_STEP_TYPES,
        USER_INPUT_ID_PREFIX,
        is_memory_id as _is_memory_id,
        is_step_id as _is_step_id,
        user_input_node_id,
    )

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Edge-type lookup — pulled from graph_schema.EDGE_PASSES (single source of
# truth). Each construction pass below pulls the EdgeType label its edges carry
# by pass_id (the edge's verb name) rather than hardcoding a duplicate string,
# so this file can never fork the edge-type vocabulary that graph_schema.py
# owns. The consumption pass (delete/update/consolidate) instead branches per
# step via CONSUMPTION_STEP_TYPE_TO_EDGE, since one pass emits three labels.
# --------------------------------------------------------------------------- #

_PASS_BY_ID = {p.pass_id: p for p in EDGE_PASSES}
_VALID_EDGE_TYPES = frozenset(p.edge_type.value for p in EDGE_PASSES)


def _edge_type_for(pass_id: str) -> str:
    """Return the EdgeType string emitted by the EDGE_PASSES entry ``pass_id``."""
    return _PASS_BY_ID[pass_id].edge_type.value


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
# GraphNode and DependencyGraph now live in graph_schema.py (the single source
# of truth for the graph's shape) and are imported above. The ID-detection
# helpers (_is_memory_id / _is_step_id) and user_input_node_id likewise come
# from graph_schema. Everything below is pure construction logic that fills
# those dataclasses in.


# --------------------------------------------------------------------------- #
# Turn assignment (rule 1) — re-implemented locally, NOT imported from data/
# --------------------------------------------------------------------------- #

def _assign_turns(steps: List[Dict[str, Any]]) -> Dict[str, int]:
    """Map each step_id to a 1-indexed turn number.

    Mirrors ``data/inject_faults.py``'s ``assign_turns``: a new turn begins
    at each ``memory_read`` step; trailing memory write/update/consolidate/
    delete steps after a ``final_answer`` keep the turn of the most recent
    ``memory_read``. Steps before the first ``memory_read`` are turn 1. This
    is re-implemented here (rather than imported) to keep the method layer
    free of dependencies on the data layer.
    """
    step_turn: Dict[str, int] = {}
    turn = 0
    for step in steps:
        if step.get("step_type") == "memory_read":
            turn += 1
        step_turn[step["step_id"]] = max(turn, 1)
    return step_turn


# --------------------------------------------------------------------------- #
# Node construction (rules 2, 3, 7)
# --------------------------------------------------------------------------- #

def _build_nodes(
    steps: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    step_turn: Dict[str, int],
    session: List[str],
) -> Dict[str, GraphNode]:
    """Create user_input nodes, then step nodes, then memory nodes.

    User input comes first (rule 5): one node_kind="user_input" per turn,
    id ``user_input_{turn}`` (1-based, matching the agent's minted id), with
    the turn's literal text stored in ``raw`` and
    ``turn`` set. Steps follow (rule 2): node_kind="step" with step_type and
    turn set. Memories come last (rule 3): node_kind="memory" with
    memory_status set, but NO turn — turn is a step-level concept and would
    be guessing on a memory. The user-then-steps-then-memories insertion
    order is preserved per rule 5/7 so downstream traversal is deterministic.
    """
    nodes: Dict[str, GraphNode] = {}
    for idx, text in enumerate(session):
        turn = idx + 1                       # session[0] is turn 1
        uid = user_input_node_id(turn)
        nodes[uid] = GraphNode(
            node_id=uid,
            node_kind="user_input",
            turn=turn,
            raw={"turn": turn, "text": text},
        )
    for step in steps:
        sid = step["step_id"]
        nodes[sid] = GraphNode(
            node_id=sid,
            node_kind="step",
            step_type=step.get("step_type"),
            turn=step_turn.get(sid),
            raw=step,
        )
    for mem in memories:
        mid = mem["memory_id"]
        nodes[mid] = GraphNode(
            node_id=mid,
            node_kind="memory",
            memory_status=mem.get("status"),
            raw=mem,
        )
    return nodes


# --------------------------------------------------------------------------- #
# Edge construction (rule 4, passes a/b/c/d/g)
# --------------------------------------------------------------------------- #

def _add_edge(
    src: str,
    dst: str,
    edge_type: str,
    nodes: Dict[str, GraphNode],
    forward: Dict[str, Dict[str, Set[str]]],
    backward: Dict[str, Dict[str, Set[str]]],
    task_id: str,
) -> None:
    """Add the label ``edge_type`` to the src->dst edge (and its backward mirror).

    Per rule 6, a reference to an id that has no node (dangling/unknown) is
    silently skipped and logged at DEBUG — never crashes, never invents a
    placeholder node. Edges live in ``{neighbour: {edge_type, ...}}`` maps, so a
    pair's labels accumulate into a SET: adding a label already present is a
    harmless no-op, and adding a SECOND, different label (e.g. a ``cite`` on a
    pair that already carries ``chain``) makes both coexist instead of one
    overwriting the other. There is no overwrite/precedence arbitration left —
    nothing to arbitrate once multiple labels can share a pair. ``edge_type`` is
    one of the EdgeType strings pulled from graph_schema by the calling pass.
    """
    if src not in nodes:
        logger.debug("task %s: skipping edge from unknown id %s", task_id, src)
        return
    if dst not in nodes:
        logger.debug("task %s: skipping edge to unknown id %s", task_id, dst)
        return
    forward.setdefault(src, {}).setdefault(dst, set()).add(edge_type)
    backward.setdefault(dst, {}).setdefault(src, set()).add(edge_type)


def _pass_step_to_step(steps, nodes, forward, backward, task_id) -> None:
    """Pass (chain): structural step->step edges from next_step_ids.

    Captures the intra-turn reasoning chain
    claim->plan->tool_action->tool_observation->final_answer that the agent
    threads through ``next_step_ids`` (the step-chain successor links; see
    trace_logger.log_step / react_agent.set_next_step_ids). A single step may
    list MORE THAN ONE next_step_id — react_agent's set_next_step_ids populates
    a plan step with every tool_action it fans out to — so one step can emit
    several chain edges; the for-loop already handles that. Edges carry the
    ``chain`` label from EDGE_PASSES.
    """
    edge_type = _edge_type_for("chain")
    for step in steps:
        sid = step["step_id"]
        for out in step.get("next_step_ids", []):
            if _is_step_id(out):
                _add_edge(sid, out, edge_type, nodes, forward, backward, task_id)


def _pass_step_to_memory(steps, nodes, forward, backward, task_id) -> None:
    """Pass (produce): production edges step->memory for write/update/consolidate.

    A memory-mutating step records the memory ids it produced in
    ``generated_memory_ids`` (react_agent._log), so the step is the upstream
    source each such memory depends on. Per the agent's _log() usage
    generated_memory_ids only ever holds memory ids, so the ``_is_memory_id``
    filter is a defensive no-op kept for defense-in-depth (it cannot
    misclassify a non-memory id here, but guards against malformed input).
    Edges carry the ``produce`` label from EDGE_PASSES.
    """
    edge_type = _edge_type_for("produce")
    for step in steps:
        if step.get("step_type") not in PRODUCING_STEP_TYPES:
            continue
        sid = step["step_id"]
        for out in step.get("generated_memory_ids", []):
            if _is_memory_id(out):
                _add_edge(sid, out, edge_type, nodes, forward, backward, task_id)


def _pass_memory_to_step_retrieval(steps, nodes, forward, backward, task_id) -> None:
    """Pass (support): retrieval edges memory->step from ``used_ids``.

    ``used_ids`` is the authoritative "this step actually relied on this id"
    signal. It is MIXED-TYPE — it can hold memory ids, the claim/
    tool_observation step ids a step drew on, and prior-user-turn ids — so this
    pass extracts ONLY the memory-shaped (m_*) entries via ``_is_memory_id``;
    the non-memory entries are handled by the cite pass. If a retrieved memory is
    corrupt, every step that used it is downstream-affected, hence the
    memory->step direction. Edges carry the ``support`` label from EDGE_PASSES —
    the label section 8.1's independent-support check keys off.
    """
    edge_type = _edge_type_for("support")
    for step in steps:
        sid = step["step_id"]
        for mid in step.get("used_ids", []):
            if _is_memory_id(mid):
                _add_edge(mid, sid, edge_type, nodes, forward, backward, task_id)


def _pass_memory_consumption(steps, nodes, forward, backward, task_id) -> None:
    """Pass (delete/update/consolidate): memory->step consumption edges.

    For ``memory_update``/``memory_consolidate``/``memory_delete`` steps (the
    schema's ``CONSUMING_STEP_TYPES``), a memory_id in ``invalidated_memory_ids``
    is the OLD memory being updated/consolidated, or — for a memory_delete — the
    memory being torn down; it flows into the step. The edge's LABEL is branched
    by the step's own ``step_type`` via ``CONSUMPTION_STEP_TYPE_TO_EDGE``
    (memory_delete -> ``delete``, memory_update -> ``update``,
    memory_consolidate -> ``consolidate``) instead of one merged label, so a
    downstream walk can tell a deletion from an update from a consolidation. A
    memory_delete produces no new memory (``generated_memory_ids`` is empty, so
    the produce pass never fires for it), which is exactly why these edges
    matter: without them a deleted memory would have NO edge connecting it to the
    step that removed it, and any walk of "what invalidated this memory" via
    ``predecessor_edges`` would silently miss the delete. This is distinct from
    the support pass (which keys off ``used_ids`` on ANY step type). Because
    labels now accumulate in a set, an id that is in both
    ``invalidated_memory_ids`` and ``used_ids`` simply carries BOTH its
    consumption label and ``support`` — no pass ordering or overwrite to
    arbitrate. Per the agent's _log() usage invalidated_memory_ids only ever
    holds memory ids, so the ``_is_memory_id`` filter is a defensive no-op kept
    for defense-in-depth (consistent with the produce pass).
    """
    for step in steps:
        step_type = step.get("step_type")
        if step_type not in CONSUMING_STEP_TYPES:
            continue
        edge_type = CONSUMPTION_STEP_TYPE_TO_EDGE[step_type].value
        sid = step["step_id"]
        for in_id in step.get("invalidated_memory_ids", []):
            if _is_memory_id(in_id):
                _add_edge(in_id, sid, edge_type, nodes, forward, backward, task_id)


def _pass_user_input_to_first_step(
    nodes, step_turn, forward, backward, n_turns, task_id
) -> None:
    """Pass (initiate): root edges user_input->first step of each turn.

    For each turn, the FIRST step is the one with the minimum index among
    all steps assigned to that turn (normally the turn's memory_read, since
    that is what starts a new turn in ``_assign_turns``). The edge makes the
    whole turn's step chain reachable from a single root node representing
    "the user spoke this turn". Turns with no steps get no edge (flagged by
    validate_graph). Relies on ``step_turn`` preserving the original step
    order so the first matching id is the earliest step. Edges carry the
    ``initiate`` label from EDGE_PASSES.
    """
    edge_type = _edge_type_for("initiate")
    first_step: Dict[int, str] = {}
    for sid, turn in step_turn.items():
        if turn not in first_step:
            first_step[turn] = sid
    for turn in range(1, n_turns + 1):
        uid = user_input_node_id(turn)
        if uid not in nodes:
            continue
        sid = first_step.get(turn)
        if sid is not None:
            _add_edge(uid, sid, edge_type, nodes, forward, backward, task_id)


def _pass_memory_lineage(memories, nodes, forward, backward, task_id) -> None:
    """Pass (derive): lineage edges memory->memory from each memory's ``derived_from``.

    For every memory record, each id in its ``derived_from`` list (section
    6.1) is an ancestor the memory was derived from, so we add an
    ancestor->derived edge. This lets a faulty ancestor reach its derived
    memories via ``descendants()`` (section 18 / the Derived Memory Cleanup
    Recall metric, section 11) without re-parsing ``GraphNode.raw`` per node.
    Edges carry the ``derive`` label from EDGE_PASSES. Ancestors that are unknown
    ids are skipped by ``_add_edge`` (rule 6) and surfaced by validate_graph as a
    dangling/unfired lineage reference.
    """
    edge_type = _edge_type_for("derive")
    for mem in memories:
        derived_id = mem["memory_id"]
        for ancestor in mem.get("derived_from", []) or []:
            if _is_memory_id(ancestor):
                _add_edge(
                    ancestor, derived_id, edge_type, nodes, forward, backward, task_id
                )


def _pass_memory_supersede(memories, nodes, forward, backward, task_id) -> None:
    """Pass (supersede): memory->memory edges from each memory's OWN ``supersedes``.

    For every memory record, its ``supersedes`` field (section 6.1) — a single
    memory_id or None, NOT a step field — names the OLD memory this record
    replaces, so we add an old->this edge (old_memory_id -> this_memory_id),
    mirroring the derive pass's memory-record-driven structure with a different
    field. This is the NEW edge that fixes the asymmetry: previously a superseded
    memory's own backward map recorded nothing about being superseded (the
    supersedes link lived only on the REPLACEMENT's record), so a walk of the old
    memory's predecessors saw only its produce-edge step. Now the old memory
    carries an incoming supersede edge from its replacement. An unknown old id is
    skipped by ``_add_edge`` (rule 6) and surfaced by validate_graph. The field
    may be absent, None, or a string; only a memory-shaped string fires an edge.
    """
    edge_type = _edge_type_for("supersede")
    for mem in memories:
        new_id = mem["memory_id"]
        old_id = mem.get("supersedes")
        if isinstance(old_id, str) and _is_memory_id(old_id):
            _add_edge(old_id, new_id, edge_type, nodes, forward, backward, task_id)


def _is_user_input_id(token: str) -> bool:
    """True if ``token`` names a user_input node (``user_input_`` prefix).

    Mirrors graph_schema's USER_INPUT_ID_PREFIX so the cite pass can recognise a
    user_input id cited in ``used_ids`` without re-deriving the format.
    """
    return token.startswith(USER_INPUT_ID_PREFIX)


def _pass_step_cites_step(steps, nodes, forward, backward, task_id) -> None:
    """Pass (cite): "cite" edges from used_ids entries that reference a claim
    step, a tool_observation step, or a user_input node — the non-memory
    entries the support pass already correctly ignores via ``_is_memory_id``.

    For each step's ``used_ids``, any entry that ``_is_step_id()`` (a claim or
    tool_observation step the citing step drew on) or matches the user_input id
    format AND is a node in the graph gets an edge from that cited node to the
    citing step (cited-node->citing-step), so a corrupt cited step/turn reaches
    everything that leaned on it via a forward walk — the same direction the
    support pass uses for a cited memory. Memory-shaped entries are skipped here
    (the support pass owns those). ``_add_edge`` drops any cited id with no node
    (rule 6). Edges carry the ``cite`` label from EDGE_PASSES. A cite edge can
    now COEXIST with the chain pass's ``chain`` label on the same (src, dst) pair
    with NO special-casing — both labels accumulate in the pair's label set,
    which is what fixes the collision found earlier in this project.
    """
    edge_type = _edge_type_for("cite")
    for step in steps:
        sid = step["step_id"]
        for cited in step.get("used_ids", []):
            if _is_memory_id(cited):
                continue  # the support pass handles memory citations
            if _is_step_id(cited) or _is_user_input_id(cited):
                _add_edge(cited, sid, edge_type, nodes, forward, backward, task_id)


def build_dependency_graph(
    task_id: str,
    steps: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    session: List[str],          # session[0] is turn 1's text, etc.
) -> DependencyGraph:
    """Build the full dependency graph for one task.

    Runs turn assignment, node construction, then the data-flow edge passes:
    the intra-turn step chain (``chain``), step->memory production (``produce``),
    memory->step retrieval (``support``) and consumption (``delete``/``update``/
    ``consolidate``, branched by the mutating step's step_type), the
    user_input->first-step root anchor (``initiate``), the memory->memory lineage
    pass (``derive``) that wires each memory's ``derived_from`` ancestors to it,
    the memory->memory ``supersede`` pass that wires each memory's OWN
    ``supersedes`` predecessor to it, and the ``cite`` pass that wires the
    non-memory ``used_ids`` entries (a cited claim/tool_observation step or
    user_input node) to the citing step. Because edge labels accumulate in a set,
    pass order no longer arbitrates anything: a pair touched by two passes (e.g.
    chain + cite, or support + a consumption label) simply carries both labels.
    Returns an in-memory DependencyGraph; performs no file I/O.
    """
    step_turn = _assign_turns(steps)
    n_turns = max(step_turn.values()) if step_turn else 0
    nodes = _build_nodes(steps, memories, step_turn, session)

    forward_edges: Dict[str, Dict[str, Set[str]]] = {}
    backward_edges: Dict[str, Dict[str, Set[str]]] = {}

    _pass_step_to_step(steps, nodes, forward_edges, backward_edges, task_id)
    _pass_step_to_memory(steps, nodes, forward_edges, backward_edges, task_id)
    _pass_memory_to_step_retrieval(steps, nodes, forward_edges, backward_edges, task_id)
    _pass_memory_consumption(steps, nodes, forward_edges, backward_edges, task_id)
    _pass_user_input_to_first_step(
        nodes, step_turn, forward_edges, backward_edges, n_turns, task_id
    )
    _pass_memory_lineage(memories, nodes, forward_edges, backward_edges, task_id)
    _pass_memory_supersede(memories, nodes, forward_edges, backward_edges, task_id)
    _pass_step_cites_step(steps, nodes, forward_edges, backward_edges, task_id)

    return DependencyGraph(
        task_id=task_id,
        nodes=nodes,
        forward_edges=forward_edges,
        backward_edges=backward_edges,
        step_turn=step_turn,
        n_turns=n_turns,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_graph(graph: DependencyGraph) -> List[str]:
    """Return human-readable warnings for structural issues (never raises).

    Checks: isolated memory nodes (likely a missing used_ids entry),
    memory-mutating steps lacking an outgoing memory edge, n_turns agreeing
    with the max step turn, that no node also appears as a dangling target
    (which would mean rule 6's skip fired incorrectly), that every label in
    every edge's label set is one declared by an EdgePassSpec, and — mirroring
    the producing-step check — that a memory with a non-empty ``derived_from``
    list has an incoming ``derive`` edge from each listed ancestor, and a memory
    naming a ``supersedes`` predecessor has an incoming ``supersede`` edge from
    it. Edge label sets may carry MORE THAN ONE label per pair (e.g. chain +
    cite), so membership tests use ``in`` against the set, not ``==``.
    """
    warnings: List[str] = []
    # PRODUCING_STEP_TYPES (NOT consuming): only write/update/consolidate steps
    # are expected to emit an outgoing memory edge. memory_delete consumes a
    # memory (the delete pass) but produces none, so it is correctly absent here
    # and is not flagged for "no outgoing memory edge".
    producing = PRODUCING_STEP_TYPES
    lineage_edge_type = _edge_type_for("derive")
    supersede_edge_type = _edge_type_for("supersede")

    for node_id, node in graph.nodes.items():
        if node.node_kind == "memory":
            has_in = bool(graph.predecessors(node_id))
            has_out = bool(graph.successors(node_id))
            if not has_in and not has_out:
                warnings.append(
                    f"isolated memory node {node_id} (status="
                    f"{node.memory_status}): no incoming or outgoing edges; "
                    f"likely a missing used_ids reference"
                )
            # Derive lineage check: each ancestor in derived_from should have
            # produced an incoming derive edge ancestor->this memory. A missing
            # one means a dangling reference or a pass that didn't fire.
            incoming = graph.predecessor_edges(node_id)
            derived_from = node.raw.get("derived_from") or []
            for ancestor in derived_from:
                if not _is_memory_id(ancestor):
                    continue
                if lineage_edge_type not in incoming.get(ancestor, set()):
                    warnings.append(
                        f"memory {node_id} lists {ancestor} in derived_from but "
                        f"has no incoming {lineage_edge_type} edge from it "
                        f"(dangling ancestor or the derive pass did not fire)"
                    )
            # Supersede check (mirrors the derive check): a memory naming an old
            # memory in its own supersedes field should have an incoming
            # supersede edge old->this memory.
            superseded = node.raw.get("supersedes")
            if isinstance(superseded, str) and _is_memory_id(superseded):
                if supersede_edge_type not in incoming.get(superseded, set()):
                    warnings.append(
                        f"memory {node_id} supersedes {superseded} but has no "
                        f"incoming {supersede_edge_type} edge from it (dangling "
                        f"reference or the supersede pass did not fire)"
                    )
        elif node.node_kind == "step" and node.step_type in producing:
            mem_targets = [
                t for t in graph.successors(node_id)
                if graph.nodes.get(t) and graph.nodes[t].node_kind == "memory"
            ]
            if not mem_targets:
                warnings.append(
                    f"{node.step_type} step {node_id} has no outgoing edge to "
                    f"a memory node (expected to produce one)"
                )
        elif node.node_kind == "user_input":
            # The invariant is "exactly one INITIATE edge (to the turn's first
            # step)", not "exactly one outgoing edge of any type": the cite pass
            # legitimately adds cite edges FROM a user_input node whenever a
            # later step cites that prior turn in its used_ids, so counting all
            # successors would false-positive on every cited turn. Count only the
            # initiate edge — and via set membership, since one pair could carry
            # both initiate AND cite if a step both starts the turn and is cited.
            initiates_type = _edge_type_for("initiate")
            initiates_targets = [
                dst for dst, etypes in graph.successor_edges(node_id).items()
                if initiates_type in etypes
            ]
            n_initiates = len(initiates_targets)
            if n_initiates == 0:
                warnings.append(
                    f"turn {node.turn} user_input node {node_id} has no "
                    f"initiate edge — no step found for turn {node.turn}"
                )
            elif n_initiates > 1:
                warnings.append(
                    f"turn {node.turn} user_input node {node_id} has "
                    f"{n_initiates} initiate edges, expected exactly 1"
                )

    max_step_turn = max(
        (n.turn for n in graph.nodes.values()
         if n.node_kind == "step" and n.turn is not None),
        default=0,
    )
    if graph.n_turns != max_step_turn:
        warnings.append(
            f"n_turns ({graph.n_turns}) != max step turn ({max_step_turn})"
        )

    # Any id referenced by an edge must have a real node (rule 6 should have
    # skipped it otherwise). Edge endpoints not in nodes => skip misfired.
    referenced: Set[str] = set()
    for src, dsts in graph.forward_edges.items():
        referenced.add(src)
        referenced.update(dsts)
    for ref in referenced:
        if ref not in graph.nodes:
            warnings.append(
                f"edge references id {ref} that has no node "
                f"(dangling-reference skip misfired)"
            )

    # Every label in every edge's label set must be one some EdgePassSpec
    # declares (no edge carries a label outside the EdgeType enum). A memory->
    # memory invariant holds for the two lineage labels: a derive edge and a
    # supersede edge must each link two memory nodes (the ancestor/old memory and
    # the derived/replacement memory). A single pair may legitimately carry
    # several labels, so we inspect the whole label set per pair.
    memory_memory_labels = {lineage_edge_type, supersede_edge_type}
    for src, dsts in graph.forward_edges.items():
        for dst, etypes in dsts.items():
            for etype in etypes:
                if etype not in _VALID_EDGE_TYPES:
                    warnings.append(
                        f"edge {src}->{dst} carries unknown edge_type {etype!r} "
                        f"(not declared by any EdgePassSpec)"
                    )
            for etype in etypes & memory_memory_labels:
                src_node = graph.nodes.get(src)
                dst_node = graph.nodes.get(dst)
                if (src_node and src_node.node_kind != "memory") or (
                    dst_node and dst_node.node_kind != "memory"
                ):
                    warnings.append(
                        f"{etype} edge {src}->{dst} is not memory->memory "
                        f"(it must link two memory nodes)"
                    )

    return warnings


# --------------------------------------------------------------------------- #
# Visualization (optional manual inspection only — NOT part of graph building)
# --------------------------------------------------------------------------- #

def draw_dependency_graph(graph: DependencyGraph, output_path: str) -> None:
    """Render ``graph`` to an image at ``output_path`` for manual inspection.

    Builds a throwaway ``networkx.DiGraph`` mirror of the DependencyGraph
    (the original is never mutated or serialized) and saves a static PNG via
    matplotlib's non-interactive Agg backend — no window is ever shown. Nodes
    are colour-coded by ``node_kind`` (user_input / step / memory) and labelled
    with their id plus a kind-specific second line (step_type, memory status/
    fault_type). The parent directory of ``output_path`` is created if missing.
    This is purely an inspection aid and has no effect on graph construction.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend: save only, never display
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Graph visualization requires matplotlib (and networkx). "
            "Install them with `pip install matplotlib networkx` and retry."
        ) from exc
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Graph visualization requires networkx (and matplotlib). "
            "Install them with `pip install matplotlib networkx` and retry."
        ) from exc

    # Colour + label scheme keyed by node_kind. Distinct hues so the three
    # kinds are easy to tell apart at a glance.
    kind_color = {
        "user_input": "#9ad0f5",   # light blue
        "step": "#b6e3b6",         # light green
        "memory": "#f5c98a",       # light orange
    }
    default_color = "#dddddd"

    def _label(node: GraphNode) -> str:
        if node.node_kind == "step":
            # Surface the section 6.4 semantic_type (claim / plan / tool_action
            # / …) so reasoning-step kinds are distinguishable at a glance; it
            # is derived from step_type, falling back to a bare "step".
            second = node.semantic_type or node.step_type or "step"
        elif node.node_kind == "memory":
            status = node.memory_status or node.raw.get("status")
            fault = node.raw.get("fault_type")
            parts = [p for p in (status, fault) if p]
            second = "/".join(parts) if parts else "memory"
        elif node.node_kind == "user_input":
            second = "user_input"
        else:
            second = node.node_kind or "?"
        return f"{node.node_id}\n{second}"

    g = nx.DiGraph()
    for node_id, node in graph.nodes.items():
        g.add_node(
            node_id,
            color=kind_color.get(node.node_kind, default_color),
            label=_label(node),
        )
    for src, dsts in graph.forward_edges.items():
        for dst in dsts:
            if src in graph.nodes and dst in graph.nodes:
                g.add_edge(src, dst)

    # Deterministic layout so repeated runs are visually comparable.
    pos = nx.spring_layout(g, seed=42, k=0.9)

    node_colors = [g.nodes[n]["color"] for n in g.nodes]
    labels = {n: g.nodes[n]["label"] for n in g.nodes}

    # Scale figure with node count so small/medium traces stay readable.
    n_nodes = max(g.number_of_nodes(), 1)
    side = min(max(8.0, n_nodes * 0.55), 30.0)
    fig, ax = plt.subplots(figsize=(side, side))

    nx.draw_networkx_edges(
        g, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=14,
        edge_color="#888888", width=1.0, alpha=0.8,
    )
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=node_colors, node_size=1600,
        edgecolors="#333333", linewidths=0.8,
    )
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)

    legend_handles = [
        Patch(facecolor=kind_color["user_input"], edgecolor="#333333",
              label="user_input"),
        Patch(facecolor=kind_color["step"], edgecolor="#333333", label="step"),
        Patch(facecolor=kind_color["memory"], edgecolor="#333333",
              label="memory"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.set_title(f"Dependency graph: {graph.task_id}")
    ax.axis("off")
    fig.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Track traversal (manual inspection only — NOT part of graph building)
# --------------------------------------------------------------------------- #

def _track_from_user_input(graph: DependencyGraph, start: str) -> List[str]:
    """Breadth-first traversal of every node reachable from ``start``.

    Walks ``forward_edges`` out of a user_input node, so it surfaces every
    step AND memory node that turn's data flow touches — not just the
    first step. Successors are sorted before being queued so the order is
    deterministic across runs even though edges live in (unordered) sets.
    Returns node ids in BFS discovery order, excluding ``start`` itself.
    """
    visited = {start}
    order: List[str] = []
    queue = deque([start])
    while queue:
        node_id = queue.popleft()
        for nxt in sorted(graph.successors(node_id)):
            if nxt not in visited:
                visited.add(nxt)
                order.append(nxt)
                queue.append(nxt)
    return order


def _describe_node(node: GraphNode) -> str:
    """Short kind-specific label for a node, used in CLI track output.

    Step nodes report their section 6.4 ``semantic_type`` (claim / plan /
    tool_action / …) so reasoning-step kinds stay distinguishable; memory and
    user_input nodes report status / kind as before.
    """
    if node.node_kind == "step":
        return f"step:{node.semantic_type or node.step_type or '?'}"
    if node.node_kind == "memory":
        return f"memory:{node.memory_status or '?'}"
    if node.node_kind == "user_input":
        return "user_input"
    return node.node_kind or "?"


# --------------------------------------------------------------------------- #
# CLI for manual testing
# --------------------------------------------------------------------------- #

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and inspect a dependency graph for one task."
    )
    parser.add_argument("--memories", required=True, help="memories JSONL path")
    parser.add_argument("--steps", required=True, help="trace steps JSONL path")
    parser.add_argument(
        "--session", required=True, help="tasks_full_session JSONL path"
    )
    parser.add_argument("--task_id", required=True, help="task_id to filter to")
    parser.add_argument(
        "--output_dir",
        default="methods/ours/test",
        help="directory to write the serialized graph (dependency_graph.json) "
             "into (default: methods/ours/test)",
    )
    parser.add_argument(
        "--draw_graph",
        action="store_true",
        help="render the dependency graph to an image for manual inspection",
    )
    parser.add_argument(
        "--graph_output",
        default=None,
        help="output image path (default: outputs/graphs/"
             "{task_id}_dependency_graph.png)",
    )
    args = parser.parse_args()

    all_steps = _load_jsonl(args.steps)
    all_memories = _load_jsonl(args.memories)
    all_sessions = _load_jsonl(args.session)
    steps = [s for s in all_steps if s.get("task_id") == args.task_id]
    memories = [m for m in all_memories if m.get("task_id") == args.task_id]
    session_recs = [r for r in all_sessions if r.get("task_id") == args.task_id]
    session = session_recs[0].get("session", []) if session_recs else []

    graph = build_dependency_graph(args.task_id, steps, memories, session)

    print(f"=== Dependency graph for {args.task_id} ===")
    print(f"steps loaded: {len(steps)}  memories loaded: {len(memories)}")

    # Serialize the built graph to dependency_graph.json under --output_dir.
    # Writing here (in the CLI block) is fine — the no-I/O boundary applies only
    # to build_dependency_graph() itself.
    os.makedirs(args.output_dir, exist_ok=True)
    graph_json_path = os.path.join(args.output_dir, "dependency_graph.json")
    with open(graph_json_path, "w", encoding="utf-8") as fh:
        json.dump(graph.to_dict(), fh, indent=2, ensure_ascii=False)
    print(f"wrote graph JSON to: {graph_json_path}")

    warnings = validate_graph(graph)
    print(f"\n--- validation warnings ({len(warnings)}) ---")
    for w in warnings:
        print(f"  ! {w}")
    if not warnings:
        print("  (none)")

    n_step = sum(1 for n in graph.nodes.values() if n.node_kind == "step")
    n_mem = sum(1 for n in graph.nodes.values() if n.node_kind == "memory")
    n_user = sum(1 for n in graph.nodes.values() if n.node_kind == "user_input")
    # n_pairs counts distinct (src, dst) edges; edge_type_counts now counts label
    # OCCURRENCES across all pairs, so a pair carrying both chain and cite adds 1
    # to n_pairs but 1 each to the chain and cite tallies (their sum exceeds
    # n_pairs whenever any pair is multi-labelled).
    n_pairs = sum(len(d) for d in graph.forward_edges.values())
    edge_type_counts: Dict[str, int] = {}
    for dsts in graph.forward_edges.values():
        for etypes in dsts.values():
            for etype in etypes:
                edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1
    n_labels = sum(edge_type_counts.values())
    n_multi = sum(
        1 for dsts in graph.forward_edges.values()
        for etypes in dsts.values() if len(etypes) > 1
    )
    print("\n--- summary ---")
    print(f"  step nodes:   {n_step}")
    print(f"  memory nodes: {n_mem}")
    print(f"  user_input nodes: {n_user}")
    print(f"  edge pairs:   {n_pairs}  (multi-label pairs: {n_multi})")
    print(f"  label occurrences: {n_labels}")
    for etype in sorted(edge_type_counts):
        print(f"    {etype}: {edge_type_counts[etype]}")
    print(f"  n_turns:      {graph.n_turns}")

    print("\n--- memory node successors (with edge_type set) ---")
    for node_id, node in graph.nodes.items():
        if node.node_kind == "memory":
            succ = graph.successor_edges(node_id)
            rendered = sorted(f"{dst} [{','.join(sorted(succ[dst]))}]" for dst in succ)
            print(f"  {node_id} (status={node.memory_status}) -> {rendered}")

    print("\n--- track per user_input (full traversal) ---")
    for node_id, node in graph.nodes.items():
        if node.node_kind == "user_input":
            text = node.raw.get("text", "")
            if len(text) > 60:
                text = text[:60] + "..."
            track = _track_from_user_input(graph, node_id)
            n_steps = sum(1 for t in track if graph.nodes[t].node_kind == "step")
            n_mem = sum(1 for t in track if graph.nodes[t].node_kind == "memory")
            print(f'  {node_id} (turn {node.turn}, "{text}")')
            print(f"    touches {n_steps} step(s), {n_mem} memory/memories")
            if track:
                for t in track:
                    print(f"      -> {t} [{_describe_node(graph.nodes[t])}]")
            else:
                print("      (no reachable nodes)")

    if args.draw_graph:
        output_path = args.graph_output or (
            f"outputs/graphs/{args.task_id}_dependency_graph.png"
        )
        draw_dependency_graph(graph, output_path)
        print(f"\n--- visualization ---")
        print(f"  saved graph image to: {output_path}")