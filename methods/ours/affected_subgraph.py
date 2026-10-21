# methods/ours/affected_subgraph.py
# Compute the downstream-affected subgraph for one OR MORE seed memory nodes.
#
# Given the fault-agnostic dependency graph from build_dependency_graph.py and
# the faulty_memory_ids injected into a task (1-3 per task, section 7's I/O
# table), this module follows forward (src->dst) edges to find the transitive
# closure of everything that depends on those memories: the steps that read
# them, the claims/plans/actions threaded off those steps, any NEW memories
# those steps produced, and — recursively — everything that depends on those
# derived memories in turn. It is pure reachability: NO independent-support
# filtering or pruning happens here (that is support_checker.py's job later);
# this module just returns the raw reachable set for rollback_optimizer.py.
#
# The real pipeline (run_ours.py) calls the MULTI-SEED compute_affected_subgraph
# with the whole faulty_memory_ids list so the union de-dups at the BFS visited-
# set level (a node reachable from two faulty memories is counted once). The
# single-seed helpers (_compute_single_seed_affected / build_affected_subgraph)
# stay for the "track one faulty memory's blast radius" debugging use case.

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Tuple, Union

try:  # importable both as ``methods.ours.affected_subgraph`` and as a script
    from .graph_schema import DependencyGraph, GraphNode, is_memory_id, is_step_id
    from .fault_provenance import find_producing_step
except ImportError:  # pragma: no cover - direct-script fallback
    from graph_schema import DependencyGraph, GraphNode, is_memory_id, is_step_id
    from fault_provenance import find_producing_step

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Edge types excluded from reachability
# --------------------------------------------------------------------------- #
# "chain" is PURE EXECUTION SEQUENCING ("this step comes next") with NO
# evidentiary content — it only records the implementation order of the
# reasoning chain, not that the later step RELIED ON the earlier one. So a node
# that is merely next-in-sequence after a contaminated node (reached only via a
# chain edge, with no support/cite/produce/derive/etc. backing) is NOT thereby
# contaminated: nothing in it actually cites the contaminated content. Because
# chain edges span turn boundaries (final_answer -> the next turn's first step
# via the full step sequence), traversing them would wrongly sweep an entire
# later turn's downstream chain into the affected set even when nothing in it
# ever cites anything contaminated. Reachability therefore follows only the
# evidentiary / structural-production labels (support, cite, produce, derive,
# delete, update, consolidate, supersede, initiate) and NEVER chain. This is
# applied as an EXCLUSION set so the change is local to affected-subgraph
# reachability and the general-purpose graph API stays edge-type-agnostic by
# default.
_REACHABILITY_EXCLUDED_EDGE_TYPES = frozenset({"chain"})


# --------------------------------------------------------------------------- #
# Node typing — the grouping key used for nodes_by_type / counts_by_type
# --------------------------------------------------------------------------- #
# The fine-grained type label is owned by graph_schema.py: GraphNode.semantic_
# type computes step_type for step nodes and node_kind for memory/user_input,
# which is exactly the granularity the downstream eval metrics key off
# (tool_action count feeds Replay Cost, memory_write feeds Derived Cleanup
# Recall, claim feeds Claim F1). A step node missing its step_type yields None,
# which we bucket under the bare kind "step" so a grouping key is always a str.


def _node_type(node: GraphNode) -> str:
    """Grouping key for an affected node — a thin call to ``semantic_type``.

    Defers entirely to ``GraphNode.semantic_type`` (the single owner of the
    node_kind+step_type -> label mapping) and only supplies the "step" fallback
    for a step node with no step_type (semantic_type returns None there).
    """
    return node.semantic_type or "step"


# --------------------------------------------------------------------------- #
# Core reachability
# --------------------------------------------------------------------------- #

def affected_node_ids(graph: DependencyGraph, *seeds: str) -> List[str]:
    """Forward-reachable closure from one or more seeds (seeds EXCLUDED).

    Thin wrapper over ``DependencyGraph.descendants`` that first validates each
    seed. Seeds are one or more memory OR step node ids — the real pipeline
    resolves each faulty memory to the step/memory it should seed from (see
    :func:`resolve_seed_id`), so a seed may be a step id (``s_*``) as well as a
    memory id (``m_*``). A seed that is missing from the graph, OR is neither a
    valid step id nor a valid memory id (i.e. not a recognized node kind at
    all), is WARNED about and SKIPPED rather than raising — mirroring
    build_dependency_graph.py's "skip + log, never crash on a bad reference"
    pattern, so one bad id in a list cannot kill the whole multi-seed report.
    ``descendants(*seeds)`` unions the reachable sets at the BFS visited-set
    level (a node reachable from two seeds is visited once), so the returned
    list is already de-duplicated and in BFS discovery order, NOT including any
    seed.

    Reachability EXCLUDES ``chain`` edges (``exclude_edge_types=
    {"chain"}``). A chain edge is pure execution sequencing ("this step comes
    next") with NO evidentiary content — a node merely next-in-sequence after a
    contaminated node is NOT thereby contaminated; it must be reached via a real
    evidentiary / structural-production edge (support / cite / produce / derive
    / delete / update / consolidate / supersede / initiate) to count as
    affected. Because chain edges span turn boundaries (final_answer -> the next
    turn's first step), counting them would wrongly sweep a later turn's entire
    downstream chain into the affected set even when nothing in it ever cites
    anything contaminated. See ``_REACHABILITY_EXCLUDED_EDGE_TYPES``. A pair
    carrying BOTH chain and a real label (e.g. chain + cite) is still traversed
    via the real label — only chain-ONLY hops are dropped.

    The multi-generation behaviour the caller cares about — a derived memory
    reached mid-walk re-seeding its own downstream effects — is handled by the
    edge structure itself (a producer step has a ``produce`` edge to the memory
    it creates, which carries ``support`` edges on to its consumers), so a single
    forward BFS already yields the full closure.
    """
    valid_seeds: List[str] = []
    for seed in seeds:
        if graph.node(seed) is None:
            print(f"  ! warning: seed {seed!r} is not a node in graph "
                  f"{graph.task_id!r}; skipping it")
            continue
        if not (is_memory_id(seed) or is_step_id(seed)):
            print(f"  ! warning: seed {seed!r} is neither a memory ('m_') nor a "
                  f"step ('s_') node id; skipping it")
            continue
        valid_seeds.append(seed)
    return graph.descendants(
        *valid_seeds, exclude_edge_types=_REACHABILITY_EXCLUDED_EDGE_TYPES
    )


def resolve_seed_id(
    memory_id: str, provenance: Dict[str, Any]
) -> Union[str, Tuple[str, str]]:
    """Apply the seed-priority rule for one faulty memory_id given its
    fault_provenance.py entry (``provenance[memory_id]``, i.e. the dict with
    ``producing_step_id`` / ``stale_window`` / ``ancestor_chain`` keys).
    Returns the seed to drive reachability from, per the three-case priority
    below. Does NOT validate the result is a real graph node — that is
    :func:`affected_node_ids`'s job, which already warns+skips invalid seeds.

    Three-way return shape (one per case):

    1. **stale-and-failed** (``stale_window`` is not None AND ``took_effect`` is
       False) -> returns a TUPLE ``(memory_id, cutoff_step_id)`` where
       ``cutoff_step_id = stale_window["mutation_step_id"]`` (the FAILED mutation
       step). This is the ONLY case that returns a tuple, and it differs in shape
       from the other two because of WHERE the contamination actually flows:
       a failed mutation (e.g. a memory_delete with empty ``next_step_ids`` and no
       outgoing produce/chain edges) is a DEAD END in the graph — seeding from
       that step reaches nothing. The contamination instead flows through the
       MEMORY ITSELF, which keeps being read/cited AFTER the failed mutation
       (support/cite edges to later steps). So we must seed from ``memory_id`` to
       reach those edges — but seeding from the memory ALSO sweeps in the CLEAN
       window between the memory's creation and the failed mutation (those steps
       are graph-reachable from the memory via support edges too, yet were already
       validated clean by ``find_stale_window`` and must stay excluded). The
       tuple therefore signals "seed from ``memory_id``, but only KEEP results
       occurring strictly AFTER ``cutoff_step_id`` in trace order" — memory
       seeding to reach the contamination, plus a temporal filter to drop the
       clean window the same BFS would otherwise incorrectly include. The filter
       itself is applied by :func:`compute_affected_subgraph`.
    2. Otherwise, ``producing_step_id`` is not None -> seed from that step_id (a
       plain str). Covers poisoned / wrong_user / summary_drift, and a stale case
       whose mutation DID take effect (which needs no special window treatment and
       can seed from the producing step like any other non-stale fault).
    3. Otherwise (no producing step at all — a root/seed memory with no upstream
       step) -> seed from ``memory_id`` itself (a plain str; the memory node,
       today's existing behavior; the ONLY plain-str case that still seeds from a
       memory directly).
    """
    stale_window = provenance.get("stale_window")
    if stale_window is not None and stale_window.get("took_effect") is False:
        return (memory_id, stale_window["mutation_step_id"])

    producing_step_id = provenance.get("producing_step_id")
    if producing_step_id is not None:
        return producing_step_id

    return memory_id


def build_affected_subgraph(
    graph: DependencyGraph, memory_id: str, include_seed: bool = True
) -> DependencyGraph:
    """Return a DependencyGraph restricted to the seed's downstream closure.

    Single-seed convenience for the "track one faulty memory's blast radius"
    debugging use case. NOTE: this debugging helper seeds DIRECTLY from the
    memory node, NOT via the provenance-resolved step seed the real pipeline
    uses (chosen option (a): these helpers are for quick single-seed inspection,
    and forcing them to take a provenance dict adds friction to that one job for
    little benefit). Use :func:`compute_affected_subgraph` with a real
    provenance dict to see the actual seed resolution
    (failed-mutation-step / producing-step / memory-node priority).

    The node set is the forward-reachable closure of ``memory_id``; the seed
    itself is included when ``include_seed`` is True (the default) so
    rollback_optimizer.py keeps the faulty node in context. Reachability stays
    consistent with the main pipeline: it routes through
    :func:`affected_node_ids`, which excludes ``chain`` edges
    (``_REACHABILITY_EXCLUDED_EDGE_TYPES``), so this debugging helper never
    silently diverges from ``compute_affected_subgraph``'s closure semantics —
    a node reachable only via a pure-sequencing chain edge is excluded here too.
    Edges are the induced subgraph — an edge survives only if BOTH endpoints are
    in the kept set — and each surviving edge carries its full ``edge_type``
    label SET through (chain edges included on the display, see
    :func:`_induced_edges`), since DependencyGraph stores edges as
    ``{neighbour: {edge_type, ...}}`` (a pair may carry several labels, e.g.
    chain + cite). ``step_turn`` and ``n_turns`` are recomputed over the
    surviving step nodes so the slice is self-consistent.

    A missing or non-memory ``memory_id`` is warned about and skipped by
    :func:`affected_node_ids` (yielding an empty closure), not raised on.
    """
    affected = affected_node_ids(graph, memory_id)
    kept = set(affected)
    if include_seed and graph.node(memory_id) is not None:
        kept.add(memory_id)

    nodes = {nid: graph.nodes[nid] for nid in kept if nid in graph.nodes}

    forward: Dict[str, Dict[str, set]] = {}
    backward: Dict[str, Dict[str, set]] = {}
    for src in nodes:
        dsts = {d: t for d, t in graph.successor_edges(src).items() if d in nodes}
        if dsts:
            forward[src] = dsts
            for d, t in dsts.items():
                backward.setdefault(d, {})[src] = set(t)

    step_turn = {
        nid: n.turn
        for nid, n in nodes.items()
        if n.node_kind == "step" and n.turn is not None
    }
    n_turns = max(step_turn.values()) if step_turn else 0

    return DependencyGraph(
        task_id=f"{graph.task_id}::affected::{memory_id}",
        nodes=nodes,
        forward_edges=forward,
        backward_edges=backward,
        step_turn=step_turn,
        n_turns=n_turns,
    )


# --------------------------------------------------------------------------- #
# Report assembly (the affected_nodes.json payload)
# --------------------------------------------------------------------------- #

def _compute_single_seed_affected(
    graph: DependencyGraph, memory_id: str
) -> Dict[str, Any]:
    """Affected-subgraph report for ONE seed memory (internal helper).

    The single-seed counterpart of :func:`compute_affected_subgraph`, kept for
    the "track one faulty memory's blast radius" debugging use case. NOTE: like
    :func:`build_affected_subgraph`, this debugging helper seeds DIRECTLY from
    the memory node, NOT via the provenance-resolved step seed the real pipeline
    uses (chosen option (a) — see that function's docstring). Use
    :func:`compute_affected_subgraph` with a real provenance dict to see the
    actual seed resolution. Returns a JSON-serializable dict shaped like the
    multi-seed payload but keyed on a singular ``seed_memory_id``. A
    missing/non-memory seed is warned + skipped by :func:`affected_node_ids`
    (yielding an empty report), not raised on.

    - ``seed_memory_id``: the input ``memory_id``.
    - ``seed_included_in_affected``: always False — the seed is the cause, not
      an affected effect, so it is EXCLUDED from ``affected_node_ids`` and from
      every count/grouping below.
    - ``affected_node_ids``: flat list of reachable node ids, seed excluded,
      in BFS discovery order.
    - ``affected_edges``: list of ``{"src", "dst"}`` edges whose endpoints are
      both within the affected subgraph (seed included as a possible endpoint,
      so seed->first-hop edges are surfaced).
    - ``nodes_by_type`` / ``counts_by_type``: grouping/count by semantic type.
    - ``total_affected``: len(affected_node_ids).
    """
    affected = affected_node_ids(graph, memory_id)
    nodes_by_type, counts_by_type = _group_by_type(graph, affected)
    affected_edges = _induced_edges(graph, set(affected) | {memory_id})

    return {
        "task_id": graph.task_id,
        "seed_memory_id": memory_id,
        "seed_included_in_affected": False,
        "affected_node_ids": affected,
        "affected_edges": affected_edges,
        "nodes_by_type": nodes_by_type,
        "counts_by_type": counts_by_type,
        "total_affected": len(affected),
    }


def _filter_after_cutoff(
    graph: DependencyGraph,
    raw_affected: List[str],
    cutoff_step_id: str,
    order_index: Dict[str, int],
) -> List[str]:
    """Keep only nodes occurring strictly AFTER ``cutoff_step_id`` in trace order.

    The temporal filter for a stale-and-failed seed (see :func:`resolve_seed_id`):
    ``raw_affected`` is the UNFILTERED forward closure of the stale memory node,
    which (correctly) reaches the post-mutation contamination via the memory's
    surviving support/cite edges, but ALSO (incorrectly) reaches the clean
    pre-mutation window via those same support edges. This drops the clean window
    by keeping only nodes whose trace position is past the failed mutation step.

    ``order_index`` maps step_id -> position in trace order. ``cutoff_position`` is
    that of ``cutoff_step_id``; if the cutoff step itself is not in the trace
    (unusual — the graph and trace disagree), we WARN and treat the cutoff as -1
    ("include everything", since we cannot place the boundary).

    Per node kind:

    * **step**: keep iff its own position > cutoff_position. A step missing from
      ``order_index`` (position defaulted to -1) cannot be confirmed to follow the
      cutoff, so it is dropped.
    * **memory**: locate its producing step via :func:`find_producing_step`. If
      one exists, keep iff the producing step's position > cutoff_position
      (a memory "occurs" when its producer ran). If NO producing step exists
      (a root memory reached via some other path), EXCLUDE it — there is no
      evidence it occurred after the cutoff, and unverified inclusion risks
      corrupting the affected set with something that may legitimately predate the
      fault (logged at DEBUG, never silently included).
    * **user_input**: a user_input node has NO incoming edges (it is a graph root
      that only ever initiates/cites forward), so it is never actually forward-
      reachable from a memory seed and this branch is effectively dead. Were one
      to appear, its trace position is not directly stored (only derivable via the
      step it initiates), so rather than reason about that awkward indirection we
      apply the same EXCLUDE-by-default rule as a root memory and log it — no
      positive evidence it follows the cutoff.
    """
    cutoff_position = order_index.get(cutoff_step_id)
    if cutoff_position is None:
        logger.warning(
            "cutoff step %s is not in steps_in_order; the graph and the supplied "
            "trace disagree. Treating the cutoff as -1 (keep everything reachable) "
            "since the temporal boundary cannot be placed.",
            cutoff_step_id,
        )
        cutoff_position = -1

    kept: List[str] = []
    for node_id in raw_affected:
        node = graph.node(node_id)
        if node is None:  # pragma: no cover - closure ids are graph nodes
            continue
        if node.node_kind == "step":
            if order_index.get(node_id, -1) > cutoff_position:
                kept.append(node_id)
        elif node.node_kind == "memory":
            producing = find_producing_step(graph, node_id)
            if producing is not None:
                if order_index.get(producing, -1) > cutoff_position:
                    kept.append(node_id)
            else:
                logger.debug(
                    "excluding root memory %s from stale seed's affected set: no "
                    "producing step, so no evidence it occurred after cutoff %s",
                    node_id,
                    cutoff_step_id,
                )
        elif node.node_kind == "user_input":
            logger.debug(
                "excluding user_input %s from stale seed's affected set: its trace "
                "position is not directly placeable relative to cutoff %s "
                "(exclude-by-default, same as a root memory)",
                node_id,
                cutoff_step_id,
            )
        # any other (unexpected) kind: leave out — no basis to place it.
    return kept


def compute_affected_subgraph(
    graph: DependencyGraph,
    faulty_memory_ids: List[str],
    provenance: Dict[str, Dict[str, Any]],
    steps_in_order: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the affected-subgraph report for ALL faulty seeds in one pass.

    This is the function run_ours.py and the real pipeline call: section 7's
    I/O table feeds it ``dependency_graph + faulty_memory_ids`` (plural; 1-3
    faults injected per task) PLUS the per-memory ``provenance`` dict from
    fault_provenance.py (run_ours.py's stage 2) AND ``steps_in_order`` (the
    ordered trace-step dicts, same shape ``find_stale_window`` consumes), needed
    to build the ``{step_id: position}`` lookup that places nodes in trace order
    for the stale-seed temporal filter.

    Each faulty memory is resolved to its seed via :func:`resolve_seed_id`, which
    returns either a PLAIN STR (the producing step, or the memory node as a last
    resort) or a ``(memory_id, cutoff_step_id)`` TUPLE for the stale-and-failed
    case. The two groups are processed differently:

    * **plain_seeds** (every plain-str resolution): handled as before — one
      combined forward BFS via ``affected_node_ids(graph, *plain_seeds)``, which
      unions their reachable sets at the BFS visited-set level (a node reachable
      from two seeds is visited once), de-duplicated and in BFS discovery order.
    * **stale_seeds** (every tuple resolution): processed INDIVIDUALLY (NOT
      unioned before filtering — each has its OWN cutoff). For each, the
      UNFILTERED closure of the memory node is computed, then
      :func:`_filter_after_cutoff` drops everything at or before that seed's
      cutoff step (the clean pre-mutation window the same BFS would otherwise
      wrongly include — see :func:`resolve_seed_id`). Each stale seed's filtered
      contribution is then UNIONED with the others (set union, to dedupe a node
      reachable from two stale seeds).

    The plain-seeds BFS result and the unioned stale-seeds filtered result are
    combined into the final ``affected_node_ids`` (de-duplicated; a node reachable
    from BOTH a plain and a stale seed appears once), preserving the plain BFS
    discovery order first, then any stale-only nodes.

    Returns a JSON-serializable dict (the affected_nodes.json payload):

    - ``seed_memory_ids``: the input list of ORACLE faulty seeds (plural,
      unchanged meaning — still the oracle fault ids for reporting).
    - ``resolved_seed_ids``: what reachability was ACTUALLY seeded from, one entry
      per faulty memory in the same order. A plain-str case is rendered as that
      str; a STALE case is rendered as a dict
      ``{"memory_id": ..., "cutoff_step_id": ...}`` (chosen over the raw tuple for
      JSON readability — a tuple would serialize as an anonymous 2-element list,
      whereas the dict labels which id is the seed and which is the cutoff
      boundary). This makes the seed resolution visible to a reader.
    - ``seed_included_in_affected``: a single bool — True if ANY resolved seed's
      cause node (a plain seed str, or a stale seed's memory_id) also turns up as
      an affected node. Seeds are still EXCLUDED from ``affected_node_ids`` and
      every count; this flag just surfaces the overlap.
    - ``affected_node_ids``: flat de-duplicated list of affected node ids, seeds
      excluded.
    - ``affected_edges``: list of ``{"src", "dst"}`` edges whose endpoints are
      both within the affected subgraph. The seed endpoints surfaced are the plain
      seed strs and each stale seed's MEMORY_ID (NOT its cutoff step — the cutoff
      is only a filter boundary, not a contamination root), so the first-hop edges
      shown leave the memory, not the failed mutation step.
    - ``nodes_by_type`` / ``counts_by_type``: grouping/count by semantic type over
      the final unioned set (each node once).
    - ``total_affected``: len(affected_node_ids).

    Missing or invalid resolved seeds are warned about and skipped (not raised on)
    by :func:`affected_node_ids`, so one bad id cannot kill the whole report.
    """
    # Resolve each oracle faulty memory to its seed (plain str OR stale tuple),
    # preserving order for the resolved_seed_ids report.
    resolved = [
        resolve_seed_id(mid, provenance.get(mid, {})) for mid in faulty_memory_ids
    ]

    plain_seeds: List[str] = [r for r in resolved if isinstance(r, str)]
    stale_seeds: List[Tuple[str, str]] = [r for r in resolved if isinstance(r, tuple)]

    order_index = {
        step.get("step_id"): i for i, step in enumerate(steps_in_order)
    }

    # Plain seeds: one combined BFS, unioned at the visited-set level.
    plain_affected: List[str] = (
        affected_node_ids(graph, *plain_seeds) if plain_seeds else []
    )

    # Stale seeds: each filtered against its OWN cutoff, then unioned together.
    stale_union: set = set()
    for memory_id, cutoff_step_id in stale_seeds:
        raw_affected = affected_node_ids(graph, memory_id)
        filtered = _filter_after_cutoff(
            graph, raw_affected, cutoff_step_id, order_index
        )
        stale_union.update(filtered)

    # Combine: plain BFS order first, then any stale-only nodes (de-duplicated).
    affected: List[str] = list(plain_affected)
    seen = set(plain_affected)
    for nid in raw_stale_order(graph, stale_seeds, order_index, stale_union):
        if nid not in seen:
            seen.add(nid)
            affected.append(nid)
    affected_set = set(affected)

    nodes_by_type, counts_by_type = _group_by_type(graph, affected)

    # Cause nodes for overlap/edge-scope purposes: plain seed strs + each stale
    # seed's MEMORY_ID (the contamination root), NOT the cutoff step (just a
    # filter boundary).
    cause_seed_set = set(plain_seeds) | {mid for mid, _ in stale_seeds}
    seed_included = bool(cause_seed_set & affected_set)

    # Induced edges: both endpoints affected, OR a cause seed -> an affected node,
    # so the seed's first-hop edges surface (from the memory in the stale case).
    affected_edges = _induced_edges(graph, affected_set | cause_seed_set)

    return {
        "task_id": graph.task_id,
        "seed_memory_ids": list(faulty_memory_ids),
        "resolved_seed_ids": [_render_resolved_seed(r) for r in resolved],
        "seed_included_in_affected": seed_included,
        "affected_node_ids": affected,
        "affected_edges": affected_edges,
        "nodes_by_type": nodes_by_type,
        "counts_by_type": counts_by_type,
        "total_affected": len(affected),
    }


def raw_stale_order(
    graph: DependencyGraph,
    stale_seeds: List[Tuple[str, str]],
    order_index: Dict[str, int],
    stale_union: set,
) -> List[str]:
    """Deterministic ordering of the unioned stale-seed nodes for the final list.

    The stale contributions were collected into a set (to dedupe across seeds),
    losing order. Re-impose a stable order by trace position (step nodes and
    memories placed by their producing step), with un-placeable nodes sorted by
    id at the end, so the appended tail of ``affected_node_ids`` is deterministic.
    """
    def _pos(node_id: str) -> int:
        node = graph.node(node_id)
        if node is not None and node.node_kind == "memory":
            producing = find_producing_step(graph, node_id)
            if producing is not None:
                return order_index.get(producing, len(order_index))
            return len(order_index)
        return order_index.get(node_id, len(order_index))

    return sorted(stale_union, key=lambda nid: (_pos(nid), nid))


def _render_resolved_seed(
    resolved: Union[str, Tuple[str, str]]
) -> Union[str, Dict[str, str]]:
    """Render a resolved seed for the JSON payload (see compute_affected_subgraph).

    Plain-str cases pass through unchanged; a stale ``(memory_id, cutoff_step_id)``
    tuple becomes ``{"memory_id": ..., "cutoff_step_id": ...}`` — the dict form is
    chosen over the raw tuple so the JSON labels which id is the seed and which is
    the temporal cutoff, instead of emitting an anonymous 2-element list.
    """
    if isinstance(resolved, tuple):
        memory_id, cutoff_step_id = resolved
        return {"memory_id": memory_id, "cutoff_step_id": cutoff_step_id}
    return resolved


def _group_by_type(
    graph: DependencyGraph, affected: List[str]
) -> tuple[Dict[str, List[str]], Dict[str, int]]:
    """Group affected ids by semantic type and count them.

    Returns ``(nodes_by_type, counts_by_type)`` over whatever id list it is
    handed — for the multi-seed caller that is the already-unioned set, so each
    node is grouped/counted exactly once. Ids within each bucket are sorted for
    deterministic output.
    """
    nodes_by_type: Dict[str, List[str]] = {}
    for nid in affected:
        nodes_by_type.setdefault(_node_type(graph.nodes[nid]), []).append(nid)
    for ids in nodes_by_type.values():
        ids.sort()
    counts_by_type = {t: len(ids) for t, ids in nodes_by_type.items()}
    return nodes_by_type, counts_by_type


def _induced_edges(
    graph: DependencyGraph, edge_scope: set
) -> List[Dict[str, str]]:
    """Edges whose BOTH endpoints lie in ``edge_scope`` (seeds included).

    ``edge_scope`` is the affected set unioned with the seed ids, so seed->
    first-hop edges are surfaced. Iterated in sorted src/dst order for a
    deterministic edge list.

    DESIGN CHOICE (chain edges in the DISPLAYED list): this calls
    ``graph.successors(src)`` with NO edge-type exclusion, so a ``chain`` edge
    IS shown whenever it connects two nodes that are BOTH already in
    ``edge_scope`` (i.e. both independently affected via real evidentiary
    paths). The ``{"chain"}`` exclusion is applied to the REACHABILITY
    computation ONLY (see :func:`affected_node_ids`) — chain never makes a node
    affected. But once two nodes are affected for other reasons, showing the
    incidental chain edge between them causes no harm and gives a reader
    inspecting the JSON slightly more context about the execution order; a chain
    edge can never appear here unless BOTH its endpoints earned their place in
    ``edge_scope`` without it. The edge list is purely descriptive (it drives no
    downstream reachability or support decision), so the display staying
    edge-type-complete while reachability stays chain-free is consistent, not
    contradictory.
    """
    affected_edges: List[Dict[str, str]] = []
    for src in sorted(edge_scope):
        for dst in sorted(graph.successors(src)):
            if dst in edge_scope:
                affected_edges.append({"src": src, "dst": dst})
    return affected_edges


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_graph(path: str) -> DependencyGraph:
    with open(path, "r", encoding="utf-8") as fh:
        return DependencyGraph.from_dict(json.load(fh))


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _render_resolved_cli(resolved: Union[str, Dict[str, str]]) -> str:
    """One-line CLI rendering of a resolved seed entry (str or stale dict)."""
    if isinstance(resolved, dict):
        return f"{resolved['memory_id']}@>{resolved['cutoff_step_id']}"
    return resolved


def _write_json(payload: Dict[str, Any], path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the downstream-affected subgraph for one or more "
                    "faulty seed memory nodes and write affected_nodes.json."
    )
    parser.add_argument(
        "--graph", required=True,
        help="dependency graph JSON path (DependencyGraph.to_dict output)",
    )
    parser.add_argument(
        "--faulty_memory_ids", required=True, nargs="+",
        help="one or more seed memory_ids (e.g. m_f001 m_f002)",
    )
    parser.add_argument(
        "--provenance", required=True,
        help="provenance.json path (fault_provenance output / run_ours.py's "
             "stage 2) used to resolve each faulty memory to its seed "
             "(failed-mutation step / producing step / memory node)",
    )
    parser.add_argument(
        "--steps", required=True,
        help="trace steps JSONL path (in trace order). Supplies the "
             "{step_id: position} lookup compute_affected_subgraph needs to "
             "place nodes in trace order for the stale-seed temporal filter. "
             "Filter to a single task with --task_id when the file holds many.",
    )
    parser.add_argument(
        "--task_id", default=None,
        help="optional task_id to filter --steps to (matches graph.task_id); "
             "omit if --steps already holds only this task's steps",
    )
    parser.add_argument(
        "--output", required=True, help="affected_nodes.json output path",
    )
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    with open(args.provenance, "r", encoding="utf-8") as fh:
        provenance = json.load(fh)
    steps_in_order = _load_jsonl(args.steps)
    if args.task_id is not None:
        steps_in_order = [
            s for s in steps_in_order if s.get("task_id") == args.task_id
        ]
    # is_memory_id / is_step_id / existence checks (and the warn+skip for bad
    # seeds) happen inside compute_affected_subgraph -> affected_node_ids, so
    # the CLI just hands the whole list, the provenance dict, and the ordered
    # steps through.
    report = compute_affected_subgraph(
        graph, args.faulty_memory_ids, provenance, steps_in_order
    )
    _write_json(report, args.output)

    seeds = " ".join(report["seed_memory_ids"])
    resolved = " ".join(
        _render_resolved_cli(r) for r in report["resolved_seed_ids"]
    )
    print(f"=== affected subgraph for seeds [{seeds}] "
          f"(task {graph.task_id}) ===")
    print(f"  resolved seed ids:    [{resolved}]")
    print(f"  total affected nodes: {report['total_affected']}")
    print(f"  seed reachable from another seed: "
          f"{report['seed_included_in_affected']}")
    print("  counts by type:")
    for t in sorted(report["counts_by_type"]):
        print(f"    {t:18s} {report['counts_by_type'][t]}")
    print(f"  affected edges: {len(report['affected_edges'])}")
    print(f"  wrote: {args.output}")


if __name__ == "__main__":
    _main()
