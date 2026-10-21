# methods/ours/support_checker.py
"""Independent-support check (spec section 8.1).

Pure computation, no repair decisions
=====================================
This module answers whether affected steps retain independent backing and,
separately, whether active derived-memory candidates have a fault-free trusted
provenance path. :func:`build_support_report` returns both verdict families in
one structured payload. It does NOT decide rollback actions, mutate the graph,
or write any repair. That boundary
mirrors ``affected_subgraph.py``'s "pure reachability, no pruning" split: this
module is consulted by ``rollback_optimizer.py``, which is the one that turns
these verdicts into actions — invalidate for a claim specifically,
replay/preserve/suspicious for everything else, depending on rollback policy.

What it consumes (does NOT recompute)
=====================================
* a :class:`DependencyGraph` from ``graph_schema`` / ``build_dependency_graph``.
* the ``affected_node_ids`` set produced by ``affected_subgraph.py``'s
  ``compute_affected_subgraph()`` — every node downstream of the faulty
  memory/memories. That is the candidate set of nodes that MIGHT need acting
  on; this module narrows it to the ones that have lost trusted backing, and
  the same set doubles as the "these memories are tainted, don't trust them as
  support" set when validating a node's evidence.

What counts as a checkable node
===============================
A node is checkable iff its ``GraphNode.semantic_type`` is one of the eight
``_CHECKABLE_SEMANTIC_TYPES``:

  memory_read, claim, plan, final_answer,
  memory_write, memory_delete, memory_update, memory_consolidate

The MECHANISM is identical for all eight: walk this node's ``used_ids``
citations backward (via support/cite/derive edges) and check whether they
trace to clean, sufficiently-trusted evidence. What "support" MEANS shades
slightly by type, but the walk does not:
  - claim: does the asserted claim have clean backing?
  - plan / final_answer: did the decision/answer rely on clean inputs?
  - memory_read: does what THIS STEP RETRIEVED (its own ``used_ids`` — what it
    pulled from the store) trace back to clean evidence? Not "is the read step
    itself contaminated" in a vaguer sense; specifically, walk ITS used_ids
    the same way a claim's are walked.
  - memory_write / memory_update / memory_consolidate: was the new/updated
    memory's content grounded in clean evidence?
  - memory_delete: was the decision to delete grounded in clean evidence (per
    the earlier project decision that deletes should also carry used_ids
    citations, not just a bare target id)?

Deliberately EXCLUDED:
  - ``tool_action`` — its correctness is fully determined by the plan it is
    chained to; there is no independent evidentiary question beyond "did it
    follow its plan correctly", so checking it separately would be redundant
    with checking the plan.
  - ``tool_observation`` — pure output data, asserts nothing, was never a
    candidate for this check.

The spec defines no non-critical tier, so EVERY checkable node is treated as
critical — there is no "this node is minor, skip it" path here. This
assumption is stated rather than silently encoded.

How support is walked
=====================
A node's supporting evidence is found by walking BACKWARD from it over BOTH
``support``-type edges (``graph.predecessor_edges(node_id)`` filtered to
``"support" in etypes`` — these run memory->step, so the node's support
predecessors are the memories it leans on) AND ``cite``-type edges (a cited
claim step, tool_observation step, or user_input node -> the citing step). A
checkable node's ``used_ids`` is mixed-type, so the memory entries surface as
``support`` predecessors and the non-memory entries (claim / tool_observation /
user_input) as ``cite`` predecessors — both are candidate evidence. A piece of
evidence can itself be backed by further evidence, so we recurse through its own
backward ``support``/``derive``/``cite`` predecessors (a ``derive`` edge runs
ancestor-memory -> derived-memory, so its backward predecessors are the ancestor
memories the support was derived from). The full evidentiary chain is
considered, not just the immediate parent. We deliberately do NOT follow
``supersede`` edges in this hop: a supersede edge means a memory was *replaced*,
not that it is *evidence for* the claim. Each pair's labels are a SET, so
membership is tested with set intersection rather than equality. A ``visited``
set guards the walk so a cycle (however unlikely) terminates cleanly instead of
infinite-looping.

A node's OWN ``invalidated_memory_ids`` is EXCLUDED from its evidence frontier
(computed once at the top of the walk and applied at every hop): for
memory_delete/update/consolidate steps, ``used_ids`` frequently also names the
very memory the step is retiring (the LLM cites "what it's acting on"), but the
mutation TARGET is not evidentiary backing — a memory must never get to vouch
for its own retirement.

When a piece of supporting evidence is "valid"
==============================================
A piece of evidence counts as real, trustworthy backing iff:
  (a) it is NOT a ``user_input`` node — user input is the triggering
      instruction, not independently verifiable evidence, so it never counts
      as trusted support no matter what else is true of it; AND
  (b) it is NOT in the affected/faulty set passed in; AND
  (c) if ``use_trust_score`` is True: its ``raw.get("trust_score", 0.0)`` is
      ``>= trust_threshold``. (When ``use_trust_score`` is False this condition
      is skipped — that is the "Ours w/o trust score" ablation.)
A missing ``trust_score`` is treated as ``0.0`` (untrusted), never raised on —
faulty/injected memories may not carry one.

Support is lenient by design
============================
A node survives if ANY independent, trusted evidentiary path exists: at least
one supporting memory found in the walk is valid. A node with ZERO supporting
memories is regarded as valid (survives). (An earlier ``"strict"`` mode —
requiring ALL evidence to be clean — was removed: it was found to compute the
same thing ``affected_subgraph.py``'s plain reachability already computes, just
via a different mechanism, so it added no information as a separate runtime
choice. Lenient is the one behavior that does something reachability cannot:
rescue a node that is technically downstream of a fault but has OTHER,
independent clean evidence.)

The "mark everything unsupported" ablation lives in the CALLER, not here
==========================================================================
There is deliberately NO flag to "skip the check and act on everything".
This module only computes verdicts; the pipeline caller decides what action
follows. The "Ours w/o independent support check" ablation does not invoke
either algorithm: it emits false step verdicts and false/empty-path derived
memory verdicts directly. Do not add an ``always_invalidate`` option here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Set, TypedDict

try:  # importable both as ``methods.ours.support_checker`` and as a script
    from .graph_schema import DependencyGraph, NodeKind
except ImportError:  # pragma: no cover - direct-script fallback
    from graph_schema import DependencyGraph, NodeKind

logger = logging.getLogger(__name__)

# Edge-type labels this module keys off. Sourced as plain strings (the values
# graph_schema.EdgeType stores on edges) since DependencyGraph stores each
# edge's labels as a SET of .value strings in its edge maps, so membership is
# tested with ``in``/set-intersection rather than ``==``.
_SUPPORT = "support"
_DERIVE = "derive"
# cite: a NON-memory used_ids citation (claim step, tool_observation step, or a
# user_input node -> the citing step). FIX A — the walk follows these too, not
# just memory-sourced "support", so a checkable node's citation of a claim /
# tool_observation / user_input is no longer invisible to the support check.
_CITE = "cite"
# Labels followed when recursing into a piece of evidence's own backing. Now
# includes "cite" (FIX A) so a cited step/user_input's own citations are walked,
# alongside "support" (the evidence's memories) and "derive" (the ancestors it
# was derived from). We deliberately STILL do NOT include "supersede": a
# supersede edge means "this memory was replaced by", not "is evidence for". A
# node citing a since-superseded memory should be judged on whether THAT memory
# was valid, not by walking forward into its replacement.
_EVIDENCE_HOP_LABELS = frozenset({_SUPPORT, _DERIVE, _CITE})
_PRODUCE = "produce"

# The semantic_type a claim node carries (graph_schema.SemanticType.CLAIM).
# Kept as a constant: it no longer gates the filter alone (see
# _CHECKABLE_SEMANTIC_TYPES), but is still referenced for the claim-specific
# downstream consequence the caller applies (invalidate a claim vs.
# replay/preserve/suspicious for the other checkable types).
_CLAIM_SEMANTIC_TYPE = "claim"

# The eight semantic types this module evaluates support for. The walk
# mechanism is identical for every one of them (see the module docstring);
# only what "support" MEANS shades by type. tool_action (redundant with its
# plan) and tool_observation (pure output, asserts nothing) are deliberately
# excluded.
_CHECKABLE_SEMANTIC_TYPES = frozenset({
    "claim", "plan", "final_answer",
    "memory_write", "memory_delete", "memory_update", "memory_consolidate",
})

# Same eight types as a tuple, fixing a stable, readable order for the CLI's
# per-type breakdown (a frozenset has no order). The set above is the source
# of truth for membership; this just orders the printout.
_CHECKABLE_TYPE_ORDER = (
    "claim", "plan", "final_answer", "memory_read",
    "memory_write", "memory_update", "memory_consolidate", "memory_delete",
)


class DerivedMemorySupportVerdict(TypedDict):
    supported: bool
    support_path: List[str]


class SupportCheckReport(TypedDict):
    step_verdicts: Dict[str, bool]
    derived_memory_verdicts: Dict[str, DerivedMemorySupportVerdict]

# --------------------------------------------------------------------------- #
# Support-chain walk
# --------------------------------------------------------------------------- #

def _collect_supporting_evidence(
    graph: DependencyGraph, node_id: str
) -> Set[str]:
    """Collect node_id's DIRECT backward evidentiary predecessors only.

    Returns node_id's immediate "support" (memory->step) and "cite"
    (claim/tool_observation/user_input->step) predecessors — ONE HOP ONLY,
    no recursion into what backs that evidence in turn. EXCLUDES any
    predecessor that also appears in node_id's OWN invalidated_memory_ids
    (FIX C — the deletion/mutation target cited in used_ids is not
    evidence, regardless of whether it happens to be faulty). Does NOT
    follow "derive" or "supersede" at all (those only ever mattered for
    the recursive hop, which no longer exists).

    Returns the set of direct evidence node ids found (node_id itself is
    never included). NOTE: invalidated_memory_ids is read here only to
    EXCLUDE node_id's own retirement target from its frontier — this
    function never treats invalidated_memory_ids as a source of edges to
    walk; it only ever follows used_ids-derived support/cite edges.
    """
    self_invalidated: Set[str] = set(
        (graph.node(node_id).raw or {}).get("invalidated_memory_ids", [])
    ) if graph.node(node_id) is not None else set()

    # Direct evidence only: node_id's backward "support" (memory->step) and
    # "cite" (claim/tool_observation/user_input->step) predecessors. etypes
    # is the SET of labels on each predecessor pair, so we test membership.
    # FIX C drops any predecessor named in this node's OWN
    # invalidated_memory_ids — that is the mutation target, not backing.
    supporting: Set[str] = {
        src
        for src, etypes in graph.predecessor_edges(node_id).items()
        if (etypes & {_SUPPORT, _CITE}) and src not in self_invalidated
    }
    if not supporting:
        logger.debug(
            "node %s has zero direct support/cite predecessors "
            "(after self-invalidated exclusion)", node_id
        )

    return supporting

def _evidence_is_valid(
    graph: DependencyGraph,
    evidence_id: str,
    faulty_affected_ids: Set[str],
    use_trust_score: bool,
    trust_threshold: float,
) -> bool:
    """True if ``evidence_id`` is real, trustworthy backing for a node.

    FIX B — a ``user_input`` node is NEVER trusted support: it is the
    triggering instruction, not independently verifiable evidence the way a
    memory's trust_score/fault-status is, so it is invalid regardless of any
    other property (and short-circuits before the faulty-set / trust-score
    checks). For all other evidence it is valid iff (a) it is not in
    ``faulty_affected_ids`` and (b), only when ``use_trust_score`` is True, its
    ``raw.trust_score`` (defaulting to 0.0 for a missing score — never raised
    on) is ``>= trust_threshold``.
    """
    node = graph.node(evidence_id)
    if node is not None and node.node_kind == NodeKind.USER_INPUT.value:
        return False
    if evidence_id in faulty_affected_ids:
        return False
    if node is not None and node.node_kind != "memory":
        return True  # step-kind evidence: no trust_score concept applies
    if not use_trust_score:
        return True
    raw = node.raw if node is not None else {}
    trust_score = raw.get("trust_score", 0.0)
    return trust_score >= trust_threshold


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def has_trusted_support(
    graph: DependencyGraph,
    claim_id: str,
    faulty_affected_ids: Set[str],
    use_trust_score: bool = False,
    trust_threshold: float = 0.8,
) -> bool:
    """True if ``claim_id`` has at least one independent trusted support path,
    or has no support memories at all.

    Raises ``KeyError`` if ``claim_id`` is not a node in ``graph`` — a bad id
    from the caller should be loud; an evidentiary gap (a claim with no support)
    should NOT raise, it just returns True.
    """
    if graph.node(claim_id) is None:
        # Caller bug — be loud. (Contrast with the silent skips below.)
        raise KeyError(
            f"claim_id {claim_id!r} is not a node in graph {graph.task_id!r}"
        )

    supporting = _collect_supporting_evidence(graph, claim_id)
    if not supporting:
        # No supporting evidence: nothing to be tainted, so the claim survives.
        logger.debug(
            "claim %s has no supporting evidence; treating as valid", claim_id
        )
        return True

    # Lenient by design: survive if AT LEAST ONE piece of evidence is valid.
    return any(
        _evidence_is_valid(
            graph, ev_id, faulty_affected_ids, use_trust_score, trust_threshold
        )
        for ev_id in supporting
    )


def check_all_nodes(
    graph: DependencyGraph,
    affected_node_ids: List[str],
    use_trust_score: bool = False,
    trust_threshold: float = 0.8,
) -> Dict[str, bool]:
    """Run :func:`has_trusted_support` for every checkable node in
    ``affected_node_ids``.

    Filters ``affected_node_ids`` to the eight ``_CHECKABLE_SEMANTIC_TYPES``
    internally and silently skips any node of another type (``tool_action``,
    ``tool_observation``, raw memory nodes, …) — callers should not have to
    pre-filter. The faulty/affected set used to judge support validity IS
    ``affected_node_ids`` (a memory downstream of the fault cannot be trusted
    backing).

    Returns ``{node_id: has_support}``. A node mapped to True still has clean,
    trusted backing despite being downstream of the fault; a node mapped to
    False has lost it. This module stops at the verdict —
    ``rollback_optimizer.py`` decides the action (invalidate a claim,
    replay/preserve/suspicious for the rest, per rollback policy).
    """
    faulty_affected_ids: Set[str] = set(affected_node_ids)
    results: Dict[str, bool] = {}
    for node_id in affected_node_ids:
        node = graph.node(node_id)
        if node is None:
            # An affected id with no node — skip silently (an evidentiary/
            # bookkeeping gap, not a caller bug worth raising on here).
            logger.debug("affected id %s has no node in graph; skipping", node_id)
            continue
        if node.semantic_type not in _CHECKABLE_SEMANTIC_TYPES:
            # Non-checkable affected node (tool_action / tool_observation /
            # raw memory / …) — not our concern; this function filters itself.
            continue
        results[node_id] = has_trusted_support(
            graph,
            node_id,
            faulty_affected_ids,
            use_trust_score=use_trust_score,
            trust_threshold=trust_threshold,
        )
    return results


def find_derived_memory_candidates(
    graph: DependencyGraph,
    faulty_memory_ids: List[str],
    affected_node_ids: List[str],
) -> List[str]:
    """Return active, downstream memories with structural derivation provenance.

    A memory has derivation provenance when it has either an incoming ``derive``
    lineage edge or an incoming ``produce`` edge from a memory-producing step.
    The latter is the graph representation used for generated memories whose
    ``derived_from`` list is empty.  Content and clean ground truth are never
    consulted.
    """
    faulty = set(faulty_memory_ids)
    candidates: List[str] = []
    for node_id in set(affected_node_ids):
        node = graph.node(node_id)
        if node is None or node.node_kind != NodeKind.MEMORY.value:
            continue
        if node_id in faulty:
            continue
        status = node.memory_status or (node.raw or {}).get("status") or "active"
        if status != "active":
            continue
        has_provenance = any(
            labels & {_DERIVE, _PRODUCE}
            for labels in graph.predecessor_edges(node_id).values()
        )
        if has_provenance:
            candidates.append(node_id)
    return sorted(candidates)


def has_fault_free_memory_support(
    graph: DependencyGraph,
    memory_id: str,
    faulty_memory_ids: List[str],
    affected_node_ids: List[str],
    use_trust_score: bool = False,
    trust_threshold: float = 0.8,
) -> DerivedMemorySupportVerdict:
    """Find one fault-free trusted evidence path for a derived memory.

    The candidate's producing step is its provenance entry point.  Beyond that
    single ``produce`` hop, recursion follows only support/cite/derive edges.
    Paths are returned trusted-root first and are deterministic.
    """
    if graph.node(memory_id) is None:
        raise KeyError(f"memory_id {memory_id!r} is not a node in graph {graph.task_id!r}")

    faulty = set(faulty_memory_ids)
    affected = set(affected_node_ids)
    contaminated_candidates = set(
        find_derived_memory_candidates(graph, faulty_memory_ids, affected_node_ids)
    )

    def evidence_predecessors(node_id: str, *, candidate: bool = False) -> List[str]:
        allowed = set(_EVIDENCE_HOP_LABELS)
        if candidate:
            allowed.add(_PRODUCE)
        return sorted(
            src
            for src, labels in graph.predecessor_edges(node_id).items()
            if labels & allowed
        )

    def trusted_root(node_id: str, predecessors: List[str]) -> bool:
        node = graph.node(node_id)
        if node is None or node_id in faulty:
            return False
        if node.node_kind == NodeKind.USER_INPUT.value:
            return True
        if node.semantic_type == "tool_observation":
            return node_id not in affected
        if node.node_kind == NodeKind.MEMORY.value:
            status = node.memory_status or (node.raw or {}).get("status") or "active"
            if status != "active" or node_id in contaminated_candidates:
                return False
            # A memory root has no incoming derivation or generation provenance.
            has_memory_provenance = any(
                labels & {_DERIVE, _PRODUCE}
                for labels in graph.predecessor_edges(node_id).values()
            )
            if predecessors or has_memory_provenance:
                return False
            if not use_trust_score:
                return True
            return (node.raw or {}).get("trust_score", 0.0) >= trust_threshold
        return node_id not in affected and not predecessors

    def walk(node_id: str, visited: Set[str], *, candidate: bool = False) -> List[str]:
        if node_id in visited or node_id in faulty:
            return []
        next_visited = set(visited)
        next_visited.add(node_id)
        predecessors = evidence_predecessors(node_id, candidate=candidate)
        if trusted_root(node_id, predecessors):
            return [node_id]
        # Prefer a direct trusted root before exploring longer branches.  This
        # keeps explanations minimal while sorted predecessor order keeps ties
        # deterministic.
        for predecessor in predecessors:
            predecessor_predecessors = evidence_predecessors(predecessor)
            if predecessor not in next_visited and trusted_root(
                predecessor, predecessor_predecessors
            ):
                return [predecessor, node_id]
        for predecessor in predecessors:
            path = walk(predecessor, next_visited)
            if path:
                return path + [node_id]
        return []

    support_path = walk(memory_id, set(), candidate=True)
    return {"supported": bool(support_path), "support_path": support_path}


def build_support_report(
    graph: DependencyGraph,
    faulty_memory_ids: List[str],
    affected_node_ids: List[str],
    use_trust_score: bool = False,
    trust_threshold: float = 0.8,
) -> SupportCheckReport:
    """Run the unchanged step checker plus derived-memory support checking."""
    # affected_subgraph payloads exclude their seeds.  The pipeline historically
    # supplied the seeds alongside the affected ids to the step checker; make
    # that contract explicit here so the standalone CLI behaves identically.
    step_taint_ids = list(affected_node_ids)
    step_taint_ids.extend(
        memory_id
        for memory_id in faulty_memory_ids
        if memory_id not in set(step_taint_ids)
    )
    step_verdicts = check_all_nodes(
        graph,
        step_taint_ids,
        use_trust_score=use_trust_score,
        trust_threshold=trust_threshold,
    )
    candidates = find_derived_memory_candidates(
        graph, faulty_memory_ids, affected_node_ids
    )
    derived_verdicts = {
        memory_id: has_fault_free_memory_support(
            graph,
            memory_id,
            faulty_memory_ids,
            affected_node_ids,
            use_trust_score=use_trust_score,
            trust_threshold=trust_threshold,
        )
        for memory_id in candidates
    }
    return {
        "step_verdicts": step_verdicts,
        "derived_memory_verdicts": derived_verdicts,
    }


# --------------------------------------------------------------------------- #
# CLI (mirrors affected_subgraph.py: argparse, small _load helpers, a printed
# summary block)
# --------------------------------------------------------------------------- #

def _load_graph(path: str) -> DependencyGraph:
    with open(path, "r", encoding="utf-8") as fh:
        return DependencyGraph.from_dict(json.load(fh))


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(payload: Dict[str, Any], path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent-support check: decide which "
                    "downstream nodes still have trusted backing once the "
                    "faulty/affected memories are discounted. Verdicts only — "
                    "no repair decisions."
    )
    parser.add_argument(
        "--graph", required=True,
        help="dependency_graph.json path (DependencyGraph.to_dict output)",
    )
    parser.add_argument(
        "--affected_nodes", required=True,
        help="affected_nodes.json path from affected_subgraph.py; its "
             "'affected_node_ids' field is read",
    )
    parser.add_argument(
        "--faulty_memory_ids", required=True, nargs="+",
        help="source faulty-memory ids (excluded from derived candidates/paths)",
    )
    # store_true/store_false pair on the same dest; default use_trust_score=False.
    parser.add_argument(
        "--use_trust_score", dest="use_trust_score", action="store_true",
        help="require supporting memories to clear --trust_threshold",
    )
    parser.add_argument(
        "--no_trust_score", dest="use_trust_score", action="store_false",
        help="ignore trust scores ('Ours w/o trust score' ablation)",
    )
    parser.set_defaults(use_trust_score=False)
    parser.add_argument(
        "--trust_threshold", type=float, default=0.8,
        help="minimum trust_score for a support to count (default: 0.8)",
    )
    parser.add_argument(
        "--output", default=None,
        help="optional path to write the structured support report; if omitted, only "
             "the summary is printed and no file is written",
    )
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    affected_payload = _load_json(args.affected_nodes)
    affected_ids = affected_payload.get("affected_node_ids", [])

    report = build_support_report(
        graph,
        args.faulty_memory_ids,
        affected_ids,
        use_trust_score=args.use_trust_score,
        trust_threshold=args.trust_threshold,
    )

    results = report["step_verdicts"]
    total = len(results)
    n_supported = sum(1 for v in results.values() if v)
    n_unsupported = total - n_supported

    # Per-semantic-type breakdown: for each checkable type actually present in
    # the results, count how many were checked and the with/without-support
    # split. Built from the graph's own node lookup so the type matches the
    # filter used in check_all_nodes.
    per_type: Dict[str, List[int]] = {}  # type -> [checked, with, without]
    for node_id, has_support in results.items():
        node = graph.node(node_id)
        stype = node.semantic_type if node is not None else "<unknown>"
        bucket = per_type.setdefault(stype, [0, 0, 0])
        bucket[0] += 1
        bucket[1 if has_support else 2] += 1

    print(f"=== independent-support check (task {graph.task_id}) ===")
    print(f"  use_trust_score: {args.use_trust_score}")
    print(f"  trust_threshold: {args.trust_threshold}")
    print(f"  nodes checked:            {total}")
    print(f"  with trusted support:     {n_supported}  (keep — clean backing)")
    print(
        f"  without trusted support:  {n_unsupported}  "
        "(act per rollback policy)"
    )
    derived = report["derived_memory_verdicts"]
    derived_supported = sum(1 for verdict in derived.values() if verdict["supported"])
    print(f"  derived candidates:       {len(derived)}")
    print(f"  supported derived:        {derived_supported}")
    print(f"  unsupported derived:      {len(derived) - derived_supported}")
    if per_type:
        print("  per semantic type (checked / with support / without):")
        # Stable, readable order: the canonical checkable order, then any
        # straggler types (e.g. "<unknown>") last.
        ordered = [t for t in _CHECKABLE_TYPE_ORDER if t in per_type]
        ordered += [t for t in per_type if t not in _CHECKABLE_TYPE_ORDER]
        width = max(len(t) for t in ordered)
        for stype in ordered:
            checked, with_s, without_s = per_type[stype]
            print(
                f"    {stype:<{width}}  {checked} checked, "
                f"{with_s} with support, {without_s} without"
            )
    if args.output:
        _write_json(report, args.output)
        print(f"  wrote: {args.output}")


if __name__ == "__main__":
    _main()
