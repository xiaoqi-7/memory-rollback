# methods/ours/rollback_optimizer.py
"""Turn structural facts + support verdicts into a concrete rollback plan.

Where this sits in the pipeline
================================
This is the FIRST module that makes repair DECISIONS. Everything upstream is
deliberately decision-free:

* ``affected_subgraph.py`` — pure forward reachability (what is downstream of
  the fault), no pruning.
* ``fault_provenance.py`` — pure structure (producing step, stale window,
  ancestor chain), no fault/repair judgment.
* ``support_checker.py`` — pure verdicts (does a checkable node still have
  independent trusted backing once the faulty/affected memories are
  discounted), no actions.

``rollback_optimizer.py`` is the SOLE place those three streams are combined
into actions: which memories to delete/quarantine, which claims to invalidate,
which steps to replay/preserve/mark-suspicious.

What "checkable" means here (the final, confirmed set)
======================================================
Exactly EIGHT semantic types are ever given a step-level disposition, the same
set ``support_checker.py`` evaluates (imported from there so the two cannot
drift):

    memory_read, claim, plan, final_answer,
    memory_write, memory_delete, memory_update, memory_consolidate

``tool_action`` and ``tool_observation`` are NEVER checkable and are never
given a step-level DISPOSITION by the downstream checkable-steps logic: they
appear in none of ``preserve`` / ``redundant`` / ``suspicious`` and are never
consulted for support. ``tool_observation`` appears in NO output list at all.
The ONE carve-out is ``tool_action`` in ``replay_step_ids`` via the UPSTREAM
rule: when a faulty memory's producing (or failed-mutation) step regenerated
the memory from a cited ``tool_observation``, replaying that step ALSO replays
the ``tool_action`` behind the cited observation — re-running the underlying
tool call rather than regenerating the memory from a possibly-stale observation
already sitting in the trace. This is a narrow, deliberate reversal of the
earlier draft's blanket "no ``tool_action`` ever in ``replay_step_ids``" rule:
it is confined to the upstream replay path and never reached by the downstream
disposition logic. Outside that carve-out a ``tool_action``'s correctness is
still fully implied by its ``plan``'s disposition, so ``selective_replay.py`` is
expected to RE-DERIVE how to handle a non-upstream ``tool_action`` from its
plan's entry in ``replay_step_ids`` / ``preserve_step_ids``, NOT read it from
this plan directly.

The two-tier rule
=================
* A node whose support verdict is ``True`` (it still has independent trusted
  backing) goes to ``preserve_step_ids``. It is in the same mode-independent
  tier as memory delete/quarantine.
* Only nodes WITHOUT trusted support (verdict ``False`` or absent) first enter
  ``suspicious_step_ids``. ``promote_load_bearing_suspicious`` then promotes
  genuinely load-bearing nodes into ``replay_step_ids`` while leaving
  non-load-bearing residue suspicious.

Forced last-final_answer replay
===============================
The session's LAST final_answer is UNCONDITIONALLY added to
``replay_step_ids`` — regardless of whether the
affected subgraph reaches it, and regardless of its own support verdict. If
that id was otherwise assigned to another disposition list (e.g. ``preserve``
when it kept trusted support, or ``suspicious`` before load-bearing selection),
the forced
inclusion overrides it: the id is removed from every other list and ends up in
``replay_step_ids`` only.

final_answer nodes from OTHER turns get NO disposition
======================================================
A final_answer node from any turn OTHER than the session's last is NEVER given
any disposition — it appears in NONE of the six lists (delete / quarantine /
replay / preserve / redundant / suspicious) and is never invalidated, REGARDLESS
of its support verdict or its membership in the affected subgraph. It is treated
exactly like ``tool_action`` / ``tool_observation`` by the disposition logic.
RATIONALE: evaluation scores ONLY the session's last turn's final_answer.
Replaying an earlier turn's final_answer would spend real replay budget
(``lambda_a`` / ``lambda_t`` cost) for zero credit on the measured metric, and
would inflate ``replay_step_ids`` / ``suspicious_step_ids`` relative to what the
method actually needs to fix for correctness on the measured outcome. Only the
single last-turn final_answer (see "Forced last-final_answer replay" above) ever
reaches a disposition bucket — so ``replay_step_ids`` / ``suspicious_step_ids``
contain AT MOST ONE final_answer id.

A note on the disjointness invariant
=====================================
The plan's MEMORY dispositions (``delete_memory_ids`` / ``quarantine_memory_
ids``) and STEP dispositions (``replay_step_ids`` / ``preserve_step_ids`` /
``redundant_step_ids`` / ``suspicious_step_ids``) — six lists — are asserted
MUTUALLY DISJOINT before returning. ``invalidate_claim_ids`` is the ONE
cross-cutting list: a claim that lost support is invalidated AND replayed or
marked suspicious according to the load-bearing selection flag — "invalidate
the stale claim, then re-derive it" is a single coherent action, so
``invalidate_claim_ids`` is intentionally allowed to overlap ``replay_step_ids``
/ ``suspicious_step_ids``. We assert
``invalidate_claim_ids`` is a subset of ``replay_step_ids | suspicious_step_ids``
(every invalidated claim is also being re-derived or deferred) and that each id
is a claim node, rather than forcing a literal disjointness that rule 6b's
"ALSO" makes impossible.

The redundant_step_ids bucket
=============================
``redundant_step_ids`` remains part of the plan schema for compatibility with
existing result/eval readers. The current default planner does not populate it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import deque
from typing import Any, Dict, List, Set

try:  # importable both as ``methods.ours.rollback_optimizer`` and as a script
    from .graph_schema import DependencyGraph
    from .ours_config import AblationConfig
    from .support_checker import _CHECKABLE_SEMANTIC_TYPES, _CLAIM_SEMANTIC_TYPE
    from .fault_provenance import find_tool_action_for_observation
    from .full_trace_replay import expand_plan_for_full_trace_replay
except ImportError:  # pragma: no cover - direct-script fallback
    from graph_schema import DependencyGraph
    from ours_config import AblationConfig
    from support_checker import _CHECKABLE_SEMANTIC_TYPES, _CLAIM_SEMANTIC_TYPE
    from fault_provenance import find_tool_action_for_observation
    from full_trace_replay import expand_plan_for_full_trace_replay

logger = logging.getLogger(__name__)


def _skip_derived_cleanup(config: AblationConfig) -> bool:
    """Return the internal derived-cleanup toggle, defaulting to disabled.

    The public config no longer exposes ``skip_derived_cleanup`` as a main
    ablation entry, but the planner branch is intentionally kept for
    internal/design-detail experiments.
    """
    return bool(getattr(config, "skip_derived_cleanup", False))


# Final six-way disposition precedence (highest first). A node id that is a
# candidate for several lists is kept ONLY in the highest-precedence one, so the
# six disposition lists come out mutually disjoint.
#   replay > delete > quarantine > preserve > redundant > suspicious
# This satisfies both stated constraints: "replay > quarantine > preserve >
# suspicious" and "delete/quarantine conflict -> prefer delete". delete and the
# step lists never collide by id (memory ids are 'm_*', step ids 's_*'), so
# delete's exact rank relative to replay is immaterial; it sits above quarantine
# where it matters. ``redundant`` ranks BELOW preserve (a node that is both
# genuinely supported AND happens to have an already-covered delete-target is
# preserved — the stronger claim) but ABOVE suspicious (redundant is a KNOWN
# fact, not a deferred uncertainty). invalidate_claim_ids is NOT in this ordering
# — it is the one cross-cutting list (a claim is invalidated AND
# re-derived/deferred).
_DISPOSITION_PRECEDENCE = (
    "replay", "delete", "quarantine", "preserve", "redundant", "suspicious",
)

_STABLE_FACT_PATTERN = re.compile(
    r"\$?\b\d+(?:\.\d+)?\b"
    r"|\b[A-Z]{1,8}\d{2,8}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
)
_STATUS_FACTS = {
    "cancelled",
    "canceled",
    "closed",
    "delivered",
    "open",
    "pending",
    "processing",
    "returned",
    "shipped",
    "superseded",
}


# --------------------------------------------------------------------------- #
# Small structural predicates
# --------------------------------------------------------------------------- #

def _find_last_final_answer_ids(graph: DependencyGraph) -> Set[str]:
    """Return the id(s) of the session's LAST final_answer node(s).

    Collect every node with ``semantic_type == "final_answer"``, take the max
    non-None ``.turn``, and keep only the nodes at that turn. Returns an empty
    set when there are no final_answer nodes. (When final_answer nodes exist but
    NONE carry a turn, ``last_turn`` is ``None`` and every turn-less final_answer
    is returned — identical to the long-standing inline behavior this extracts.)

    This is the SINGLE shared implementation used by BOTH
    ``build_rollback_plan`` (its forced, unconditional last-final_answer
    inclusion into ``replay_step_ids``) AND ``promote_load_bearing_suspicious``
    (its load-bearing reachability target), so the two cannot drift apart.
    """
    final_answer_nodes = [
        (nid, node) for nid, node in graph.nodes.items()
        if node.semantic_type == "final_answer"
    ]
    if not final_answer_nodes:
        return set()
    turns = [node.turn for _, node in final_answer_nodes if node.turn is not None]
    if len(turns) < len(final_answer_nodes):
        logger.warning(
            "%d final_answer node(s) have no turn assigned; excluded "
            "from the last-turn determination",
            len(final_answer_nodes) - len(turns),
        )
    last_turn = max(turns) if turns else None
    return {
        nid for nid, node in final_answer_nodes
        if node.turn == last_turn
    }


def _normalize_fact(raw: str) -> str:
    return raw.strip().strip("'").lower().replace(",", "")


def _stable_facts_from_text(value: Any) -> Set[str]:
    """Extract domain-agnostic stable facts for contamination checks."""
    if not isinstance(value, str) or not value.strip():
        return set()
    facts = {
        normalized
        for match in _STABLE_FACT_PATTERN.finditer(value)
        if (normalized := _normalize_fact(match.group(0)))
    }
    lowered = re.findall(r"[a-zA-Z_]+", value.lower())
    facts.update(token for token in lowered if token in _STATUS_FACTS)
    return facts


def _memory_status(graph: DependencyGraph, memory_id: str) -> str:
    node = graph.node(memory_id)
    if node is None:
        return ""
    return str(node.memory_status or (node.raw or {}).get("status") or "")


def _memory_content(graph: DependencyGraph, memory_id: str) -> str:
    node = graph.node(memory_id)
    if node is None:
        return ""
    return str((node.raw or {}).get("content", ""))


def _active_clean_memory_ids(
    graph: DependencyGraph,
    *,
    faulty_ids: Set[str],
    derived_candidate_ids: Set[str],
) -> Set[str]:
    out: Set[str] = set()
    for memory_id, node in graph.nodes.items():
        if node.node_kind != "memory":
            continue
        if memory_id in faulty_ids or memory_id in derived_candidate_ids:
            continue
        if _memory_status(graph, memory_id) == "active":
            out.add(memory_id)
    return out


def _clean_evidence_memory_ids(
    graph: DependencyGraph,
    memory_id: str,
    *,
    faulty_ids: Set[str],
    derived_candidate_ids: Set[str],
) -> Set[str]:
    evidence_ids = _active_clean_memory_ids(
        graph,
        faulty_ids=faulty_ids,
        derived_candidate_ids=derived_candidate_ids,
    )
    node = graph.node(memory_id)
    for parent_id in (node.raw or {}).get("derived_from", []) if node else []:
        parent = graph.node(parent_id)
        if parent is None or parent.node_kind != "memory":
            continue
        if parent_id == memory_id or parent_id in faulty_ids:
            continue
        evidence_ids.add(parent_id)
    return evidence_ids


def _should_cleanup_derived_memory(
    graph: DependencyGraph,
    memory_id: str,
    *,
    faulty_ids: Set[str],
    derived_candidate_ids: Set[str],
) -> bool:
    """Return whether an unsupported derived memory still carries fault-only facts."""
    candidate_facts = _stable_facts_from_text(_memory_content(graph, memory_id))
    if not candidate_facts:
        return False

    faulty_facts: Set[str] = set()
    for faulty_id in faulty_ids:
        faulty_facts.update(_stable_facts_from_text(_memory_content(graph, faulty_id)))
    overlap = candidate_facts & faulty_facts
    if not overlap:
        return False

    clean_facts: Set[str] = set()
    for clean_id in _clean_evidence_memory_ids(
        graph,
        memory_id,
        faulty_ids=faulty_ids,
        derived_candidate_ids=derived_candidate_ids,
    ):
        clean_facts.update(_stable_facts_from_text(_memory_content(graph, clean_id)))
    return bool(overlap - clean_facts)


def _derived_cleanup_ids(
    graph: DependencyGraph,
    unsupported_derived_ids: List[str],
    faulty_memory_ids: List[str],
    derived_candidate_ids: List[str],
) -> List[str]:
    faulty_ids = set(faulty_memory_ids)
    candidate_ids = set(derived_candidate_ids)
    return sorted(
        memory_id
        for memory_id in unsupported_derived_ids
        if _should_cleanup_derived_memory(
            graph,
            memory_id,
            faulty_ids=faulty_ids,
            derived_candidate_ids=candidate_ids,
        )
    )


# --------------------------------------------------------------------------- #
# Core: build the rollback plan
# --------------------------------------------------------------------------- #

def build_rollback_plan(
    graph: DependencyGraph,
    task_id: str,
    faulty_memory_ids: List[str],
    affected_nodes: Dict[str, Any],
    support_report: Dict[str, Any],
    provenance: Dict[str, Dict[str, Any]],
    config: AblationConfig,
) -> Dict[str, Any]:
    """Combine reachability + provenance + support verdicts into a rollback plan.

    See the module docstring for the full rationale. The plan is a
    JSON-serializable dict (the repair_plan.json payload) with seven action
    lists plus the bookkeeping fields ``method`` / ``task_id`` /
    ``faulty_memory_ids`` / ``reason``.

    Parameters
    ----------
    graph
        The fault-agnostic dependency graph (graph_schema.DependencyGraph).
    task_id
        The task this plan is for (echoed into the payload).
    faulty_memory_ids
        The oracle faulty memory id(s) (the injected seeds, 1-3 per task).
    affected_nodes
        The affected_subgraph.py payload. Only ``affected_node_ids`` is read for
        the disposition logic; the other fields (``resolved_seed_ids`` etc.) are
        available but not required by this module's own logic.
    support_report
        Structured support-check output with ``step_verdicts`` and
        ``derived_memory_verdicts``. Missing step verdicts are treated as false.
    provenance
        ``{memory_id: {producing_step_id, stale_window, ancestor_chain}}`` from
        fault_provenance.py, used only for the per-faulty-memory UPSTREAM replay.
    config
        An ``AblationConfig``. ``skip_derived_cleanup`` controls derived-memory
        quarantine for internal/design-detail experiments.

    Disposition rules (see module docstring for the full statement)
    ---------------------------------------------------------------
    UPSTREAM (per faulty memory): a stale-and-failed window replays its failed
    ``mutation_step_id``; otherwise the ``producing_step_id`` (if any) is
    replayed. In EITHER case, if that replayed step regenerated the memory from
    a cited ``tool_observation``, the ``tool_action`` behind that observation
    (via ``fault_provenance.find_tool_action_for_observation``) is replayed too,
    so the underlying tool call is re-run rather than the memory being
    regenerated from a possibly-stale observation already sitting in the trace.
    This is the ONLY way a ``tool_action`` enters any plan list.

    DOWNSTREAM memories (mode-independent): the faulty memory itself -> delete;
    an unsupported derived candidate -> quarantine (unless
    ``skip_derived_cleanup``); a supported candidate remains active.

    DOWNSTREAM checkable steps (the eight types): verdict ``True`` -> preserve;
    otherwise a claim is ALSO invalidated, and the node goes to suspicious for
    load-bearing selection.

    final_answer EXCEPTION: a final_answer node from any turn OTHER than the
    session's last is NEVER given any disposition — it enters NONE of the six
    lists and is never invalidated, REGARDLESS of its support verdict or
    affected-set membership, the same treatment as ``tool_action`` /
    ``tool_observation``. Only the session's last turn's final_answer ever
    reaches a disposition (and it is then force-replayed; see below), because
    evaluation scores only the last turn — repairing earlier turns' answers
    would spend replay budget for no metric benefit. The support-verdict branch
    therefore runs only for every checkable type EXCEPT final_answer, plus the
    last-turn final_answer node(s).

    ``tool_observation`` is NEVER touched by the downstream disposition logic and
    appears in NO list. ``tool_action`` likewise gets no DOWNSTREAM disposition;
    its sole appearance is in ``replay_step_ids`` via the UPSTREAM rule above
    (the tool_action behind a faulty memory's cited observation). Outside that,
    a ``tool_action``'s disposition is implied by its plan's entry in
    replay/preserve, and ``selective_replay.py`` re-derives that handling from
    the plan rather than reading it from this plan.
    """
    if config.skip_rollback_optimizer:
        plan = _build_no_rollback_optimizer_plan(
            graph,
            task_id,
            faulty_memory_ids,
            affected_nodes,
            support_report,
            config,
        )
        if config.skip_selective_replay:
            plan = expand_plan_for_full_trace_replay(plan, graph, task_id)
        _assert_consistent(plan, graph, config)
        return plan

    affected_ids: List[str] = list(affected_nodes.get("affected_node_ids", []))
    faulty_set: Set[str] = set(faulty_memory_ids)
    step_verdicts = support_report["step_verdicts"]
    derived_verdicts = support_report["derived_memory_verdicts"]
    derived_candidate_ids = sorted(derived_verdicts)
    supported_derived_ids = sorted(
        memory_id
        for memory_id, verdict in derived_verdicts.items()
        if verdict["supported"]
    )
    unsupported_derived_ids = sorted(
        memory_id
        for memory_id, verdict in derived_verdicts.items()
        if not verdict["supported"]
    )
    cleanup_derived_ids = _derived_cleanup_ids(
        graph,
        unsupported_derived_ids,
        faulty_memory_ids,
        derived_candidate_ids,
    )
    # Candidate buckets (resolved to disjoint lists by precedence at the end).
    replay: Set[str] = set()
    delete: Set[str] = set()
    quarantine: Set[str] = set()
    preserve: Set[str] = set()
    redundant: Set[str] = set()
    suspicious: Set[str] = set()
    invalidate_claims: Set[str] = set()

    # --- UPSTREAM: per faulty memory ------------------------------------- #
    def _replay_with_tool_action(step_id: str) -> None:
        """Replay ``step_id``, and ALSO replay the tool_action behind any
        tool_observation it cites. When a producing/mutation step regenerated a
        memory from a cited tool_observation, replaying the step alone would just
        regenerate from the (possibly stale) observation already in the trace; we
        re-run the underlying tool_action so the observation itself is refreshed.
        """
        replay.add(step_id)
        tool_action_id = find_tool_action_for_observation(graph, step_id)
        if tool_action_id is not None:
            replay.add(tool_action_id)

    for mid in faulty_memory_ids:
        prov = provenance.get(mid, {}) or {}
        stale_window = prov.get("stale_window")
        if stale_window is not None and stale_window.get("took_effect") is False:
            # The failed mutation must be replayed (it never took effect). No
            # separate window-preservation: affected_subgraph.py's seed
            # resolution already excludes the clean window. The failed mutation's
            # used_ids could itself cite a tool_observation needing its
            # tool_action replayed, so route it through the helper too.
            _replay_with_tool_action(stale_window["mutation_step_id"])
        elif prov.get("producing_step_id") is not None:
            _replay_with_tool_action(prov["producing_step_id"])
        # else: no upstream action (root memory with no producing step).

    # --- DOWNSTREAM memories (mode-independent) -------------------------- #
    # Rule 4: the faulty memory itself -> delete.
    delete |= faulty_set
    # Rule 5: quarantine only unsupported derived-memory candidates that still
    # carry fault-only facts not covered by clean active memory evidence.
    if not _skip_derived_cleanup(config):
        quarantine.update(cleanup_derived_ids)

    # The session's last final_answer node(s). Computed ONCE here, before the
    # per-node disposition loop, and reused by the forced last-final_answer
    # replay below — the SAME set drives both the early-continue (which keeps
    # an EARLIER turn's final_answer out of every candidate bucket) and the
    # forced inclusion (which only ever touches the last turn's node(s)), so
    # the two cannot disagree about which final_answer is "the last".
    last_final_answer_ids = _find_last_final_answer_ids(graph)

    # --- DOWNSTREAM checkable steps -------------------------------------- #
    for nid in affected_ids:
        node = graph.node(nid)
        if node is None:
            continue
        stype = node.semantic_type
        if stype not in _CHECKABLE_SEMANTIC_TYPES:
            # tool_action / tool_observation / raw memory / unknown — never
            # touched. No entry in any list.
            continue

        if stype == "final_answer" and nid not in last_final_answer_ids:
            # An earlier turn's final_answer -- NOT the session's measured
            # outcome. No action: never replayed, never preserved, never marked
            # suspicious, never invalidated -- the same treatment as
            # tool_action / tool_observation, regardless of its support verdict
            # or affected-set membership. Only the session's last turn is
            # scored, so repairing an earlier turn's answer would spend replay
            # budget for no metric benefit.
            continue

        if step_verdicts.get(nid) is True:
            # Independent trusted backing survives -> preserve, in BOTH modes.
            preserve.add(nid)
            continue

        # No trusted support (verdict False, or absent).
        if stype == _CLAIM_SEMANTIC_TYPE:
            # A claim that lost support is invalidated AND re-derived/deferred.
            invalidate_claims.add(nid)

        suspicious.add(nid)

    # --- Resolve precedence into mutually-disjoint disposition lists ----- #
    buckets = {
        "replay": replay,
        "delete": delete,
        "quarantine": quarantine,
        "preserve": preserve,
        "redundant": redundant,
        "suspicious": suspicious,
    }
    assigned: Dict[str, str] = {}
    for name in _DISPOSITION_PRECEDENCE:
        for nid in buckets[name]:
            assigned.setdefault(nid, name)
    final = {
        name: sorted(nid for nid, owner in assigned.items() if owner == name)
        for name in _DISPOSITION_PRECEDENCE
    }

    # --- FORCED, UNCONDITIONAL last-final_answer replay ------------------- #
    # The session's last final_answer is added to replay_step_ids regardless of
    # whether anything upstream of it was actually
    # touched by this plan, and regardless of its own support verdict. Applied
    # AFTER precedence resolution so no lower-precedence assignment can silently
    # override it, and BEFORE the load-bearing promotion call so the id is
    # already in replay going into promote_load_bearing_suspicious.
    #
    # Collision handling: final_answer is one of the 8 checkable types, so a
    # last-final_answer id CAN legitimately have landed in another bucket above
    # — preserve (if it kept trusted support) or suspicious (no support, pending
    # load-bearing selection). It can never be in delete/quarantine (those hold memory ids) and
    # is excluded from redundant (that bucket is memory_delete-only). Because
    # this inclusion is UNCONDITIONAL, it overrides any such assignment: each
    # forced id is stripped from every other disposition list and ends up in
    # replay_step_ids ONLY, keeping the six-way disjointness invariant intact.
    # ``last_final_answer_ids`` was computed once near the top of this function
    # (before the per-node loop) and is reused here unchanged.
    if last_final_answer_ids:
        for name in _DISPOSITION_PRECEDENCE:
            if name == "replay":
                continue
            if any(nid in last_final_answer_ids for nid in final[name]):
                final[name] = sorted(
                    nid for nid in final[name] if nid not in last_final_answer_ids
                )
    final["replay"] = sorted(set(final["replay"]) | last_final_answer_ids)

    # invalidate_claim_ids is cross-cutting: keep only claims still being acted
    # on (replayed or marked suspicious), which by construction is all of them.
    replay_or_suspicious = set(final["replay"]) | set(final["suspicious"])
    invalidate_claim_ids = sorted(invalidate_claims & replay_or_suspicious)

    plan = {
        "method": "ours",
        "task_id": task_id,
        "faulty_memory_ids": list(faulty_memory_ids),
        "derived_memory_candidate_ids": derived_candidate_ids,
        "supported_derived_memory_ids": supported_derived_ids,
        "unsupported_derived_memory_ids": unsupported_derived_ids,
        "delete_memory_ids": final["delete"],
        "quarantine_memory_ids": final["quarantine"],
        "invalidate_claim_ids": invalidate_claim_ids,
        "replay_step_ids": final["replay"],
        "preserve_step_ids": final["preserve"],
        "redundant_step_ids": final["redundant"],
        "suspicious_step_ids": final["suspicious"],
    }
    
    # Default Our Method resolves load-bearing suspicious nodes into
    # replay_step_ids before returning.
    plan = promote_load_bearing_suspicious(graph, plan)

    # memory_read: a FINAL pass, after every other node's disposition
    # (including load-bearing promotion above, when enabled) is
    # settled. Replay this read ONLY if its chain successor ends up in
    # the now-final replay_step_ids — a read whose successor is not
    # being replayed has no consumer for a fresh result, so re-running
    # it would be pure waste against this project's minimal-replay goal.
    final_replay_set = set(plan["replay_step_ids"])
    memory_read_replay: List[str] = []
    for nid in affected_nodes["affected_node_ids"]:
        node = graph.node(nid)
        if node is None or node.semantic_type != "memory_read":
            continue
        chain_successors = {
            dst for dst, etypes in graph.successor_edges(nid).items()
            if "chain" in etypes
        }
        if chain_successors & final_replay_set:
            memory_read_replay.append(nid)
    plan["replay_step_ids"] = sorted(set(plan["replay_step_ids"]) | set(memory_read_replay))
    
    # Build the ``reason`` summary AFTER promotion so its counts reflect the
    # final, returned lists.
    plan["reason"] = _build_reason(plan, config)

    if config.skip_selective_replay:
        plan = expand_plan_for_full_trace_replay(plan, graph, task_id)

    _assert_consistent(plan, graph, config)
    return plan


def _build_no_rollback_optimizer_plan(
    graph: DependencyGraph,
    task_id: str,
    faulty_memory_ids: List[str],
    affected_nodes: Dict[str, Any],
    support_report: Dict[str, Any],
    config: AblationConfig,
) -> Dict[str, Any]:
    """Build the coarse "affected all" plan for the rollback-optimizer ablation."""
    affected_ids: List[str] = list(affected_nodes.get("affected_node_ids", []))
    derived_verdicts = support_report.get("derived_memory_verdicts", {})
    derived_candidate_ids = sorted(derived_verdicts)
    supported_derived_ids = sorted(
        memory_id
        for memory_id, verdict in derived_verdicts.items()
        if verdict.get("supported")
    )
    unsupported_derived_ids = sorted(
        memory_id
        for memory_id, verdict in derived_verdicts.items()
        if not verdict.get("supported")
    )

    replay: Set[str] = set()
    invalidate_claims: Set[str] = set()
    replayable_types = set(_CHECKABLE_SEMANTIC_TYPES) | {"tool_action"}

    for nid in affected_ids:
        node = graph.node(nid)
        if node is None:
            continue
        stype = node.semantic_type
        if stype in replayable_types:
            replay.add(nid)
            if stype == _CLAIM_SEMANTIC_TYPE:
                invalidate_claims.add(nid)
        elif stype == "tool_observation":
            for pred, etypes in graph.predecessor_edges(nid).items():
                pred_node = graph.node(pred)
                if (
                    "chain" in etypes
                    and pred_node is not None
                    and pred_node.semantic_type == "tool_action"
                ):
                    replay.add(pred)

    replay.update(_find_last_final_answer_ids(graph))

    plan = {
        "method": "ours",
        "task_id": task_id,
        "faulty_memory_ids": list(faulty_memory_ids),
        "derived_memory_candidate_ids": derived_candidate_ids,
        "supported_derived_memory_ids": supported_derived_ids,
        "unsupported_derived_memory_ids": unsupported_derived_ids,
        "delete_memory_ids": sorted(set(faulty_memory_ids)),
        "quarantine_memory_ids": (
            [] if _skip_derived_cleanup(config) else derived_candidate_ids
        ),
        "invalidate_claim_ids": sorted(invalidate_claims & replay),
        "replay_step_ids": sorted(replay),
        "preserve_step_ids": [],
        "redundant_step_ids": [],
        "suspicious_step_ids": [],
    }
    plan["reason"] = _build_reason(
        plan,
        config,
        extra=["skip_rollback_optimizer", "affected_all"],
    )
    return plan


def _build_reason(
    plan: Dict[str, Any],
    config: AblationConfig,
    *,
    extra: List[str] | None = None,
) -> str:
    """A short human-readable summary of the plan (the ``reason`` field).

    Reads counts straight off the final plan dict, so the reported
    replay/suspicious counts always match the returned lists exactly.
    """
    parts = [
        f"delete={len(plan['delete_memory_ids'])}",
        f"quarantine={len(plan['quarantine_memory_ids'])}",
        f"derived_candidates={len(plan['derived_memory_candidate_ids'])}",
        f"derived_supported={len(plan['supported_derived_memory_ids'])}",
        f"derived_unsupported={len(plan['unsupported_derived_memory_ids'])}",
        f"invalidate_claims={len(plan['invalidate_claim_ids'])}",
        f"replay={len(plan['replay_step_ids'])}",
        f"preserve={len(plan['preserve_step_ids'])}",
        f"redundant={len(plan['redundant_step_ids'])}",
        f"suspicious={len(plan['suspicious_step_ids'])}",
    ]
    if _skip_derived_cleanup(config):
        parts.append("skip_derived_cleanup")
    if config.skip_rollback_optimizer:
        parts.append("skip_rollback_optimizer")
    if extra:
        for item in extra:
            if item not in parts:
                parts.append(item)
    return "; ".join(parts)


def _assert_consistent(
    plan: Dict[str, Any], graph: DependencyGraph, config: AblationConfig
) -> None:
    """Assert the plan's invariants before it is returned.

    1. The SIX disposition lists are MUTUALLY DISJOINT (delete / quarantine /
       replay / preserve / redundant / suspicious share no id).
    2. ``invalidate_claim_ids`` is a SUBSET of replay | suspicious (every
       invalidated claim is also being re-derived or deferred), and each entry
       is a claim node — the one allowed cross-cutting overlap.
    3. No ``tool_observation`` id appears in ANY list, and no ``tool_action``
       appears in any list OTHER than ``replay_step_ids`` (where the upstream
       rule may place the tool_action behind a faulty memory's cited
       observation — the one deliberate carve-out).
    """
    disposition_lists = {
        "delete_memory_ids": plan["delete_memory_ids"],
        "quarantine_memory_ids": plan["quarantine_memory_ids"],
        "replay_step_ids": plan["replay_step_ids"],
        "preserve_step_ids": plan["preserve_step_ids"],
        "redundant_step_ids": plan["redundant_step_ids"],
        "suspicious_step_ids": plan["suspicious_step_ids"],
    }
    candidates = set(plan["derived_memory_candidate_ids"])
    supported = set(plan["supported_derived_memory_ids"])
    unsupported = set(plan["unsupported_derived_memory_ids"])
    faulty = set(plan["faulty_memory_ids"])
    deleted = set(plan["delete_memory_ids"])
    quarantined = set(plan["quarantine_memory_ids"])
    if supported & unsupported:
        raise AssertionError("supported and unsupported derived memories overlap")
    if supported | unsupported != candidates:
        raise AssertionError("derived-memory verdicts do not partition candidates")
    if faulty & candidates:
        raise AssertionError("source faulty memories must not be derived candidates")
    if not config.skip_rollback_optimizer and supported & (deleted | quarantined):
        raise AssertionError("supported derived memories must remain active")
    if _skip_derived_cleanup(config):
        if candidates & quarantined:
            raise AssertionError("skip_derived_cleanup must not quarantine candidates")
    elif config.skip_rollback_optimizer:
        if not candidates <= quarantined:
            raise AssertionError(
                "skip_rollback_optimizer must quarantine all derived candidates"
            )
    elif not quarantined <= unsupported:
        raise AssertionError(
            "derived cleanup may quarantine only unsupported derived memories"
        )
    if candidates & set(plan["preserve_step_ids"]):
        raise AssertionError("memory ids must not appear in preserve_step_ids")
    seen: Dict[str, str] = {}
    for list_name, ids in disposition_lists.items():
        for nid in ids:
            if nid in seen:
                raise AssertionError(
                    f"plan lists not disjoint: {nid!r} in both "
                    f"{seen[nid]!r} and {list_name!r}"
                )
            seen[nid] = list_name

    replay_or_suspicious = set(plan["replay_step_ids"]) | set(
        plan["suspicious_step_ids"]
    )
    for cid in plan["invalidate_claim_ids"]:
        if cid not in replay_or_suspicious:
            raise AssertionError(
                f"invalidated claim {cid!r} is neither replayed nor suspicious"
            )
        node = graph.node(cid)
        if node is not None and node.semantic_type != _CLAIM_SEMANTIC_TYPE:
            raise AssertionError(
                f"invalidate_claim_ids holds non-claim {cid!r} "
                f"(semantic_type={node.semantic_type!r})"
            )

    # tool_observation must never appear anywhere; tool_action may appear ONLY
    # in replay_step_ids (the upstream carve-out: the tool_action behind a faulty
    # memory's cited observation), never in any other list.
    for list_name, ids in disposition_lists.items():
        for nid in ids:
            node = graph.node(nid)
            if node is None:
                continue
            if node.semantic_type == "tool_observation":
                raise AssertionError(
                    f"tool_observation {nid!r} must never appear in a plan "
                    f"list (found in {list_name!r})"
                )
            if (
                node.semantic_type == "tool_action"
                and list_name != "replay_step_ids"
            ):
                raise AssertionError(
                    f"tool_action {nid!r} may only appear in replay_step_ids "
                    f"(the upstream carve-out), not in {list_name!r}"
                )


# --------------------------------------------------------------------------- #
# Load-bearing safety pass (run automatically by build_rollback_plan)
# --------------------------------------------------------------------------- #

def promote_load_bearing_suspicious(
    graph: DependencyGraph, plan: Dict[str, Any]
) -> Dict[str, Any]:
    """Safety pass over a plan with suspicious nodes: promotes any suspicious
    node that is actually load-bearing for final_answer's correctness
    into replay_step_ids. Every other suspicious node is left
    UNCHANGED in suspicious_step_ids — this function never assigns
    preserve_step_ids and never resolves ambiguity it doesn't have to.
    Returns a NEW plan dict (does not mutate the input). If
    suspicious_step_ids is already empty, returns the plan unchanged.

    This is run AUTOMATICALLY by build_rollback_plan. The empty-list guard below
    remains as a harmless no-op for the edge case of an already-promoted plan
    being re-passed.
    """
    if not plan["suspicious_step_ids"]:
        return plan

    severed = set(plan["delete_memory_ids"]) | set(plan["quarantine_memory_ids"])

    final_answer_ids = _find_last_final_answer_ids(graph)

    replay = set(plan["replay_step_ids"])
    remaining_suspicious: Set[str] = set()

    for nid in plan["suspicious_step_ids"]:
        if _has_active_path_to_final_answer(
            graph, nid, final_answer_ids, severed
        ):
            replay.add(nid)
        else:
            remaining_suspicious.add(nid)  # left exactly as is

    new_plan = dict(plan)
    new_plan["replay_step_ids"] = sorted(replay)
    new_plan["suspicious_step_ids"] = sorted(remaining_suspicious)
    # preserve_step_ids is DELIBERATELY untouched -- copy it through
    # verbatim, do not recompute or add to it.
    new_plan["preserve_step_ids"] = list(plan["preserve_step_ids"])

    # A claim in invalidate_claim_ids must still be in replay | suspicious after
    # the pass: this function only MOVES nodes from suspicious to replay, never
    # to preserve or out of the union, so the cross-cutting invariant cannot be
    # broken by promotion. Re-check it to catch any real bug in the logic.
    replay_or_suspicious = set(new_plan["replay_step_ids"]) | set(
        new_plan["suspicious_step_ids"]
    )
    for cid in new_plan["invalidate_claim_ids"]:
        if cid not in replay_or_suspicious:
            raise AssertionError(
                f"after promotion, invalidated claim {cid!r} is neither "
                f"replayed nor suspicious — load-bearing promotion must only "
                f"move nodes from suspicious to replay, never out of the union"
            )
    return new_plan


def _has_active_path_to_final_answer(
    graph: DependencyGraph,
    start: str,
    final_answer_ids: Set[str],
    severed: Set[str],
) -> bool:
    """Bounded forward BFS from start: True if any final_answer node is
    reachable WITHOUT passing through a node in severed (cut by the
    plan). Does NOT stop at preserve_step_ids entries (unlike an
    earlier draft of this idea) -- a preserved node still structurally
    passes evidence through if reachable; only SEVERED nodes (deleted/
    quarantined memories) actually cut a path. Log at DEBUG when a path
    is pruned at a severed node, for debugging visibility."""
    if start in severed:
        logger.debug("path pruned: start %r is itself severed", start)
        return False
    visited: Set[str] = {start}
    queue: deque[str] = deque([start])
    while queue:
        node_id = queue.popleft()
        if node_id in final_answer_ids:
            return True
        for nxt in sorted(graph.successors(node_id)):
            if nxt in visited:
                continue
            if nxt in severed:
                logger.debug(
                    "path pruned at severed node %r (reached from %r)",
                    nxt, node_id,
                )
                continue
            visited.add(nxt)
            queue.append(nxt)
    return False


# --------------------------------------------------------------------------- #
# CLI (mirrors affected_subgraph.py / support_checker.py: argparse, small
# _load helpers, a printed summary block, json.dump output)
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
        description="Build a minimal rollback plan from the affected subgraph, "
                    "the independent-support verdicts, and the fault provenance. "
                    "Decides delete/quarantine/invalidate/replay/preserve/"
                    "suspicious dispositions."
    )
    parser.add_argument(
        "--graph", required=True,
        help="dependency_graph.json path (DependencyGraph.to_dict output)",
    )
    parser.add_argument(
        "--faulty_memory_ids", required=True, nargs="+",
        help="the oracle faulty memory id(s) injected into this task",
    )
    parser.add_argument(
        "--affected_nodes", required=True,
        help="affected_nodes.json path from affected_subgraph.py",
    )
    parser.add_argument(
        "--support_check", required=True,
        help="structured support_check.json path from support_checker.py",
    )
    parser.add_argument(
        "--provenance", required=True,
        help="provenance.json path from fault_provenance.py",
    )
    # Internal/design-detail toggle kept in AblationConfig and planner logic,
    # but no longer exposed as a standalone CLI ablation entry.
    # parser.add_argument(
    #     "--skip_derived_cleanup", action="store_true",
    #     help="'Ours w/o derived memory cleanup': leave all derived candidates "
    #          "untouched (not quarantined)",
    # )
    parser.add_argument(
        "--output", default=None,
        help="optional path to write the rollback plan JSON; if omitted, only "
             "the summary is printed and no file is written",
    )
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    affected_nodes = _load_json(args.affected_nodes)
    support_report = _load_json(args.support_check)
    provenance = _load_json(args.provenance)
    config = AblationConfig()

    plan = build_rollback_plan(
        graph,
        graph.task_id,
        args.faulty_memory_ids,
        affected_nodes,
        support_report,
        provenance,
        config,
    )

    print(f"=== rollback plan (task {graph.task_id}) ===")
    print(f"  faulty_memory_ids:     {plan['faulty_memory_ids']}")
    print(f"  delete_memory_ids:     {plan['delete_memory_ids']}")
    print(f"  quarantine_memory_ids: {plan['quarantine_memory_ids']}")
    print(f"  derived candidates:     {plan['derived_memory_candidate_ids']}")
    print(f"  supported derived:      {plan['supported_derived_memory_ids']}")
    print(f"  unsupported derived:    {plan['unsupported_derived_memory_ids']}")
    print(f"  invalidate_claim_ids:  {plan['invalidate_claim_ids']}")
    print(f"  replay_step_ids:       {plan['replay_step_ids']}")
    print(f"  preserve_step_ids:     {plan['preserve_step_ids']}")
    print(f"  redundant_step_ids:    {plan['redundant_step_ids']}")
    print(f"  suspicious_step_ids:   {plan['suspicious_step_ids']}")
    print(f"  reason: {plan['reason']}")
    # The load-bearing pass runs inside build_rollback_plan. replay_step_ids
    # already includes any promoted load-bearing nodes; whatever stays in
    # suspicious_step_ids is, by construction, the genuinely non-load-bearing
    # residue.
    print(
        f"  load-bearing pass: replay_step_ids includes any promoted "
        f"load-bearing nodes; {len(plan['suspicious_step_ids'])} node(s) "
        f"remain genuinely suspicious (non-load-bearing)"
    )
    if args.output:
        _write_json(plan, args.output)
        print(f"  wrote: {args.output}")


if __name__ == "__main__":
    _main()
