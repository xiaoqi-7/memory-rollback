"""Rollback state reconstruction and structured selective replay.

The canonical execution flow is::

    setup_replay_state
    -> replay_chain
    -> ReplayBackend operations
    -> repaired trace/state

``setup_replay_state`` and ``build_replay_sequence`` remain independently
usable for setup-only diagnostics. Full replay is implemented only by the
structured :class:`ReplayRuntime` + :class:`ReplayBackend` path; this module
does not construct a fake or production backend.

The setup phase produces two inspectable artifacts:

* ``replay_state_snapshot.json`` — the agent state immediately before replay
  begins (memories loaded + dispositions applied, task_id / turn_index set to
  the first replay point). Two fields are time-scoped to that point:
  ``history_preview`` reports every turn STRICTLY BEFORE the first replay step's
  own turn (the current, not-yet-replayed turn is deliberately excluded — a
  turn's input is appended to ``self.history`` only AFTER that turn completes in
  ``react_agent.run()``, so a turn that has not been replayed is correctly
  absent), and ``memories_as_of_cutoff`` serializes the reconstructed store at
  that exact boundary.
* ``replay_sequence.json`` — the ordered replay manifest: for every step from
  the first replay point to the end of the trace, its disposition, its
  chain-predecessor context dependencies, and whether it is superseded.

State reconstruction
====================
The rollback executor reduces the trace prefix into the live agent state.  It
restores cutoff-scoped memories, completed-turn history, the current 1-based
turn index, and successful read-only tool calls.  Snapshot fields are serialized
from that same live state rather than computed through a parallel code path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional

from .graph_schema import DependencyGraph
from . import repair_ops
from .full_trace_replay import remap_memory_mutation_step
from .replay_types import ReplayContext, ReplayContextObserver

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard agent import
    from agent.react_agent import ReactAgent

logger = logging.getLogger(__name__)

# The sequencing edge label whose predecessors are a step's reasoning-chain
# context (graph_schema.EdgeType.CHAIN). Replay context comes from graph
# predecessors, not from the manifest's previous entry.
_CHAIN = "chain"
_CITE = "cite"

# The four STEP-level disposition lists on the plan, mapped to the disposition
# label this manifest reports. delete/quarantine are MEMORY dispositions (m_*
# ids) and never name a step in steps_in_order, so they are not consulted here.
# The plan's six lists are already mutually disjoint (rollback_optimizer asserts
# it), so at most one of these ever matches a given step id.
_STEP_DISPOSITION_LISTS = (
    ("replay", "replay_step_ids"),
    ("preserve", "preserve_step_ids"),
    ("redundant", "redundant_step_ids"),
    ("suspicious", "suspicious_step_ids"),
)

# Semantic types that get no step-level disposition of their own and so are
# reported as pass_through; the ``superseded`` flag is only ever computed for
# these two. A tool_action/tool_observation is superseded by EITHER of two
# distinct paths: (1) its owning plan was replayed, so the whole plan — and
# every tool_action/tool_observation under it — is regenerated; or (2) the
# upstream carve-out, where rollback_optimizer.py replayed a tool_action
# DIRECTLY (re-running the tool call behind a faulty memory's cited
# observation) without necessarily replaying that tool_action's own plan, so
# only the observation it produced is replaced. See ``_is_superseded``.
_PASS_THROUGH_TYPES = frozenset({"tool_action", "tool_observation"})
_PLAN_TYPE = "plan"

_TRACE_LIST_FIELDS = (
    "prev_step_ids",
    "next_step_ids",
    "used_ids",
    "generated_memory_ids",
    "invalidated_memory_ids",
)
_STEP_REFERENCE_FIELDS = ("prev_step_ids", "next_step_ids")
_MEMORY_REFERENCE_FIELDS = ("generated_memory_ids", "invalidated_memory_ids")
_FIXED_REPAIR_POLICY = {
    "enable_final_turn_missing_action_replan": True,
    "synthetic_tools_read_only_only": True,
}


def _resolve_remap(identifier: Any, remap: Dict[str, str]) -> Any:
    """Follow a possibly transitive ID remap without looping on bad input."""
    if not isinstance(identifier, str):
        return identifier
    seen = set()
    current = identifier
    while current not in seen:
        seen.add(current)
        mapped = remap.get(current)
        if not isinstance(mapped, str) or not mapped:
            break
        current = mapped
    return current


def _copy_complete_trace_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Return a detached, schema-complete trace record."""
    normalized = dict(step)
    for field in _TRACE_LIST_FIELDS:
        value = normalized.get(field, [])
        if not isinstance(value, list):
            raise ValueError(f"trace step field {field!r} must be a list")
        normalized[field] = list(value)
    normalized.setdefault("tool_name", None)
    normalized.setdefault("tool_args", None)
    normalized.setdefault("timestamp", None)
    return normalized


def _normalize_historical_step(
    step: Dict[str, Any],
    step_remap: Dict[str, str],
) -> Dict[str, Any]:
    """Repair structural step references without rewriting old provenance."""
    normalized = _copy_complete_trace_step(step)
    for field in _STEP_REFERENCE_FIELDS:
        normalized[field] = [
            _resolve_remap(identifier, step_remap)
            for identifier in normalized[field]
        ]
    # used_ids mixes step IDs, memory IDs, and external IDs. Only known step
    # replacements are resolved for a historical record; memory provenance and
    # mutation fields remain byte-for-byte equivalent at the list-item level.
    normalized["used_ids"] = [
        _resolve_remap(identifier, step_remap)
        for identifier in normalized["used_ids"]
    ]
    return normalized


def _normalize_replacement_step(
    step: Dict[str, Any],
    step_remap: Dict[str, str],
    memory_remap: Dict[str, str],
) -> Dict[str, Any]:
    """Return a newly generated record with all current IDs resolved."""
    normalized = _normalize_historical_step(step, step_remap)
    def resolve_mutation_memory_id(identifier: Any) -> Any:
        if isinstance(identifier, str) and identifier.startswith("m_n"):
            return identifier
        return _resolve_remap(identifier, memory_remap)

    if normalized.get("step_type") not in {
        "memory_write", "memory_update", "memory_consolidate"
    }:
        normalized["generated_memory_ids"] = [
            _resolve_remap(identifier, memory_remap)
            for identifier in normalized["generated_memory_ids"]
        ]
    else:
        normalized["generated_memory_ids"] = [
            resolve_mutation_memory_id(identifier)
            for identifier in normalized["generated_memory_ids"]
        ]
    if normalized.get("step_type") not in {
        "memory_delete", "memory_update", "memory_consolidate"
    }:
        normalized["invalidated_memory_ids"] = [
            _resolve_remap(identifier, memory_remap)
            for identifier in normalized["invalidated_memory_ids"]
        ]
    else:
        normalized["invalidated_memory_ids"] = [
            resolve_mutation_memory_id(identifier)
            for identifier in normalized["invalidated_memory_ids"]
        ]
    # used_ids deliberately accepts external values such as user_input_*. IDs
    # that are known steps or memories are remapped; all other values survive.
    normalized["used_ids"] = [
        _resolve_remap(
            _resolve_remap(identifier, step_remap), memory_remap
        )
        for identifier in normalized["used_ids"]
    ]
    return normalized


def _validate_repaired_trace(
    steps: List[Dict[str, Any]],
    step_remap: Dict[str, str],
    memory_remap: Dict[str, str],
    removed_step_ids: Optional[set[str]] = None,
    *,
    historical_step_ids: Optional[set[str]] = None,
    replacement_step_ids: Optional[set[str]] = None,
    repaired_memories: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Validate the structural and reference invariants of a repaired trace."""
    step_ids = [step.get("step_id") for step in steps]
    if any(not isinstance(step_id, str) or not step_id for step_id in step_ids):
        raise ValueError("every repaired trace step must have a non-empty step_id")
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("repaired trace contains duplicate step IDs")
    existing = set(step_ids)
    removed_steps = set(removed_step_ids or ())
    emitted_removed = existing.intersection(removed_steps)
    if emitted_removed:
        raise ValueError(
            "repaired trace still emits removed tool steps: "
            f"{sorted(emitted_removed)!r}"
        )
    superseded_steps = {
        old_id for old_id, new_id in step_remap.items()
        if isinstance(new_id, str) and new_id and new_id != old_id
    }
    replaced_memories = {
        old_id for old_id, new_id in memory_remap.items()
        if isinstance(new_id, str) and new_id and new_id != old_id
    }
    historical = set(historical_step_ids or ())
    replacements = (
        set(replacement_step_ids)
        if replacement_step_ids is not None
        else existing - historical
    )
    inactive_memories = {
        memory.get("memory_id")
        for memory in (repaired_memories or [])
        if memory.get("status") in {"deleted", "quarantined", "superseded"}
    }
    invalidated_by_replacement_mutation = {
        memory_id
        for step in steps
        if step.get("step_id") in replacements
        and step.get("step_type") in {
            "memory_delete", "memory_update", "memory_consolidate"
        }
        for memory_id in (step.get("invalidated_memory_ids") or [])
        if isinstance(memory_id, str)
    }
    for step in steps:
        for field in _TRACE_LIST_FIELDS:
            if not isinstance(step.get(field), list):
                raise ValueError(
                    f"repaired trace step {step['step_id']!r} field {field!r} "
                    "must be a list"
                )
        for field in _STEP_REFERENCE_FIELDS:
            references = step[field]
            if len(references) != len(set(references)):
                raise ValueError(
                    f"repaired trace step {step['step_id']!r} has duplicate "
                    f"step references in {field}: {references!r}"
                )
            stale = (superseded_steps | removed_steps).intersection(references)
            if stale:
                raise ValueError(
                    f"repaired trace step {step['step_id']!r} has superseded "
                    f"step references in {field}: {sorted(stale)!r}"
                )
            dangling = [reference for reference in references if reference not in existing]
            if dangling:
                raise ValueError(
                    f"repaired trace step {step['step_id']!r} has dangling "
                    f"step references in {field}: {dangling!r}"
                )
        is_historical = step["step_id"] in historical
        for field in ("used_ids",) + _MEMORY_REFERENCE_FIELDS:
            stale_steps = (superseded_steps | removed_steps).intersection(step[field])
            stale_memories = (
                set() if is_historical
                else replaced_memories.intersection(step[field])
            )
            if (
                field in {"generated_memory_ids", "invalidated_memory_ids"}
                and step.get("step_type") in {
                    "memory_write", "memory_delete", "memory_update",
                    "memory_consolidate"
                }
            ):
                stale_memories = set()
            if stale_steps or stale_memories:
                removed_dependencies = removed_steps.intersection(step[field])
                if field == "used_ids" and removed_dependencies:
                    old_id = sorted(removed_dependencies)[0]
                    raise ValueError(
                        f"retained step {step['step_id']} still depends on "
                        f"removed tool step {old_id}; the dependent step "
                        "must also be replayed"
                    )
                raise ValueError(
                    f"repaired trace step {step['step_id']!r} has stale "
                    f"references in {field}: "
                    f"{sorted(stale_steps | stale_memories)!r}"
                )
        if step["step_id"] in replacements:
            inactive_evidence = inactive_memories.intersection(step["used_ids"])
            inactive_evidence -= invalidated_by_replacement_mutation
            if inactive_evidence:
                raise ValueError(
                    f"replacement trace step {step['step_id']!r} cites inactive "
                    f"memories: {sorted(inactive_evidence)!r}"
                )

    by_id = {step["step_id"]: step for step in steps}
    for step in steps:
        step_id = step["step_id"]
        for predecessor_id in step["prev_step_ids"]:
            if step_id not in by_id[predecessor_id]["next_step_ids"]:
                raise ValueError(
                    f"non-reciprocal edge {predecessor_id!r} -> {step_id!r}: "
                    f"{step_id!r} is absent from the predecessor's next_step_ids"
                )
        for successor_id in step["next_step_ids"]:
            if step_id not in by_id[successor_id]["prev_step_ids"]:
                raise ValueError(
                    f"non-reciprocal edge {step_id!r} -> {successor_id!r}: "
                    f"{step_id!r} is absent from the successor's prev_step_ids"
                )


# --------------------------------------------------------------------------- #
# Small shared lookups (implemented ONCE, reused by both halves)
# --------------------------------------------------------------------------- #

def _plan_step_sets(plan: Dict[str, Any]) -> Dict[str, set]:
    """Pre-build the four STEP-disposition lists as sets for O(1) membership.

    Reads the plan's own lists verbatim. This is pure REPORTING — it does not
    re-derive any precedence (rollback_optimizer already resolved the six lists
    to be mutually disjoint); it only indexes them for fast lookup.
    """
    return {label: set(plan.get(key, [])) for label, key in _STEP_DISPOSITION_LISTS}


def _disposition_for(step_id: str, step_sets: Dict[str, set]) -> str:
    """Report a step's disposition by reading the plan's lists (no decisions).

    Returns the first matching label in report precedence, or ``"pass_through"``
    when the id is in none of the four step lists — a ``tool_action`` /
    ``tool_observation`` (never given a step-level disposition) or a step outside
    the affected region encountered because the walk runs to the end of trace.
    """
    for label, _key in _STEP_DISPOSITION_LISTS:
        if step_id in step_sets[label]:
            return label
    return "pass_through"


def _chain_predecessor_ids(graph: DependencyGraph, step_id: str) -> List[str]:
    """Sorted CHAIN predecessors of ``step_id`` (its reasoning-chain context).

    Reads ``graph.predecessor_edges(step_id)`` and keeps only neighbours whose
    label set contains ``"chain"``. This is the context a future
    replay context uses — NOT the previous entry in the manifest list.
    """
    return sorted(
        src
        for src, labels in graph.predecessor_edges(step_id).items()
        if _CHAIN in labels
    )


def _context_predecessor_ids(graph: DependencyGraph, step_id: str) -> List[str]:
    """Sorted predecessors that may provide replay context.

    ``chain`` captures structural sequencing while ``cite`` captures explicit
    provenance.  They intentionally remain distinct in the graph and are
    combined only at the replay-context boundary.
    """
    return sorted(
        src
        for src, labels in graph.predecessor_edges(step_id).items()
        if labels & {_CHAIN, _CITE}
    )


def _owning_plan_id(graph: DependencyGraph, step_id: str) -> Optional[str]:
    """Nearest ``plan`` step reachable backward over CHAIN edges, or None.

    A ``tool_action``'s chain predecessor IS its plan; a ``tool_observation``'s
    chain predecessor is its ``tool_action``, whose chain predecessor is the
    plan. A bounded backward walk over chain edges returns the first ``plan``
    node found (predecessors sorted for determinism).
    """
    visited: set = {step_id}
    frontier: List[str] = [step_id]
    while frontier:
        nxt: List[str] = []
        for cur in frontier:
            for src in _chain_predecessor_ids(graph, cur):
                if src in visited:
                    continue
                visited.add(src)
                node = graph.node(src)
                if node is not None and node.semantic_type == _PLAN_TYPE:
                    return src
                nxt.append(src)
        frontier = sorted(nxt)
    return None


def _is_superseded(
    graph: DependencyGraph,
    step_id: str,
    semantic_type: Optional[str],
    replay_set: set,
) -> bool:
    """True for a tool_action/tool_observation that will be REPLACED once
    the backend re-runs whatever produced it, rather than copied through
    unchanged. Two distinct paths lead here:

    1. NORMAL CASE: this step's owning plan (found via ``_owning_plan_id``,
       walking backward over chain edges) is in ``replay_step_ids`` — the
       whole plan was replayed, so every tool_action/tool_observation under
       it is replaced.
    2. UPSTREAM CARVE-OUT CASE: this step is a tool_observation whose DIRECT
       chain predecessor (the tool_action that produced it) is itself in
       ``replay_step_ids`` — this happens when ``rollback_optimizer.py``'s
       upstream rule replays the tool_action behind a faulty memory's cited
       observation DIRECTLY, WITHOUT necessarily replaying that tool_action's
       own governing plan (the plan may have nothing wrong with it; only the
       specific tool call was flagged). The plan-only check in case 1 misses
       this entirely.

    The flag is only ever computed for the two pass-through types
    (tool_action, tool_observation); every other semantic type returns False
    immediately.
    """
    if semantic_type not in _PASS_THROUGH_TYPES:
        return False

    # Case 1: owning plan replayed (the normal, whole-plan-regenerated case).
    plan_id = _owning_plan_id(graph, step_id)
    if plan_id is not None and plan_id in replay_set:
        return True

    # Case 2: a tool_observation whose OWN producing tool_action was replayed
    # directly (the upstream carve-out — see docstring). Only meaningful for
    # tool_observation; a tool_action has no "producing tool_action" of its
    # own to check.
    if semantic_type == "tool_observation":
        direct_predecessors = _chain_predecessor_ids(graph, step_id)
        if any(pred in replay_set for pred in direct_predecessors):
            return True

    return False


# --------------------------------------------------------------------------- #
# SETUP half: prepare agent state up to (not including) the first replay step
# --------------------------------------------------------------------------- #

def _reduce_agent_state(
    agent: "ReactAgent",
    graph: DependencyGraph,
    plan: Dict[str, Any],
    steps_in_order: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    session: List[str],
    start_position: int,
    stop_position: int,
) -> None:
    """Reduce trace steps in ``[start_position, stop_position)`` into ``agent``.

    The live agent is the sole state being reconstructed.  Old state effects
    are applied for pass-through, preserve, and suspicious steps; replay,
    redundant, and superseded steps are skipped because their replacement (or
    deliberate removal) owns the effect.
    """
    from agent.react_agent import HistoryTurn

    memory_by_id = {m["memory_id"]: m for m in memories}
    step_sets = _plan_step_sets(plan)
    replay_set = step_sets["replay"]
    deleted_ids = set(plan.get("delete_memory_ids", []))
    quarantined_ids = set(plan.get("quarantine_memory_ids", []))

    last_step_of_turn: Dict[int, int] = {}
    errored_actions: set = set()
    for position, step in enumerate(steps_in_order):
        node = graph.node(step.get("step_id"))
        if node is not None and node.turn is not None:
            last_step_of_turn[node.turn] = position
        if step.get("step_type") == "tool_observation":
            try:
                observation = json.loads(step.get("content") or "")
            except (TypeError, json.JSONDecodeError):
                observation = None
            if isinstance(observation, dict) and "error" in observation:
                errored_actions.update(step.get("prev_step_ids") or [])

    existing_history_ids = {h.user_input_id for h in agent.history}

    # Memories without a produce edge pre-date the trace.  Load them before
    # reducing steps so persistent session state is available at every cutoff.
    if start_position <= 0:
        root_records = []
        for record in memories:
            memory_id = record.get("memory_id")
            producers = [
                source for source, labels in graph.predecessor_edges(memory_id).items()
                if "produce" in labels
            ]
            if not producers and agent.memory_store.get_memory(memory_id) is None:
                root_records.append(dict(record))
        if root_records:
            agent.memory_store.load_memories(root_records)

    def load_generated(memory_id: str) -> None:
        if agent.memory_store.get_memory(memory_id) is not None:
            return
        record = memory_by_id.get(memory_id)
        if record is None:
            logger.warning("generated memory %s has no serialized record", memory_id)
            return
        as_created = dict(record)
        as_created["status"] = "active"
        agent.memory_store.load_memories([as_created])
        memory = agent.memory_store.get_memory(memory_id)
        if memory_id in deleted_ids:
            memory.status = "deleted"
        elif memory_id in quarantined_ids:
            memory.status = "quarantined"

        # A durable user_history memory replaces the transient history turn it
        # promoted.  Prefer the serialized author and fall back to trace cites.
        if memory.source == "user_history":
            source_ids = {memory.last_modified_by}
            agent.history[:] = [
                item for item in agent.history if item.user_input_id not in source_ids
            ]
            existing_history_ids.difference_update(source_ids)

    start = max(0, start_position)
    stop = min(len(steps_in_order), max(start, stop_position))
    for position in range(start, stop):
        step = steps_in_order[position]
        step_id = step.get("step_id")
        node = graph.node(step_id)
        semantic_type = node.semantic_type if node is not None else step.get("step_type")
        turn = node.turn if node is not None else None
        if turn is not None:
            agent.turn_index = turn

        disposition = _disposition_for(step_id, step_sets)
        superseded = _is_superseded(graph, step_id, semantic_type, replay_set)
        applies_old_effect = disposition in {
            "pass_through", "preserve", "suspicious"
        } and not superseded

        if applies_old_effect and semantic_type == "tool_action":
            tool_name = step.get("tool_name")
            if (
                tool_name
                and step_id not in errored_actions
                and agent._is_read_only(tool_name)
            ):
                agent.executed.add(
                    agent._call_key(tool_name, step.get("tool_args") or {})
                )

        if applies_old_effect and semantic_type in {
            "memory_write", "memory_update", "memory_consolidate"
        }:
            for memory_id in step.get("generated_memory_ids") or []:
                load_generated(memory_id)
            promoted_ids = {
                uid for uid in (step.get("used_ids") or [])
                if isinstance(uid, str) and uid.startswith("user_input_")
            }
            if promoted_ids:
                generated = [
                    agent.memory_store.get_memory(mid)
                    for mid in (step.get("generated_memory_ids") or [])
                ]
                if any(memory is not None and memory.source == "user_history" for memory in generated):
                    agent.history[:] = [
                        item for item in agent.history
                        if item.user_input_id not in promoted_ids
                    ]
                    existing_history_ids.difference_update(promoted_ids)

        if applies_old_effect and semantic_type in {
            "memory_update", "memory_consolidate"
        }:
            for memory_id in step.get("invalidated_memory_ids") or []:
                memory = agent.memory_store.get_memory(memory_id)
                if memory is not None and memory.status not in {
                    "deleted", "quarantined"
                }:
                    memory.status = "superseded"
        elif applies_old_effect and semantic_type == "memory_delete":
            for memory_id in step.get("invalidated_memory_ids") or []:
                memory = agent.memory_store.get_memory(memory_id)
                if memory is not None and memory.status != "quarantined":
                    memory.status = "deleted"

        if turn is not None and last_step_of_turn.get(turn) == position:
            user_input_id = f"user_input_{turn}"
            if user_input_id not in existing_history_ids and turn <= len(session):
                agent.history.append(
                    HistoryTurn(user_input_id=user_input_id, text=session[turn - 1])
                )
                existing_history_ids.add(user_input_id)


def setup_replay_state(
    agent: "ReactAgent",
    graph: DependencyGraph,
    plan: Dict[str, Any],
    steps_in_order: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    session: List[str],
) -> Dict[str, Any]:
    """Prepare agent state up to (not including) the first replay step.

    Steps:
      a. Clear the agent and reduce the trace prefix into its live state. Memory
         records are loaded only when their producing step is crossed.
      b. ``repair_ops.apply_memory_dispositions`` flips the plan's
         ``delete_memory_ids`` to ``deleted`` and ``quarantine_memory_ids`` to
         ``quarantined`` (soft, in place).
      c. Find the FIRST step id in ``steps_in_order`` that is in
         ``plan["replay_step_ids"]`` — the earliest replay point. If
         ``replay_step_ids`` is empty, WARN and treat the first replay step as
         ``None`` (nothing to replay — a legitimate, if unusual, plan).
      d. Set ``agent.task_id`` and the 1-based ``turn_index`` of the replay
         step. Restore completed prefix turns and successful read-only calls.

    Two snapshot fields are time-scoped to the first replay point using
    ``session`` (the raw per-turn prompt text, ``session[0]`` = turn 1):

      * ``history_preview`` reports every turn STRICTLY BEFORE the first replay
        step's turn (``node_turn``) — ``session[:node_turn - 1]`` paired with the
        ``user_input_{turn}`` id each turn minted (1-based). The current,
        not-yet-replayed turn is
        deliberately EXCLUDED (a turn's input is appended to history only AFTER
        it completes in ``react_agent.run()``). When there is no first replay
        step, no ``node_turn``, or ``node_turn == 1``, ``history_preview`` is
        correctly EMPTY — not an error.
      * ``memories_as_of_cutoff`` is a direct serialization of the live memory
        store after prefix reduction and plan dispositions. Future memories are
        absent; memories retired only by future steps retain their cutoff
        status.

    Returns a JSON-serializable snapshot (see the keys in the return dict).
    """
    # Earliest replay point: the first step in trace order that is in
    # replay_step_ids.
    replay_set = set(plan.get("replay_step_ids", []))
    first_replay_step_id: Optional[str] = None
    trace_prefix_length = len(steps_in_order)  # all-untouched if none matches
    for position, step in enumerate(steps_in_order):
        if step.get("step_id") in replay_set:
            first_replay_step_id = step.get("step_id")
            trace_prefix_length = position  # steps strictly before this one
            break

    if first_replay_step_id is None:
        logger.warning(
            "plan for task %s has no replay_step_ids that appear in "
            "steps_in_order; treating the first replay step as None (nothing to "
            "replay — a legitimate, if unusual, plan)",
            plan.get("task_id"),
        )

    # Reconstruct the one live state from trace time, never from final memories.
    # The reducer loads a memory only when its producing step is crossed, which
    # makes future records impossible to leak into the cutoff state.
    agent.clear()
    agent.task_id = plan.get("task_id")
    node_turn: Optional[int] = None
    if first_replay_step_id is not None:
        node = graph.node(first_replay_step_id)
        node_turn = node.turn if node is not None else None
    _reduce_agent_state(
        agent, graph, plan, steps_in_order, memories, session,
        start_position=0, stop_position=trace_prefix_length,
    )
    if node_turn is not None:
        # At step-level replay this is the current (1-based) turn, even when the
        # cutoff occurs before that turn's final step/history append.
        agent.turn_index = node_turn

    disposition_summary = repair_ops.apply_memory_dispositions(
        agent.memory_store,
        plan.get("delete_memory_ids", []),
        plan.get("quarantine_memory_ids", []),
    )
    memories_as_of_cutoff = agent.memory_store.get_memories()
    history_preview = [
        {"user_input_id": h.user_input_id, "text": h.text}
        for h in agent.history
    ]

    return {
        "first_replay_step_id": first_replay_step_id,
        "memories_as_of_cutoff": memories_as_of_cutoff,
        "turn_index": agent.turn_index,
        "history_preview": history_preview,
        "trace_prefix_length": trace_prefix_length,
        # Echoed for hand-inspection: confirms which ids were deleted vs
        # quarantined (so the snapshot is self-contained for verification).
        "memory_disposition_summary": disposition_summary,
    }


# --------------------------------------------------------------------------- #
# SEQUENCING half: build the ordered replay manifest (no execution)
# --------------------------------------------------------------------------- #

def build_replay_sequence(
    graph: DependencyGraph,
    plan: Dict[str, Any],
    steps_in_order: List[Dict[str, Any]],
    first_replay_step_id: str,
) -> List[Dict[str, Any]]:
    """Build the ordered replay manifest consumed by :func:`replay_chain`.

    Starting from ``first_replay_step_id``'s position in ``steps_in_order``,
    walk forward through the END of the trace. For EACH step in that range emit:

      ``step_id``                — the step id.
      ``semantic_type``          — the graph node's semantic type.
      ``disposition``            — replay / preserve / redundant / suspicious
                                   (read from the plan's four step lists), or
                                   pass_through for a tool_action /
                                   tool_observation / any step in no plan list
                                   (e.g. a step outside the affected region the
                                   walk reaches on its way to the end).
      ``chain_predecessor_ids``  — ``graph.predecessor_edges`` filtered to chain
                                   edges. This is the context replay must
                                   gather, NOT this list's own previous
                                   entry.
      ``context_predecessor_ids``— predecessors connected by chain OR cite;
                                   replay context resolution uses this union.
      ``superseded``             — True for a tool_action/tool_observation that
                                   will be replaced once the backend re-runs
                                   whatever produced it: EITHER its own
                                   plan is dispositioned replay (whole plan
                                   regenerated) OR its direct producing
                                   tool_action was replayed via the upstream
                                   carve-out (see ``_is_superseded``).

    Disposition is read directly from the plan's lists via the single shared
    ``_disposition_for`` lookup; no precedence is re-derived here (pure
    reporting). If ``first_replay_step_id`` is None or not found, an empty
    manifest is returned (nothing to replay).
    """
    if first_replay_step_id is None:
        return []

    start_position: Optional[int] = None
    for position, step in enumerate(steps_in_order):
        if step.get("step_id") == first_replay_step_id:
            start_position = position
            break
    if start_position is None:
        logger.warning(
            "first_replay_step_id %s not found in steps_in_order; returning an "
            "empty replay sequence",
            first_replay_step_id,
        )
        return []

    step_sets = _plan_step_sets(plan)
    replay_set = step_sets["replay"]

    sequence: List[Dict[str, Any]] = []
    for step in steps_in_order[start_position:]:
        step_id = step.get("step_id")
        node = graph.node(step_id)
        semantic_type = node.semantic_type if node is not None else None
        sequence.append(
            {
                "step_id": step_id,
                "semantic_type": semantic_type,
                "disposition": _disposition_for(step_id, step_sets),
                "chain_predecessor_ids": _chain_predecessor_ids(graph, step_id),
                "context_predecessor_ids": _context_predecessor_ids(
                    graph, step_id
                ),
                "superseded": _is_superseded(
                    graph, step_id, semantic_type, replay_set
                ),
            }
        )
    return sequence


# --------------------------------------------------------------------------- #
# Entry point for THIS PASS: setup + sequence, write both artifacts, no replay
# --------------------------------------------------------------------------- #

def execute_rollback_plan_setup_only(
    agent: "ReactAgent",
    graph: DependencyGraph,
    plan: Dict[str, Any],
    steps_in_order: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    session: List[str],
    output_dir: str,
) -> Dict[str, Any]:
    """Run the setup + sequencing halves, write both artifacts, return both.

    Calls :func:`setup_replay_state` (passing ``session``, the raw per-turn
    prompt text used to scope ``history_preview``) then
    :func:`build_replay_sequence`, writes ``replay_state_snapshot.json`` and
    ``replay_sequence.json`` to ``output_dir`` (one serializer, ``json.dump``
    directly), prints a summary, and returns the combined dict. It performs no
    replay operation and makes no model call.
    """
    snapshot = setup_replay_state(
        agent, graph, plan, steps_in_order, memories, session
    )
    first_replay_step_id = snapshot["first_replay_step_id"]
    sequence = build_replay_sequence(
        graph, plan, steps_in_order, first_replay_step_id
    )

    os.makedirs(output_dir, exist_ok=True)
    snapshot_path = os.path.join(output_dir, "replay_state_snapshot.json")
    sequence_path = os.path.join(output_dir, "replay_sequence.json")
    with open(snapshot_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, ensure_ascii=False)
    with open(sequence_path, "w", encoding="utf-8") as fh:
        json.dump(sequence, fh, indent=2, ensure_ascii=False)

    # Per-disposition count breakdown over the sequence. superseded is a flag
    # ORTHOGONAL to disposition (a pass_through step can also be superseded), so
    # it is counted separately rather than as a disposition bucket.
    disposition_counts: Dict[str, int] = {}
    for entry in sequence:
        disposition_counts[entry["disposition"]] = (
            disposition_counts.get(entry["disposition"], 0) + 1
        )
    superseded_count = sum(1 for entry in sequence if entry["superseded"])

    print(f"=== rollback executor (setup-only) for task {plan.get('task_id')} ===")
    print(f"  first replay step:    {first_replay_step_id}")
    print(f"  trace prefix length:  {snapshot['trace_prefix_length']}")
    print(f"  steps in replay seq:  {len(sequence)}")
    print(f"  turn_index:           {snapshot['turn_index']}")
    print("  memory dispositions:")
    summary = snapshot["memory_disposition_summary"]
    print(f"    deleted:     {summary['deleted']}")
    print(f"    quarantined: {summary['quarantined']}")
    print(f"    not_found:   {summary['not_found']}")
    print("  per-disposition counts:")
    for label in ("replay", "preserve", "redundant", "suspicious", "pass_through"):
        print(f"    {label:13s} {disposition_counts.get(label, 0)}")
    print(f"    {'superseded':13s} {superseded_count}")
    print(f"  wrote: {snapshot_path}")
    print(f"  wrote: {sequence_path}")

    return {
        "replay_state_snapshot": snapshot,
        "replay_sequence": sequence,
        "disposition_counts": disposition_counts,
        "superseded_count": superseded_count,
    }


# --------------------------------------------------------------------------- #
# REPLAY half: walk the chain, executing each step per its disposition
# --------------------------------------------------------------------------- #
# Replay operations are delegated exclusively through ReplayBackend. Context
# construction remains deterministic and does not call a model.

_MEMORY_MUTATION_TYPES = frozenset(
    {"memory_write", "memory_update", "memory_consolidate"}
)

def _max_step_number(steps_in_order: List[Dict[str, Any]]) -> int:
    """Highest numeric suffix among ``s_*`` step ids in the trace.

    Confirmed against react_agent.py: it has no reusable max-step utility
    (``restore_executed_from_trace`` walks steps but never tracks a max id; the
    logger mints ids internally), so this is implemented here. New replay ids
    are numbered from this value + 1, guaranteeing they never collide with an original id. Ids
    without a parseable trailing integer are ignored; an empty/nameless trace
    yields 0 (so the first new id is ``s_n001``).
    """
    best = 0
    for step in steps_in_order:
        sid = step.get("step_id")
        if not isinstance(sid, str):
            continue
        _, _, tail = sid.rpartition("_")
        if tail.startswith("n"):
            tail = tail[1:]
        if tail.isdigit():
            best = max(best, int(tail))
    return best


def _max_memory_number(memories: List[Dict[str, Any]]) -> int:
    """Highest numeric suffix among existing ``m_*``, ``m_f*``, and ``m_n*`` ids."""
    best = 0
    for memory in memories:
        memory_id = memory.get("memory_id")
        if not isinstance(memory_id, str) or not memory_id.startswith("m_"):
            continue
        suffix = memory_id[2:]
        if suffix.startswith(("f", "n")):
            suffix = suffix[1:]
        if suffix.isdigit():
            best = max(best, int(suffix))
    return best


def _finalize_memory_disposition_records(
    memory_store: Any,
    original_memories: List[Dict[str, Any]],
    plan: Dict[str, Any],
) -> None:
    """Materialize planned terminal records after replay has completed.

    Missing records are loaded only now, so future tombstones can never appear
    in ``ReplayContext.active_memories``. Deletion wins over quarantine.
    """
    original_by_id = {
        record.get("memory_id"): record
        for record in original_memories
        if isinstance(record.get("memory_id"), str)
    }
    deleted_ids = set(plan.get("delete_memory_ids", []))
    quarantined_ids = set(plan.get("quarantine_memory_ids", [])) - deleted_ids

    def finalize(memory_id: str, status: str) -> None:
        memory = memory_store.get_memory(memory_id)
        if memory is not None:
            if memory.status == "active":
                memory.status = status
            return
        original = original_by_id.get(memory_id)
        if original is None:
            logger.warning(
                "planned %s memory %s is absent from original memories",
                status,
                memory_id,
            )
            return
        tombstone = dict(original)
        tombstone["status"] = status
        memory_store.load_memories([tombstone])

    for memory_id in sorted(deleted_ids):
        finalize(memory_id, "deleted")
    for memory_id in sorted(quarantined_ids):
        finalize(memory_id, "quarantined")


def replay_chain(
    agent: "ReactAgent",
    graph: DependencyGraph,
    plan: Dict[str, Any],
    steps_in_order: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    session: List[str],
    snapshot: Dict[str, Any],
    *,
    runtime: "Any",
    backend: "Any",
    replay_context_observer: Optional[ReplayContextObserver] = None,
) -> Dict[str, Any]:
    """Replay through the required structured runtime/backend interface."""
    if runtime is None:
        raise TypeError("ReplayRuntime is required for replay_chain")
    if backend is None:
        raise TypeError("ReplayBackend is required for replay_chain")
    backend.reset_llm_replay_cost()
    from .selective_replay import (
        build_replay_context,
        normalize_observation_value,
    )

    first_id = snapshot.get("first_replay_step_id")
    prefix_length = int(snapshot.get("trace_prefix_length", 0))
    manifest = build_replay_sequence(graph, plan, steps_in_order, first_id)
    by_id = {step.get("step_id"): step for step in steps_in_order}
    positions = {step.get("step_id"): index for index, step in enumerate(steps_in_order)}
    invalidated_claim_ids: List[str] = []
    seen_invalidated_claim_ids: set[str] = set()
    replay_ids = set(plan.get("replay_step_ids", []))
    suspicious_ids = set(plan.get("suspicious_step_ids", []))
    for claim_id in plan.get("invalidate_claim_ids", []):
        if claim_id in seen_invalidated_claim_ids:
            raise ValueError(f"duplicate invalidate_claim_ids entry {claim_id!r}")
        seen_invalidated_claim_ids.add(claim_id)
        step = by_id.get(claim_id)
        if step is None:
            raise ValueError(
                f"invalidate_claim_ids references unknown original step {claim_id!r}"
            )
        node = graph.node(claim_id)
        semantic_type = node.semantic_type if node is not None else step.get("step_type")
        if semantic_type != "claim":
            raise ValueError(
                f"invalidate_claim_ids holds non-claim {claim_id!r} "
                f"(semantic_type={semantic_type!r})"
            )
        if claim_id not in replay_ids | suspicious_ids:
            raise ValueError(
                f"invalidated claim {claim_id!r} must be replayed or suspicious"
            )
        invalidated_claim_ids.append(claim_id)
    invalidated_claim_set = set(invalidated_claim_ids)

    repaired = [dict(step) for step in steps_in_order[:prefix_length]]
    historical_step_ids = {
        step["step_id"] for step in repaired
        if isinstance(step.get("step_id"), str)
    }
    replacement_step_ids: set[str] = set()
    current_content: Dict[str, Any] = {
        step.get("step_id"): step.get("content")
        for step in steps_in_order[:prefix_length]
        if step.get("step_id") not in invalidated_claim_set
    }
    step_remap: Dict[str, str] = {}
    memory_remap: Dict[str, str] = {}
    new_memory_ids: List[str] = []
    next_num = _max_step_number(steps_in_order) + 1
    next_memory_num = _max_memory_number(memories) + 1
    existing_step_ids = {
        step["step_id"] for step in steps_in_order
        if isinstance(step.get("step_id"), str)
    }
    existing_memory_ids = {
        memory["memory_id"] for memory in memories
        if isinstance(memory.get("memory_id"), str)
    }
    replayed = pass_through = superseded_skipped = 0
    full_trace_replay = "full_trace_replay" in str(plan.get("reason", ""))
    reconciliation_mutation_types = set(_MEMORY_MUTATION_TYPES)
    # Preserve the deterministic fixture path's historical mechanical delete.
    # Injected-LLM reconciliation owns all four semantic action types.
    if getattr(backend, "llm", None) is not None or full_trace_replay:
        reconciliation_mutation_types.add("memory_delete")
    mutation_steps_by_turn: Dict[int, List[Dict[str, Any]]] = {}
    protected_memory_outputs_by_turn: Dict[int, List[Dict[str, Any]]] = {}
    for entry in manifest:
        node = graph.node(entry["step_id"])
        if (
            entry["disposition"] == "replay"
            and entry["semantic_type"] in reconciliation_mutation_types
            and node is not None and node.turn is not None
        ):
            mutation_steps_by_turn.setdefault(node.turn, []).append(by_id[entry["step_id"]])
        elif (
            entry["disposition"] != "replay"
            and entry["semantic_type"] in _MEMORY_MUTATION_TYPES
            and node is not None and node.turn is not None
        ):
            step = by_id[entry["step_id"]]
            protected_memory_outputs_by_turn.setdefault(node.turn, []).append({
                "step_id": step.get("step_id"),
                "step_type": step.get("step_type"),
                "content": step.get("content"),
                "generated_memory_ids": list(step.get("generated_memory_ids") or []),
            })
    reconciled_turns: set = set()
    promoted_history_ids: set = set()
    adjacency_overrides: Dict[str, Dict[str, List[str]]] = {}
    removed_step_ids: set[str] = set()
    live_observations_by_turn: Dict[
        int, List[tuple[str, Any]]
    ] = {}

    original_final_steps = [
        step for step in steps_in_order
        if step.get("step_type") == "final_answer"
        and isinstance(step.get("step_id"), str)
    ]
    evaluated_final_id = (
        original_final_steps[-1]["step_id"] if original_final_steps else None
    )
    evaluated_node = graph.node(evaluated_final_id) if evaluated_final_id else None
    evaluated_turn = evaluated_node.turn if evaluated_node is not None else None
    evaluated_turn_steps = [
        step for step in steps_in_order
        if (graph.node(step.get("step_id")) is not None)
        and graph.node(step.get("step_id")).turn == evaluated_turn
    ] if evaluated_turn is not None else []
    original_plan_present = any(
        step.get("step_type") == "plan" for step in evaluated_turn_steps
    )
    original_tool_action_present = any(
        step.get("step_type") == "tool_action" for step in evaluated_turn_steps
    )
    faulty_memory_set = set(plan.get("faulty_memory_ids", []))
    exposed_faulty_memory_ids = sorted({
        identifier
        for step in evaluated_turn_steps
        if step.get("step_type") == "memory_read"
        for identifier in (step.get("used_ids") or [])
        if identifier in faulty_memory_set
    })
    omission_enabled = _FIXED_REPAIR_POLICY[
        "enable_final_turn_missing_action_replan"
    ]
    synthetic_read_only_only = _FIXED_REPAIR_POLICY[
        "synthetic_tools_read_only_only"
    ]
    omission_repair: Dict[str, Any] = {
        "enabled": omission_enabled,
        "triggered": False,
        "turn": evaluated_turn,
        "exposed_faulty_memory_ids": exposed_faulty_memory_ids,
        "original_plan_present": original_plan_present,
        "original_tool_action_present": original_tool_action_present,
        "synthetic_replan_attempted": False,
        "synthetic_plan_step_ids": [],
        "synthetic_tool_action_step_ids": [],
        "synthetic_tool_observation_step_ids": [],
        "synthetic_tool_call_count": 0,
    }
    omission_replan_consumed = False

    def record_live_observation(
        turn: int,
        observation_step_id: str,
        observation: Any,
    ) -> None:
        """Record one current observation without disturbing execution order."""
        resolved_id = _resolve_remap(observation_step_id, step_remap)
        if (
            observation_step_id in removed_step_ids
            or resolved_id in removed_step_ids
        ):
            return
        observation = normalize_observation_value(observation)

        observations = live_observations_by_turn.setdefault(turn, [])
        normalized: List[tuple[str, Any]] = []
        positions_by_id: Dict[str, int] = {}
        for recorded_id, recorded_value in observations:
            current_id = _resolve_remap(recorded_id, step_remap)
            if (
                recorded_id in removed_step_ids
                or current_id in removed_step_ids
            ):
                continue
            if current_id in positions_by_id:
                continue
            positions_by_id[current_id] = len(normalized)
            normalized.append((current_id, recorded_value))

        existing_position = positions_by_id.get(resolved_id)
        if existing_position is None:
            normalized.append((resolved_id, observation))
        else:
            # Preserve the first-seen position while refreshing both the
            # canonical ID and its replayed value.
            normalized[existing_position] = (resolved_id, observation)
        observations[:] = normalized

    def remove_live_observations(step_ids: Iterable[str]) -> None:
        """Remove stale raw and remapped observation IDs from every turn."""
        stale_ids = set(step_ids)
        stale_ids.update(
            _resolve_remap(step_id, step_remap) for step_id in tuple(stale_ids)
        )
        for observations in live_observations_by_turn.values():
            observations[:] = [
                (resolved_id, value)
                for observation_id, value in observations
                for resolved_id in [_resolve_remap(observation_id, step_remap)]
                if observation_id not in stale_ids
                and resolved_id not in stale_ids
                and observation_id not in removed_step_ids
                and resolved_id not in removed_step_ids
            ]

    def build_live_replay_context(step_id: str) -> ReplayContext:
        """Overlay graph context with authoritative effective tool results."""
        context = build_replay_context(
            runtime, graph, step_id, session, current_content, step_remap
        )
        # build_replay_context deliberately falls back to immutable graph
        # storage. Cancel that fallback for an invalidated old claim. Once the
        # claim has a replay replacement, its resolved new ID/content is safe.
        if context.claim_step_id in invalidated_claim_set:
            context.claim_step_id = None
            context.claim_text = None
        if context.turn in live_observations_by_turn:
            live_observations = live_observations_by_turn[context.turn]
            context.observation_step_ids = [
                observation_id for observation_id, _value in live_observations
            ]
            context.observations = [
                value for _observation_id, value in live_observations
            ]
        return context

    def observe(operation: str, context: ReplayContext) -> None:
        if replay_context_observer is not None:
            replay_context_observer(operation, context)

    def remap_used(ids: Any) -> List[str]:
        remapped: List[str] = []
        for uid in ids or []:
            if uid in invalidated_claim_set and uid not in step_remap:
                raise ValueError(
                    f"replay output cites invalidated claim {uid!r} before a "
                    "replacement is available"
                )
            step_id = _resolve_remap(uid, step_remap)
            remapped.append(_resolve_remap(step_id, memory_remap))
        return remapped

    def emit(old: Dict[str, Any], result: "Any", new_id: str) -> Dict[str, Any]:
        out = dict(old)
        out["step_id"] = new_id
        out["content"] = result.content
        out["used_ids"] = remap_used(result.used_ids)
        step_remap[old["step_id"]] = new_id
        current_content[old["step_id"]] = result.content
        current_content[new_id] = result.content
        repaired.append(out)
        replacement_step_ids.add(new_id)
        return out

    def split_tool_targets(
        superseded_ids: List[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split superseded step ids into tool_action and tool_observation
        target records (originals from ``by_id``). Only ``tool_action`` records
        are handed to the LLM as replay targets; their paired observations are
        resolved deterministically inside ``repair_ops.replay_targeted_tools``.
        """
        actions = [
            by_id[step_id] for step_id in superseded_ids
            if step_id in by_id and by_id[step_id].get("step_type") == "tool_action"
        ]
        observations = [
            by_id[step_id] for step_id in superseded_ids
            if step_id in by_id
            and by_id[step_id].get("step_type") == "tool_observation"
        ]
        return actions, observations

    def integrate_tool_result(
        tool_result: "Any", superseded_step_ids: List[str], turn: int
    ) -> None:
        """Merge generated tools with the superseded steps' graph boundaries.

        Target correspondence is now explicit: the deterministic tool executor
        supplies ``step_id_remap`` (each replaced target action/observation ->
        its replacement) and ``removed_step_ids`` (targets whose slot was
        dropped). The executor no longer re-derives that correspondence by
        cardinality or call-key heuristics; it VALIDATES that the mapped and
        removed target IDs exactly partition the superseded targets — separately
        for tool_action and tool_observation targets — then owns trace
        integration (inherited adjacency preserves every external branch).
        """
        nonlocal next_num
        superseded = {
            step_id for step_id in superseded_step_ids if step_id in by_id
        }

        def targets_of_type(step_type: str) -> set[str]:
            return {
                step_id for step_id in superseded
                if by_id[step_id].get("step_type") == step_type
            }

        calculated_remap = {
            old_id: new_id
            for old_id, new_id in tool_result.step_id_remap.items()
            if old_id in superseded
        }
        removed_now = set(tool_result.removed_step_ids) & superseded
        mapped_old_ids = set(calculated_remap)

        if removed_now:
            raise ValueError(
                "mandatory targeted tool replay cannot remove selected targets: "
                f"{sorted(removed_now)!r}"
            )

        # Explicit target correspondence must map every selected action and
        # observation. No positional or call-key matching of unrelated ops.
        if mapped_old_ids & removed_now:
            raise ValueError(
                "targeted tool replay mapped and removed the same step: "
                f"{sorted(mapped_old_ids & removed_now)!r}"
            )
        for step_type in ("tool_action", "tool_observation"):
            targets = targets_of_type(step_type)
            covered = (mapped_old_ids | removed_now) & targets
            if covered != targets:
                missing = sorted(targets - covered)
                raise ValueError(
                    f"targeted tool replay {step_type} targets are not fully "
                    f"partitioned by mapped+removed IDs: missing={missing}"
                )
        stray = (mapped_old_ids | removed_now) - superseded
        if stray:
            raise ValueError(
                "targeted tool replay referenced non-target steps: "
                f"{sorted(stray)!r}"
            )

        tool_result.step_id_remap = calculated_remap
        tool_result.removed_step_ids = [
            step_id for step_id in superseded_step_ids if step_id in removed_now
        ]
        removed_step_ids.update(removed_now)
        step_remap.update(calculated_remap)
        remove_live_observations(removed_now)

        external_predecessors = list(dict.fromkeys(
            predecessor
            for step_id in superseded_step_ids if step_id in by_id
            for predecessor in (by_id[step_id].get("prev_step_ids") or [])
            if predecessor not in superseded
        ))
        external_successors = list(dict.fromkeys(
            successor
            for step_id in superseded_step_ids if step_id in by_id
            for successor in (by_id[step_id].get("next_step_ids") or [])
            if successor not in superseded
        ))

        original_by_new_id = {
            new_id: by_id[old_id]
            for old_id, new_id in calculated_remap.items()
            if old_id in by_id
        }
        normalized_steps = []
        for backend_step in tool_result.new_steps:
            inherited = original_by_new_id.get(backend_step.get("step_id"), {})
            merged = dict(inherited)
            merged.update(backend_step)
            for field in _STEP_REFERENCE_FIELDS:
                inherited_ids = inherited.get(field) or []
                backend_ids = backend_step.get(field) or []
                merged[field] = list(dict.fromkeys(
                    list(inherited_ids) + list(backend_ids)
                ))
            normalized = _normalize_replacement_step(merged, step_remap, memory_remap)
            for field in _STEP_REFERENCE_FIELDS:
                normalized[field] = list(dict.fromkeys(normalized[field]))
            normalized_steps.append(normalized)

        new_chain_ids = [
            step["step_id"] for step in normalized_steps
            if step.get("step_type") in {"tool_action", "tool_observation"}
        ]
        mapped_predecessors = [
            _resolve_remap(identifier, step_remap)
            for identifier in external_predecessors
        ]
        mapped_successors = [
            _resolve_remap(identifier, step_remap)
            for identifier in external_successors
        ]

        def rewired_adjacency(
            identifier: str, field: str, replacements: List[str]
        ) -> List[str]:
            resolved = _resolve_remap(identifier, step_remap)
            current = next((
                step for step in repaired if step.get("step_id") == resolved
            ), by_id.get(identifier, {}))
            result: List[str] = []
            inserted = False
            for item in current.get(field) or []:
                if item in superseded:
                    if not inserted:
                        result.extend(replacements)
                        inserted = True
                else:
                    result.append(item)
            if not inserted:
                result.extend(replacements)
            return list(dict.fromkeys(result))

        if new_chain_ids:
            first_new, last_new = new_chain_ids[0], new_chain_ids[-1]
            adjacency_overrides.setdefault(first_new, {})["prev_step_ids"] = mapped_predecessors
            adjacency_overrides.setdefault(last_new, {})["next_step_ids"] = mapped_successors
            for predecessor in external_predecessors:
                resolved = _resolve_remap(predecessor, step_remap)
                adjacency_overrides.setdefault(resolved, {})["next_step_ids"] = rewired_adjacency(
                    predecessor, "next_step_ids", [first_new]
                )
            for successor in external_successors:
                resolved = _resolve_remap(successor, step_remap)
                adjacency_overrides.setdefault(resolved, {})["prev_step_ids"] = rewired_adjacency(
                    successor, "prev_step_ids", [last_new]
                )
        else:
            for predecessor in external_predecessors:
                resolved = _resolve_remap(predecessor, step_remap)
                adjacency_overrides.setdefault(resolved, {})["next_step_ids"] = rewired_adjacency(
                    predecessor, "next_step_ids", external_successors
                )
            for successor in external_successors:
                resolved = _resolve_remap(successor, step_remap)
                adjacency_overrides.setdefault(resolved, {})["prev_step_ids"] = rewired_adjacency(
                    successor, "prev_step_ids", external_predecessors
                )

        repaired.extend(normalized_steps)
        replacement_step_ids.update(
            step["step_id"] for step in normalized_steps
        )
        next_num = tool_result.next_new_step_num
        for step in normalized_steps:
            sid = step.get("step_id")
            if isinstance(sid, str):
                current_content[sid] = step.get("content")
        for old_id, mapped_id in calculated_remap.items():
            current_content[old_id] = current_content.get(mapped_id)

        # Presence of the turn key is authoritative even when no tool ran.
        # This prevents a later context build from recovering removed old
        # observations through the immutable original dependency graph.
        live_observations_by_turn.setdefault(turn, [])
        generated_observations = [
            step for step in normalized_steps
            if step.get("step_type") == "tool_observation"
        ]
        replay_values = list(getattr(tool_result, "observations", []) or [])
        for index, observation_step in enumerate(generated_observations):
            observation_id = observation_step.get("step_id")
            if not isinstance(observation_id, str):
                continue
            value = (
                replay_values[index]
                if index < len(replay_values)
                else observation_step.get("content")
            )
            record_live_observation(turn, observation_id, value)

    def maybe_replan_missing_actions(step_id: str, turn: int) -> Optional[str]:
        """Run the exact local final-turn omission trigger at most once."""
        nonlocal next_num, omission_replan_consumed
        if (
            omission_replan_consumed
            or not omission_enabled
            or step_id != evaluated_final_id
            or step_id not in replay_ids
        ):
            return None
        active_ids = {
            memory.get("memory_id") for memory in runtime.active_memories()
            if isinstance(memory.get("memory_id"), str)
        }
        # Planning is forbidden until every known faulty memory has disappeared
        # from the live active store.
        if active_ids.intersection(faulty_memory_set):
            return None

        omission_replan_consumed = True
        omission_repair["triggered"] = True
        omission_repair["synthetic_replan_attempted"] = True
        context = build_live_replay_context(step_id)
        feedback = ""
        for attempt in range(3):
            synthetic_plan = backend.replan_missing_actions(
                runtime, context, feedback=feedback
            )
            allowed_evidence = {
                memory.get("memory_id") for memory in context.active_memories
                if isinstance(memory.get("memory_id"), str)
            }
            allowed_evidence.update(
                turn_item.get("user_input_id") for turn_item in context.history
                if isinstance(turn_item.get("user_input_id"), str)
            )
            allowed_evidence.update(context.observation_step_ids)
            if context.user_input_id:
                allowed_evidence.add(context.user_input_id)
            if context.claim_step_id:
                allowed_evidence.add(context.claim_step_id)
            synthetic_plan.used_ids = list(dict.fromkeys(
                identifier for identifier in synthetic_plan.used_ids
                if identifier in allowed_evidence
            ))
            synthetic_plan.used_ids = remap_used(synthetic_plan.used_ids)
            if not synthetic_plan.tool_calls:
                return None
            observe("replan_missing_actions", context)
            synthetic_plan.content = json.dumps(
                [
                    {
                        "type": "tool",
                        "tool_name": call.tool_name,
                        "args": dict(call.args),
                    }
                    for call in synthetic_plan.tool_calls
                ],
                ensure_ascii=False,
            )

            plan_step_id = f"s_n{next_num:03d}"
            plan_step: Dict[str, Any] = {
                "step_id": plan_step_id,
                "task_id": runtime.task_id,
                "step_type": "plan",
                "content": synthetic_plan.content,
                "prev_step_ids": [context.claim_step_id] if context.claim_step_id else [],
                "next_step_ids": [],
                "used_ids": list(synthetic_plan.used_ids),
            }
            observe("execute_synthetic_tools", context)
            try:
                tool_result = repair_ops.execute_synthetic_tool_calls(
                    runtime,
                    synthetic_plan,
                    plan_step_id,
                    next_num + 1,
                    read_only_only=synthetic_read_only_only,
                )
                next_num += 1
                break
            except ValueError as exc:
                if (
                    "synthetic missing-action tool is not read-only" not in str(exc)
                    or attempt + 1 >= 3
                ):
                    raise
                feedback = (
                    f"Previous missing-action plan failed execution validation: {exc}. "
                    "This final-turn omission repair may only add read-only "
                    "lookup/search/check/compare tools. Do not plan side-effecting "
                    "tools such as create/open/book/update/delete. If no read-only "
                    "tool is required, return an empty steps list."
                )
        else:
            raise RuntimeError("unreachable synthetic missing-action retry state")
        next_num = tool_result.next_new_step_num
        if tool_result.new_steps:
            plan_step["next_step_ids"] = [tool_result.new_steps[0]["step_id"]]
        synthetic_steps = [plan_step] + list(tool_result.new_steps)
        repaired.extend(synthetic_steps)
        replacement_step_ids.update(
            step["step_id"] for step in synthetic_steps
        )
        existing_step_ids.update(step["step_id"] for step in synthetic_steps)
        for synthetic_step in synthetic_steps:
            current_content[synthetic_step["step_id"]] = synthetic_step.get("content")

        if context.claim_step_id:
            claim_id = context.claim_step_id
            claim_step = next((
                item for item in repaired if item.get("step_id") == claim_id
            ), {})
            old_successors = list(claim_step.get("next_step_ids") or [])
            rewired = [
                plan_step_id if item in {step_id, evaluated_final_id} else item
                for item in old_successors
            ]
            if plan_step_id not in rewired:
                rewired.append(plan_step_id)
            adjacency_overrides.setdefault(claim_id, {})["next_step_ids"] = list(
                dict.fromkeys(rewired)
            )

        action_ids = [
            step["step_id"] for step in tool_result.new_steps
            if step.get("step_type") == "tool_action"
        ]
        observation_steps = [
            step for step in tool_result.new_steps
            if step.get("step_type") == "tool_observation"
        ]
        observation_ids = [step["step_id"] for step in observation_steps]
        omission_repair["synthetic_plan_step_ids"] = [plan_step_id]
        omission_repair["synthetic_tool_action_step_ids"] = action_ids
        omission_repair["synthetic_tool_observation_step_ids"] = observation_ids
        omission_repair["synthetic_tool_call_count"] = len(action_ids)
        live_observations_by_turn.setdefault(turn, [])
        for index, observation_step in enumerate(observation_steps):
            value = (
                tool_result.observations[index]
                if index < len(tool_result.observations)
                else observation_step.get("content")
            )
            record_live_observation(turn, observation_step["step_id"], value)
        return observation_ids[-1] if observation_ids else None

    def reconcile(turn: int, trigger_id: str) -> None:
        nonlocal next_num, next_memory_num
        old_mutations = mutation_steps_by_turn.get(turn, [])
        if turn in reconciled_turns or not old_mutations:
            return
        context = build_live_replay_context(trigger_id)
        blocked_memory_ids = set(plan.get("delete_memory_ids", [])) | set(
            plan.get("quarantine_memory_ids", [])
        )
        remapped_mutations: List[Dict[str, Any]] = []
        dropped_target_ids: set[str] = set()
        generated_old_ids_so_far: set[str] = set()
        for old_step in old_mutations:
            remapped_step, _dropped_ids = remap_memory_mutation_step(
                old_step, memory_remap, runtime.memory_store, blocked_memory_ids
            )
            step_type = remapped_step.get("step_type")
            invalidated = remapped_step.get("invalidated_memory_ids") or []
            if (
                step_type in {"memory_delete", "memory_update", "memory_consolidate"}
                and not invalidated
            ):
                batch_local_targets = [
                    memory_id for memory_id in (old_step.get("invalidated_memory_ids") or [])
                    if isinstance(memory_id, str)
                    and memory_id in generated_old_ids_so_far
                ]
                if batch_local_targets:
                    remapped_step["invalidated_memory_ids"] = batch_local_targets
                    invalidated = batch_local_targets
                    remapped_step["used_ids"] = [
                        identifier
                        for identifier in (old_step.get("used_ids") or [])
                        if not (
                            isinstance(identifier, str)
                            and identifier.startswith("m_")
                            and identifier not in generated_old_ids_so_far
                        )
                    ]
            if (
                step_type in {"memory_delete", "memory_update", "memory_consolidate"}
                and not invalidated
            ):
                old_step_id = old_step.get("step_id")
                if isinstance(old_step_id, str):
                    dropped_target_ids.add(old_step_id)
                continue
            remapped_mutations.append(remapped_step)
            generated_old_ids_so_far.update(
                memory_id for memory_id in (old_step.get("generated_memory_ids") or [])
                if isinstance(memory_id, str)
            )

        if dropped_target_ids:
            removed_step_ids.update(dropped_target_ids)
        if not remapped_mutations:
            reconciled_turns.add(turn)
            return

        target_step_ids = [
            step["step_id"]
            for step in remapped_mutations
            if isinstance(step.get("step_id"), str)
        ]
        context.trigger_step_id = trigger_id
        context.target_step_ids = target_step_ids
        context.protected_memory_outputs = list(
            protected_memory_outputs_by_turn.get(turn, [])
        )
        # The trigger is the repaired final answer itself, so expose it even
        # though dependency search only walks backward from the old graph node.
        context.final_answer_text = str(current_content.get(trigger_id, context.final_answer_text))
        context.final_answer_step_id = step_remap.get(trigger_id, trigger_id)
        observe("reconcile_memories", context)
        result = backend.reconcile_memories(
            runtime, context, remapped_mutations, next_num, next_memory_num
        )

        expected_targets = set(context.target_step_ids)
        mapped_targets = set(result.step_id_remap)
        removed_targets = set(result.removed_step_ids)
        if len(removed_targets) != len(result.removed_step_ids):
            raise ValueError("memory reconciliation removed_step_ids must be unique")
        if removed_targets.intersection(expected_targets):
            raise ValueError(
                "mandatory memory reconciliation cannot remove selected targets"
            )
        if mapped_targets != expected_targets:
            missing = sorted(expected_targets - mapped_targets)
            unexpected = sorted(mapped_targets - expected_targets)
            raise ValueError(
                "memory reconciliation step mapping mismatch: "
                f"missing={missing}, unexpected={unexpected}; every selected "
                "target requires one replacement"
            )

        new_steps_by_id = {
            step["step_id"]: step
            for step in result.new_steps
            if isinstance(step.get("step_id"), str)
        }
        generated_step_ids = [step.get("step_id") for step in result.new_steps]
        if len(result.new_steps) != len(expected_targets):
            raise ValueError(
                "mandatory memory reconciliation must produce exactly one "
                "replacement step per selected target"
            )
        expected_step_ids = [
            f"s_n{number:03d}"
            for number in range(next_num, next_num + len(result.new_steps))
        ]
        if generated_step_ids != expected_step_ids:
            raise ValueError(
                "memory reconciliation step IDs must begin at the supplied "
                f"next step number and be contiguous: expected={expected_step_ids!r}, "
                f"actual={generated_step_ids!r}"
            )
        if len(generated_step_ids) != len(set(generated_step_ids)) or any(
            step_id in existing_step_ids for step_id in generated_step_ids
        ):
            raise ValueError("memory reconciliation produced duplicate or colliding step IDs")

        expected_memory_ids = [
            f"m_n{number:03d}"
            for number in range(
                next_memory_num, next_memory_num + len(result.new_memory_ids)
            )
        ]
        if result.new_memory_ids != expected_memory_ids:
            raise ValueError(
                "memory reconciliation memory IDs must begin at the supplied "
                f"next memory number and be contiguous: expected={expected_memory_ids!r}, "
                f"actual={result.new_memory_ids!r}"
            )
        if len(result.new_memory_ids) != len(set(result.new_memory_ids)) or any(
            memory_id in existing_memory_ids for memory_id in result.new_memory_ids
        ):
            raise ValueError("memory reconciliation produced duplicate or colliding memory IDs")
        for old_id, new_id in result.step_id_remap.items():
            if old_id not in expected_targets:
                raise ValueError(
                    "invalid memory reconciliation step mapping "
                    f"{old_id!r} -> {new_id!r}: old step is not one of "
                    f"the reconciliation targets {context.target_step_ids!r}"
                )
            if new_id not in new_steps_by_id:
                raise ValueError(
                    "invalid memory reconciliation step mapping "
                    f"{old_id!r} -> {new_id!r}: new step is absent from "
                    "MemoryReplayResult.new_steps"
                )
        if len(set(result.step_id_remap.values())) != len(result.step_id_remap):
            raise ValueError(
                "memory reconciliation cannot map multiple old steps to one new step"
            )
        if set(result.step_id_remap.values()) != set(new_steps_by_id):
            raise ValueError(
                "memory reconciliation cannot introduce an unbound mutation step"
            )
        expected_new_memory_count = sum(
            1 for step in remapped_mutations if step.get("step_type") != "memory_delete"
        )
        if len(result.new_memory_ids) != expected_new_memory_count:
            raise ValueError(
                "mandatory memory reconciliation produced the wrong number of "
                "replacement memories"
            )

        # Reconciliation backends provide semantic deltas. Rebuild every
        # mapped replacement from its original target so omitted trace
        # structure is retained, then let every explicitly supplied backend
        # field override the inherited value.
        step_remap.update(result.step_id_remap)
        memory_remap.update(result.memory_id_remap)
        removed_step_ids.update(result.removed_step_ids)
        remapped_original_by_id = {
            step["step_id"]: step
            for step in remapped_mutations
            if isinstance(step.get("step_id"), str)
        }
        original_by_new_id = {
            new_id: remapped_original_by_id[old_id]
            for old_id, new_id in result.step_id_remap.items()
        }
        normalized_new_steps = []
        for backend_step in result.new_steps:
            inherited = original_by_new_id.get(backend_step.get("step_id"), {})
            merged = dict(inherited)
            merged.update(backend_step)
            normalized_new_steps.append(
                _normalize_replacement_step(merged, step_remap, memory_remap)
            )
            adjacency_overrides[backend_step["step_id"]] = {
                field: list(normalized_new_steps[-1][field])
                for field in _STEP_REFERENCE_FIELDS if field in backend_step
            }
        normalized_new_steps_by_id = {
            step["step_id"]: step for step in normalized_new_steps
        }
        generated_memory_ids = [
            memory_id
            for step in normalized_new_steps
            for memory_id in step["generated_memory_ids"]
        ]
        if (
            len(generated_memory_ids) != len(set(generated_memory_ids))
            or set(generated_memory_ids) != set(result.new_memory_ids)
        ):
            raise ValueError(
                "memory reconciliation generated_memory_ids must match new_memory_ids"
            )

        repaired.extend(normalized_new_steps)
        replacement_step_ids.update(
            step["step_id"] for step in normalized_new_steps
        )
        existing_step_ids.update(generated_step_ids)
        existing_memory_ids.update(result.new_memory_ids)
        for memory_id in result.new_memory_ids:
            if memory_id not in new_memory_ids:
                new_memory_ids.append(memory_id)
        for step in normalized_new_steps:
            sid = step.get("step_id")
            if isinstance(sid, str):
                current_content[sid] = step.get("content")
        next_num += len(result.new_steps)
        next_memory_num += len(result.new_memory_ids)
        for old_id, new_id in result.step_id_remap.items():
            current_content[old_id] = normalized_new_steps_by_id[new_id].get("content")
        promoted = {
            uid for step in normalized_new_steps for uid in step["used_ids"]
            if isinstance(uid, str) and uid.startswith("user_input_")
        }
        if promoted:
            promoted_history_ids.update(promoted)
            runtime.history[:] = [item for item in runtime.history if item.user_input_id not in promoted]
        reconciled_turns.add(turn)

    def commit_replayed_position(position: Optional[int]) -> None:
        if position is not None:
            _reduce_agent_state(
                agent,
                graph,
                plan,
                steps_in_order,
                memories,
                session,
                position,
                position + 1,
            )
            if promoted_history_ids:
                runtime.history[:] = [
                    item for item in runtime.history
                    if item.user_input_id not in promoted_history_ids
                ]

    for entry in manifest:
        step_id = entry["step_id"]
        old = by_id.get(step_id, {"step_id": step_id})
        node = graph.node(step_id)
        turn = node.turn if node is not None and node.turn is not None else 1
        runtime.turn_index = turn
        agent.turn_index = turn
        position = positions.get(step_id)

        if entry["superseded"]:
            current_content.setdefault(step_id, old.get("content"))
            superseded_skipped += 1
            if entry["disposition"] == "replay":
                commit_replayed_position(position)
            continue

        if entry["disposition"] != "replay":
            if step_id not in invalidated_claim_set:
                current_content[step_id] = old.get("content")
            copied = dict(old)
            repaired.append(copied)
            historical_step_ids.add(step_id)
            pass_through += 1
            if entry["semantic_type"] == "tool_observation":
                current_step_id = _resolve_remap(step_id, step_remap)
                observation = current_content.get(
                    current_step_id,
                    current_content.get(step_id, copied.get("content")),
                )
                record_live_observation(
                    turn,
                    current_step_id,
                    observation,
                )
            if entry["semantic_type"] == "final_answer":
                reconcile(turn, step_id)
            if position is not None:
                _reduce_agent_state(agent, graph, plan, steps_in_order, memories, session, position, position + 1)
            continue

        semantic_type = entry["semantic_type"]
        # Old per-step mutations are replaced by the one turn reconciliation.
        if semantic_type in reconciliation_mutation_types:
            replayed += 1
            commit_replayed_position(position)
            continue

        synthetic_tail_id: Optional[str] = None
        if semantic_type == "final_answer":
            synthetic_tail_id = maybe_replan_missing_actions(step_id, turn)
        context = build_live_replay_context(step_id)

        # The optimizer may replay a tool action directly (without replaying
        # its owning plan). This is MECHANICAL direct replay, not LLM plan
        # regeneration: build one explicit replace decision targeting exactly
        # this action with its original tool name/args, then execute it (and its
        # paired observation) through the deterministic tool executor.
        if semantic_type == "tool_action":
            from .replay_types import PlanReplayResult, TargetedToolDecision
            direct_superseded = [step_id] + [
                candidate["step_id"] for candidate in manifest
                if candidate.get("superseded")
                and step_id in candidate.get("chain_predecessor_ids", [])
            ]
            target_actions, target_observations = split_tool_targets(
                direct_superseded
            )
            direct_plan = PlanReplayResult(
                content="[]",
                used_ids=remap_used(old.get("used_ids")),
                tool_decisions=[
                    TargetedToolDecision(
                        target_action_step_id=step_id,
                        tool_name=old.get("tool_name", ""),
                        args=old.get("tool_args") or {},
                    )
                ],
            )
            observe("replay_tools", context)
            tool_result = repair_ops.replay_targeted_tools(
                runtime, direct_plan, target_actions, target_observations,
                next_num,
            )
            integrate_tool_result(tool_result, direct_superseded, turn)
            replayed += 1
            commit_replayed_position(position)
            continue

        new_id = f"s_n{next_num:03d}"
        next_num += 1
        if semantic_type == "claim":
            observe("replay_claim", context)
            result = backend.replay_claim(runtime, context, old)
        elif semantic_type == "plan":
            superseded_ids = [
                candidate["step_id"] for candidate in manifest
                if candidate.get("superseded")
                and _owning_plan_id(graph, candidate["step_id"]) == step_id
            ]
            target_actions, target_observations = split_tool_targets(
                superseded_ids
            )
            observe("replay_plan", context)
            result = backend.replay_plan(
                runtime, context, old, target_actions, target_observations
            )
        elif semantic_type == "final_answer":
            observe("replay_final_answer", context)
            result = backend.replay_final_answer(runtime, context, old)
        elif semantic_type == "memory_read":
            from .replay_types import ReplayStepResult
            observe("memory_read", context)
            result = ReplayStepResult(
                content=f"Retrieved {len(runtime.active_memories())} relevant memories.",
                used_ids=[item["memory_id"] for item in runtime.active_memories()],
            )
        elif semantic_type == "memory_delete":
            observe("memory_delete", context)
            repair_ops.replay_memory_delete(runtime.memory_store, old.get("invalidated_memory_ids", []))
            from .replay_types import ReplayStepResult
            # The target remains explicit in invalidated_memory_ids; it is not
            # valid active evidence after deletion.
            result = ReplayStepResult(content=old.get("content"), used_ids=[])
        else:
            logger.warning("unknown replay semantic type %r for %s; preserving", semantic_type, step_id)
            current_content[step_id] = old.get("content")
            repaired.append(dict(old))
            historical_step_ids.add(step_id)
            pass_through += 1
            commit_replayed_position(position)
            continue
        result.used_ids = remap_used(result.used_ids)
        emit(old, result, new_id)
        if semantic_type == "final_answer" and synthetic_tail_id is not None:
            adjacency_overrides.setdefault(new_id, {})["prev_step_ids"] = [
                synthetic_tail_id
            ]
            adjacency_overrides.setdefault(synthetic_tail_id, {})[
                "next_step_ids"
            ] = [new_id]
        replayed += 1

        if semantic_type == "plan":
            tool_context = build_live_replay_context(step_id)
            observe("replay_tools", tool_context)
            tool_result = repair_ops.replay_targeted_tools(
                runtime, result, target_actions, target_observations, next_num
            )
            integrate_tool_result(tool_result, superseded_ids, turn)
        if semantic_type == "final_answer":
            reconcile(turn, step_id)
        commit_replayed_position(position)

    # A malformed trace may omit a final answer. Reconcile affected turns once
    # at the end rather than replaying each old mutation independently.
    for turn in sorted(mutation_steps_by_turn):
        if turn not in reconciled_turns:
            trigger = mutation_steps_by_turn[turn][0]["step_id"]
            reconcile(turn, trigger)

    agent.history = runtime.history
    agent.executed = runtime.executed
    agent.memory_store = runtime.memory_store
    _finalize_memory_disposition_records(
        runtime.memory_store, memories, plan
    )
    # Mappings are discovered incrementally. Structural references are repaired
    # everywhere, but only generated replacements receive memory-ID remapping.
    repaired = [
        (
            _normalize_replacement_step(step, step_remap, memory_remap)
            if step.get("step_id") in replacement_step_ids
            else _normalize_historical_step(step, step_remap)
        )
        for step in repaired
        if step.get("step_id") not in removed_step_ids
    ]
    for step in repaired:
        for field in _STEP_REFERENCE_FIELDS:
            step[field] = [
                identifier for identifier in step[field]
                if identifier not in removed_step_ids
            ]
        step["used_ids"] = [
            identifier for identifier in step["used_ids"]
            if identifier not in removed_step_ids
        ]
    repaired_by_id = {step["step_id"]: step for step in repaired}
    resolved_overrides: Dict[str, Dict[str, List[str]]] = {}
    for step_id, fields in adjacency_overrides.items():
        resolved_step_id = _resolve_remap(step_id, step_remap)
        resolved_overrides.setdefault(resolved_step_id, {}).update({
            field: list(dict.fromkeys(
                _resolve_remap(identifier, step_remap)
                for identifier in identifiers
                if _resolve_remap(identifier, step_remap) not in removed_step_ids
            ))
            for field, identifiers in fields.items()
        })
    for step_id, fields in resolved_overrides.items():
        step = repaired_by_id.get(step_id)
        if step is None:
            continue
        for field, identifiers in fields.items():
            step[field] = identifiers
        if "prev_step_ids" in fields:
            desired_predecessors = set(step["prev_step_ids"])
            for candidate in repaired:
                if candidate["step_id"] not in desired_predecessors:
                    candidate["next_step_ids"] = [
                        item for item in candidate["next_step_ids"]
                        if item != step_id
                    ]
            for predecessor_id in step["prev_step_ids"]:
                predecessor = repaired_by_id.get(predecessor_id)
                if predecessor is not None and step_id not in predecessor["next_step_ids"]:
                    predecessor["next_step_ids"].append(step_id)
        if "next_step_ids" in fields:
            desired_successors = set(step["next_step_ids"])
            for candidate in repaired:
                if candidate["step_id"] not in desired_successors:
                    candidate["prev_step_ids"] = [
                        item for item in candidate["prev_step_ids"]
                        if item != step_id
                    ]
            for successor_id in step["next_step_ids"]:
                successor = repaired_by_id.get(successor_id)
                if successor is not None and step_id not in successor["prev_step_ids"]:
                    successor["prev_step_ids"].append(step_id)
    repaired_memories = runtime.memory_store.get_memories()
    active_memory_ids = {
        memory.get("memory_id")
        for memory in repaired_memories
        if memory.get("status") == "active"
        and isinstance(memory.get("memory_id"), str)
    }
    for step in repaired:
        if step.get("step_id") not in replacement_step_ids:
            continue
        step["used_ids"] = [
            identifier for identifier in step["used_ids"]
            if not (
                isinstance(identifier, str)
                and identifier.startswith("m_")
                and identifier not in active_memory_ids
            )
        ]
    _validate_repaired_trace(
        repaired, step_remap, memory_remap, removed_step_ids,
        historical_step_ids=historical_step_ids,
        replacement_step_ids=replacement_step_ids,
        repaired_memories=repaired_memories,
    )
    return {
        "repaired_tracesteps": repaired,
        "repaired_memories": repaired_memories,
        "new_memory_ids": new_memory_ids,
        "memory_id_remap": memory_remap,
        "step_id_remap": step_remap,
        "removed_step_ids": sorted(removed_step_ids),
        "invalidated_claim_ids": invalidated_claim_ids,
        "omission_repair": omission_repair,
        "generated_replacement_step_ids": [
            step["step_id"] for step in repaired
            if step.get("step_id") in replacement_step_ids
        ],
        "llm_replay_call_count": backend.llm_replay_call_count,
        "llm_replay_calls_by_operation": dict(
            backend.llm_replay_calls_by_operation
        ),
        "llm_token_usage": deepcopy(backend.llm_token_usage),
        "summary": {"replayed": replayed, "pass_through": pass_through, "superseded_skipped": superseded_skipped},
    }


def _validated_id_list(plan: Dict[str, Any], field: str) -> List[str]:
    """Copy a validated plan ID list for the result artifact."""
    value = plan.get(field, [])
    if not isinstance(value, list) or any(
        not isinstance(identifier, str) or not identifier
        for identifier in value
    ):
        raise ValueError(f"rollback plan field {field!r} must be a list of IDs")
    return list(value)


def _validated_llm_operation_counts(
    value: Any,
    *,
    field: str,
) -> Dict[str, int]:
    """Copy an LLM operation-count map after strict schema validation."""
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dict")
    if any(
        not isinstance(operation, str)
        or not operation
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for operation, count in value.items()
    ):
        raise ValueError(
            f"{field} must map non-empty strings to non-negative integers"
        )
    return dict(value)


def _validated_llm_call_count(value: Any, *, field: str) -> int:
    """Validate one aggregate LLM call count without accepting bools."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


_LLM_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


def _zero_llm_phase_usage(*, unavailable_call_count: int = 0) -> Dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "usage_available_call_count": 0,
        "usage_unavailable_call_count": unavailable_call_count,
        "by_operation": {},
    }


def _validated_llm_phase_usage(
    value: Any,
    *,
    actual_call_count: int,
    field: str,
) -> Dict[str, Any]:
    """Validate and detach one phase's exact API token accounting."""
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dict")
    totals = {
        name: _validated_llm_call_count(value.get(name), field=f"{field}.{name}")
        for name in _LLM_TOKEN_FIELDS
    }
    available = _validated_llm_call_count(
        value.get("usage_available_call_count"),
        field=f"{field}.usage_available_call_count",
    )
    unavailable = _validated_llm_call_count(
        value.get("usage_unavailable_call_count"),
        field=f"{field}.usage_unavailable_call_count",
    )
    if totals["input_tokens"] + totals["output_tokens"] != totals["total_tokens"]:
        raise ValueError(f"{field} input_tokens + output_tokens must equal total_tokens")
    if available + unavailable != actual_call_count:
        raise ValueError(
            f"{field} available plus unavailable calls must equal actual LLM call count"
        )

    raw_operations = value.get("by_operation")
    if not isinstance(raw_operations, dict):
        raise ValueError(f"{field}.by_operation must be a dict")
    operations: Dict[str, Dict[str, int]] = {}
    for operation, raw_usage in raw_operations.items():
        operation_field = f"{field}.by_operation[{operation!r}]"
        if not isinstance(operation, str) or not operation:
            raise ValueError(f"{field}.by_operation keys must be non-empty strings")
        if not isinstance(raw_usage, dict):
            raise ValueError(f"{operation_field} must be a dict")
        operation_usage = {
            name: _validated_llm_call_count(
                raw_usage.get(name), field=f"{operation_field}.{name}"
            )
            for name in _LLM_TOKEN_FIELDS
        }
        operation_usage["call_count"] = _validated_llm_call_count(
            raw_usage.get("call_count"), field=f"{operation_field}.call_count"
        )
        if (
            operation_usage["input_tokens"]
            + operation_usage["output_tokens"]
            != operation_usage["total_tokens"]
        ):
            raise ValueError(
                f"{operation_field} input_tokens + output_tokens must equal total_tokens"
            )
        operations[operation] = operation_usage

    for token_field in _LLM_TOKEN_FIELDS:
        if sum(item[token_field] for item in operations.values()) != totals[token_field]:
            raise ValueError(
                f"{field} by-operation {token_field} must sum to phase total"
            )
    if sum(item["call_count"] for item in operations.values()) != available:
        raise ValueError(
            f"{field} by-operation call_count must sum to available call count"
        )
    return {
        **totals,
        "usage_available_call_count": available,
        "usage_unavailable_call_count": unavailable,
        "by_operation": operations,
    }


def _combined_llm_token_usage(
    decision: Dict[str, Any], replay: Dict[str, Any],
) -> Dict[str, Any]:
    combined = {
        field: decision[field] + replay[field]
        for field in _LLM_TOKEN_FIELDS
    }
    combined["usage_available_call_count"] = (
        decision["usage_available_call_count"]
        + replay["usage_available_call_count"]
    )
    combined["usage_unavailable_call_count"] = (
        decision["usage_unavailable_call_count"]
        + replay["usage_unavailable_call_count"]
    )
    combined["by_phase"] = {
        "repair_decision": decision,
        "replay": replay,
    }
    return combined


def _validate_llm_accounting(
    *,
    llm_repair_decision_call_count: Any,
    llm_repair_decision_calls_by_operation: Any,
    llm_replay_call_count: Any,
    llm_replay_calls_by_operation: Any,
    total_llm_call_count: Any,
) -> None:
    """Enforce the phase and total LLM accounting invariants."""
    decision_count = _validated_llm_call_count(
        llm_repair_decision_call_count,
        field="llm_repair_decision_call_count",
    )
    decision_operations = _validated_llm_operation_counts(
        llm_repair_decision_calls_by_operation,
        field="llm_repair_decision_calls_by_operation",
    )
    replay_count = _validated_llm_call_count(
        llm_replay_call_count,
        field="llm_replay_call_count",
    )
    replay_operations = _validated_llm_operation_counts(
        llm_replay_calls_by_operation,
        field="llm_replay_calls_by_operation",
    )
    total_count = _validated_llm_call_count(
        total_llm_call_count,
        field="total_llm_call_count",
    )
    if sum(decision_operations.values()) != decision_count:
        raise ValueError(
            "repair-decision LLM operation counts must sum to "
            "llm_repair_decision_call_count"
        )
    if sum(replay_operations.values()) != replay_count:
        raise ValueError(
            "replay LLM operation counts must sum to llm_replay_call_count"
        )
    if total_count != decision_count + replay_count:
        raise ValueError(
            "total_llm_call_count must equal repair-decision plus replay "
            "LLM call counts"
        )


def build_repaired_result_payload(
    plan: Dict[str, Any],
    original_trace_steps: List[Dict[str, Any]],
    original_memories: List[Dict[str, Any]],
    repaired_trace_steps: List[Dict[str, Any]],
    repaired_memories: List[Dict[str, Any]],
    replay_result: Dict[str, Any],
    *,
    llm_repair_decision_calls_by_operation: Optional[Dict[str, int]] = None,
    llm_repair_decision_token_usage: Optional[Dict[str, Any]] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    artifact_filenames: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build the evaluation-ready, ground-truth-free method artifact.

    All inputs are explicit so this pure transformation never rereads pipeline
    files or reaches into replay state. The returned tree contains only JSON
    value types and keeps replay execution facts separate from later metrics.
    """
    final_answers = [
        step for step in repaired_trace_steps
        if step.get("step_type") == "final_answer"
    ]
    if not final_answers:
        raise ValueError("repaired trace contains no final_answer step")
    final_answer = final_answers[-1]

    faulty_memory_ids = _validated_id_list(plan, "faulty_memory_ids")
    planned_delete_ids = _validated_id_list(plan, "delete_memory_ids")
    planned_quarantine_ids = _validated_id_list(
        plan, "quarantine_memory_ids"
    )
    planned_invalidate_ids = _validated_id_list(
        plan, "invalidate_claim_ids"
    )
    planned_replay_ids = _validated_id_list(plan, "replay_step_ids")

    method = plan.get("method", "ours")
    task_id = plan.get("task_id", replay_result.get("task_id"))
    config = runtime_config or {}
    domain = config.get("domain")

    original_memory_by_id = {
        memory.get("memory_id"): memory for memory in original_memories
        if isinstance(memory.get("memory_id"), str)
    }
    final_memory_by_id = {
        memory.get("memory_id"): memory for memory in repaired_memories
        if isinstance(memory.get("memory_id"), str)
    }
    selected_fault_types = [
        original_memory_by_id[memory_id].get("fault_type")
        for memory_id in faulty_memory_ids
        if memory_id in original_memory_by_id
    ]
    fault_types = sorted({
        value if isinstance(value, str) and value else "unknown"
        for value in selected_fault_types
    })

    original_active_ids = [
        memory["memory_id"] for memory in original_memories
        if memory.get("status") == "active"
        and isinstance(memory.get("memory_id"), str)
    ]
    final_active_ids = [
        memory["memory_id"] for memory in repaired_memories
        if memory.get("status") == "active"
        and isinstance(memory.get("memory_id"), str)
    ]
    final_active_set = set(final_active_ids)
    preserved_original_ids = [
        memory_id for memory_id in original_active_ids
        if memory_id in final_active_set
    ]
    non_active_statuses = {"deleted", "quarantined", "superseded"}
    deactivated_ids = [
        memory_id for memory_id in original_active_ids
        if final_memory_by_id.get(memory_id, {}).get("status")
        in non_active_statuses
    ]

    step_remap_value = replay_result.get("step_id_remap", {})
    memory_remap_value = replay_result.get("memory_id_remap", {})
    if not isinstance(step_remap_value, dict):
        raise ValueError("replay result step_id_remap must be a dict")
    if not isinstance(memory_remap_value, dict):
        raise ValueError("replay result memory_id_remap must be a dict")
    step_remap = dict(step_remap_value)
    memory_remap = dict(memory_remap_value)

    planned_replay_set = set(planned_replay_ids)
    replayed_target_ids = [
        step["step_id"] for step in original_trace_steps
        if step.get("step_id") in planned_replay_set
        and step.get("step_id") in step_remap
    ]
    replayed_target_set = set(replayed_target_ids)
    replacement_id_set = {
        replacement_id for replacement_id in step_remap.values()
        if isinstance(replacement_id, str) and replacement_id
    }
    generated_value = replay_result.get("generated_replacement_step_ids")
    generated_replacement_ids = (
        list(generated_value)
        if isinstance(generated_value, list)
        else [
            step["step_id"] for step in repaired_trace_steps
            if step.get("step_id") in replacement_id_set
        ]
    )
    omission_repair = dict(replay_result.get("omission_repair") or {})
    synthetic_tool_ids = list(
        omission_repair.get("synthetic_tool_action_step_ids") or []
    )
    original_step_ids = {
        step.get("step_id") for step in original_trace_steps
        if isinstance(step.get("step_id"), str)
    }
    repaired_step_ids = {
        step.get("step_id") for step in repaired_trace_steps
        if isinstance(step.get("step_id"), str)
    }
    preserved_original_step_count = len(
        original_step_ids & repaired_step_ids
    )
    decision_operation_counts = (
        {}
        if llm_repair_decision_calls_by_operation is None
        else llm_repair_decision_calls_by_operation
    )
    llm_repair_decision_calls_by_operation = _validated_llm_operation_counts(
        decision_operation_counts,
        field="llm_repair_decision_calls_by_operation",
    )
    llm_repair_decision_call_count = sum(
        llm_repair_decision_calls_by_operation.values()
    )
    llm_replay_call_count = _validated_llm_call_count(
        replay_result.get("llm_replay_call_count", 0),
        field="replay result llm_replay_call_count",
    )
    llm_replay_calls_by_operation = _validated_llm_operation_counts(
        replay_result.get("llm_replay_calls_by_operation", {}),
        field="replay result llm_replay_calls_by_operation",
    )
    if sum(llm_replay_calls_by_operation.values()) != llm_replay_call_count:
        raise ValueError(
            "replay result LLM operation counts must sum to llm_replay_call_count"
        )
    total_llm_call_count = (
        llm_repair_decision_call_count + llm_replay_call_count
    )
    _validate_llm_accounting(
        llm_repair_decision_call_count=llm_repair_decision_call_count,
        llm_repair_decision_calls_by_operation=(
            llm_repair_decision_calls_by_operation
        ),
        llm_replay_call_count=llm_replay_call_count,
        llm_replay_calls_by_operation=llm_replay_calls_by_operation,
        total_llm_call_count=total_llm_call_count,
    )
    raw_decision_usage = (
        _zero_llm_phase_usage(
            unavailable_call_count=llm_repair_decision_call_count
        )
        if llm_repair_decision_token_usage is None
        else llm_repair_decision_token_usage
    )
    decision_token_usage = _validated_llm_phase_usage(
        raw_decision_usage,
        actual_call_count=llm_repair_decision_call_count,
        field="repair-decision LLM token usage",
    )
    raw_replay_usage = replay_result.get("llm_token_usage")
    if raw_replay_usage is None:
        raw_replay_usage = _zero_llm_phase_usage(
            unavailable_call_count=llm_replay_call_count
        )
    replay_token_usage = _validated_llm_phase_usage(
        raw_replay_usage,
        actual_call_count=llm_replay_call_count,
        field="replay LLM token usage",
    )
    llm_token_usage = _combined_llm_token_usage(
        decision_token_usage, replay_token_usage
    )
    internal_summary = replay_result.get("summary", {})
    if not isinstance(internal_summary, dict):
        raise ValueError("replay result summary must be a dict")

    original_tool_action_ids = [
        step["step_id"] for step in original_trace_steps
        if step.get("step_type") == "tool_action"
        and isinstance(step.get("step_id"), str)
    ]
    original_tool_action_set = set(original_tool_action_ids)
    planned_replay_tool_ids = [
        step_id for step_id in planned_replay_ids
        if step_id in original_tool_action_set
    ]
    replayed_tool_ids = [
        step_id for step_id in original_tool_action_ids
        if step_id in replayed_target_set
    ]

    applied_invalidated_value = replay_result.get(
        "invalidated_claim_ids", []
    )
    if not isinstance(applied_invalidated_value, list):
        raise ValueError("replay result invalidated_claim_ids must be a list")
    applied_invalidated_ids = list(applied_invalidated_value)
    replayed_invalidated_ids = [
        claim_id for claim_id in applied_invalidated_ids
        if claim_id in replayed_target_set
    ]
    invalidated_remap = {
        claim_id: step_remap[claim_id]
        for claim_id in replayed_invalidated_ids
        if claim_id in step_remap
    }

    filenames = {
        "repaired_tracesteps": "repaired_tracesteps.jsonl",
        "repaired_memories": "repaired_memories.jsonl",
        "replay_context": "replay_context.json",
        "rollback_plan": "rollback_plan.json",
    }
    if artifact_filenames:
        filenames.update(artifact_filenames)
    for name, path in filenames.items():
        if not isinstance(path, str) or not path:
            raise ValueError(f"artifact path {name!r} must be a non-empty string")
        if os.path.isabs(path):
            raise ValueError(f"artifact path {name!r} must be relative")

    case_parts = [
        str(domain) if domain is not None else "unknown",
        "+".join(fault_types) if fault_types else "unknown",
        str(task_id) if task_id is not None else "unknown",
        "+".join(sorted(faulty_memory_ids)) if faulty_memory_ids else "none",
    ]
    case_id = plan.get("case_id") or "/".join(case_parts)
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("rollback plan case_id must be a non-empty string")
    payload = {
        "case": {
            "method": method,
            "task_id": task_id,
            "domain": domain,
            "fault_types": fault_types,
            "faulty_memory_ids": faulty_memory_ids,
            "case_id": case_id,
        },
        "recovery": {
            "final_answer_step_id": final_answer.get("step_id"),
            "final_answer": final_answer.get("content"),
        },
        "memory_repair": {
            "original_active_memory_ids": original_active_ids,
            "final_active_memory_ids": final_active_ids,
            "preserved_original_active_memory_ids": preserved_original_ids,
            "planned_delete_memory_ids": planned_delete_ids,
            "planned_quarantine_memory_ids": planned_quarantine_ids,
            "deactivated_by_repair_memory_ids": deactivated_ids,
            "deactivation_status_by_id": {
                memory_id: final_memory_by_id[memory_id]["status"]
                for memory_id in deactivated_ids
            },
            "new_memory_ids": list(replay_result.get("new_memory_ids", [])),
            "memory_id_remap": memory_remap,
        },
        "claim_repair": {
            "planned_invalidate_claim_ids": planned_invalidate_ids,
            "applied_invalidated_claim_ids": applied_invalidated_ids,
            "replayed_invalidated_claim_ids": replayed_invalidated_ids,
            "invalidated_claim_id_remap": invalidated_remap,
        },
        "replay_cost": {
            "total_original_steps": len(original_trace_steps),
            "planned_replay_step_ids": planned_replay_ids,
            "replayed_target_step_ids": replayed_target_ids,
            "replayed_target_step_count": len(replayed_target_ids),
            "total_original_tool_action_ids": original_tool_action_ids,
            "total_original_tool_call_count": len(original_tool_action_ids),
            "planned_replay_tool_action_ids": planned_replay_tool_ids,
            "replayed_tool_action_ids": replayed_tool_ids,
            "replayed_tool_call_count": len(replayed_tool_ids),
            "synthetic_tool_action_step_ids": synthetic_tool_ids,
            "synthetic_tool_call_count": len(synthetic_tool_ids),
            "total_repair_tool_call_count": (
                len(replayed_tool_ids) + len(synthetic_tool_ids)
            ),
            "generated_replacement_step_ids": generated_replacement_ids,
            "generated_replacement_step_count": len(
                generated_replacement_ids
            ),
            "llm_repair_decision_call_count": (
                llm_repair_decision_call_count
            ),
            "llm_repair_decision_calls_by_operation": dict(
                llm_repair_decision_calls_by_operation
            ),
            "llm_replay_call_count": llm_replay_call_count,
            "llm_replay_calls_by_operation": dict(
                llm_replay_calls_by_operation
            ),
            "total_llm_call_count": total_llm_call_count,
            "llm_token_usage": llm_token_usage,
            "step_id_remap": step_remap,
        },
        "plan": {
            "preserve_step_ids": _validated_id_list(plan, "preserve_step_ids"),
            "redundant_step_ids": _validated_id_list(plan, "redundant_step_ids"),
            "suspicious_step_ids": _validated_id_list(plan, "suspicious_step_ids"),
            "reason": plan.get("reason"),
        },
        "artifacts": filenames,
        "summary": {
            "replayed_target_step_count": len(replayed_target_ids),
            "preserved_original_step_count": preserved_original_step_count,
            "replay_suffix_pass_through_count": int(
                internal_summary.get("pass_through", 0)
            ),
            "superseded_original_step_count": int(
                internal_summary.get("superseded_skipped", 0)
            ),
            "generated_replacement_step_count": len(
                generated_replacement_ids
            ),
        },
    }
    # Fail here, close to construction, if a caller supplies a non-JSON replay
    # value such as a set, dataclass, Memory object, or callable.
    json.dumps(payload, ensure_ascii=False)
    return payload


def execute_rollback_plan(
    agent: "ReactAgent",
    graph: DependencyGraph,
    plan: Dict[str, Any],
    steps_in_order: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    session: List[str],
    output_dir: str,
    *,
    runtime: "Any",
    backend: "Any",
    llm_repair_decision_calls_by_operation: Optional[Dict[str, int]] = None,
    replay_context_observer: Optional[ReplayContextObserver] = None,
) -> Dict[str, Any]:
    """Complete rollback through the caller-supplied runtime and backend.

    1. ``setup_replay_state`` prepares agent state and returns the snapshot. It
       ALREADY calls ``repair_ops.apply_memory_dispositions`` internally (step
       (b) of its body), so this function does NOT call it a second time —
       doing so would double-flip already-deleted/quarantined records.
    2. ``replay_chain`` walks from the first replay step to the end of trace
       using only the supplied :class:`ReplayBackend` operations.
    3. On success, writes ``repaired_memories.jsonl``, ``repaired_tracesteps.jsonl``,
       and ``repaired_result.json`` to ``output_dir`` (one serializer per file,
       the same convention as every other stage).

    ``execute_rollback_plan_setup_only`` is left unchanged for diagnostic use.
    """
    if runtime is None:
        raise TypeError("ReplayRuntime is required for execute_rollback_plan")
    if backend is None:
        raise TypeError("ReplayBackend is required for execute_rollback_plan")

    snapshot = setup_replay_state(
        agent, graph, plan, steps_in_order, memories, session
    )

    # setup_replay_state clears/reconstructs the temporary agent, replacing its
    # containers. Point the caller's configured runtime at that reconstructed
    # state while preserving its tools/model/configuration.
    runtime.memory_store = agent.memory_store
    runtime.history = agent.history
    runtime.executed = agent.executed
    runtime.task_id = agent.task_id
    runtime.turn_index = agent.turn_index

    # setup_replay_state already applied the plan's memory dispositions; do NOT
    # re-apply them here.
    result = replay_chain(
        agent, graph, plan, steps_in_order, memories, session, snapshot,
        runtime=runtime,
        backend=backend,
        replay_context_observer=replay_context_observer,
    )

    os.makedirs(output_dir, exist_ok=True)
    memories_path = os.path.join(output_dir, "repaired_memories.jsonl")
    tracesteps_path = os.path.join(output_dir, "repaired_tracesteps.jsonl")
    result_path = os.path.join(output_dir, "repaired_result.json")

    with open(memories_path, "w", encoding="utf-8") as fh:
        for memory in result["repaired_memories"]:
            fh.write(json.dumps(memory, ensure_ascii=False) + "\n")
    with open(tracesteps_path, "w", encoding="utf-8") as fh:
        for step in result["repaired_tracesteps"]:
            fh.write(json.dumps(step, ensure_ascii=False) + "\n")
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(
            build_repaired_result_payload(
                plan,
                steps_in_order,
                memories,
                result["repaired_tracesteps"],
                result["repaired_memories"],
                result,
                llm_repair_decision_calls_by_operation=(
                    llm_repair_decision_calls_by_operation
                ),
                runtime_config=getattr(runtime, "config", None),
            ),
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(f"=== rollback executor (full) for task {plan.get('task_id')} ===")
    print(f"  first replay step:    {snapshot['first_replay_step_id']}")
    print(f"  trace prefix length:  {snapshot['trace_prefix_length']}")
    print(f"  replayed:             {result['summary']['replayed']}")
    print(f"  pass_through:         {result['summary']['pass_through']}")
    print(f"  superseded_skipped:   {result['summary']['superseded_skipped']}")
    print(f"  new memory ids:       {result['new_memory_ids']}")
    print(f"  wrote: {memories_path}")
    print(f"  wrote: {tracesteps_path}")
    print(f"  wrote: {result_path}")

    return result


# --------------------------------------------------------------------------- #
# CLI (mirrors rollback_optimizer.py: argparse, small _load helpers, a printed
# summary, json.dump output written by execute_rollback_plan_setup_only)
# --------------------------------------------------------------------------- #

def _load_graph(path: str) -> DependencyGraph:
    with open(path, "r", encoding="utf-8") as fh:
        return DependencyGraph.from_dict(json.load(fh))


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts (blank lines skipped, utf-8)."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _records_for_task(rows: List[Dict[str, Any]], task_id: str) -> List[Dict[str, Any]]:
    """Keep only rows whose ``task_id`` matches (mirrors run_ours.py's filter)."""
    return [r for r in rows if r.get("task_id") == task_id]


def _load_session(path: str, task_id: str) -> List[str]:
    """Load the session utterance list for ``task_id`` (session[0] is turn 1).

    Same loading convention as ``run_ours._load_session``: filter the session
    JSONL to ``task_id`` and return the first matching record's ``session`` list
    (the raw per-turn prompt text), or ``[]`` if there is no matching record.
    """
    session_recs = _records_for_task(_load_jsonl(path), task_id)
    return session_recs[0].get("session", []) if session_recs else []


def _make_executor_state_agent(
    *,
    tools: Optional[Dict[str, Callable[..., object]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> "ReactAgent":
    """Build the temporary holder for reconstructed executor state.

    This object supplies only ``MemoryStore``, history, and executed-call state.
    It is not the selective-replay operation backend; callers must provide a
    separate ``ReplayBackend`` for full replay.
    """
    from agent.react_agent import ReactAgent

    return ReactAgent(tools=dict(tools or {}), config=dict(config or {}))


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6a: prepare agent state up to the first replay step "
                    "and build the ordered replay manifest, writing both as "
                    "inspectable artifacts. NO replay and NO LLM call — this is "
                    "the setup + sequencing half of selective replay."
    )
    parser.add_argument(
        "--graph", required=True,
        help="dependency_graph.json path (DependencyGraph.to_dict output)",
    )
    parser.add_argument(
        "--plan", required=True,
        help="rollback_plan.json path (rollback_optimizer.py output)",
    )
    parser.add_argument(
        "--steps", required=True,
        help="trace steps JSONL path (the ORIGINAL, pre-disposition trace)",
    )
    parser.add_argument(
        "--memories", required=True,
        help="memories JSONL path (the ORIGINAL, pre-disposition memory list)",
    )
    parser.add_argument(
        "--session", required=True,
        help="tasks_full_session JSONL path — the raw per-turn prompt text used "
             "to scope history_preview (session[0] is turn 1)",
    )
    parser.add_argument(
        "--output_dir", default="outputs/{task_id}",
        help="directory to write replay_state_snapshot.json and "
             "replay_sequence.json into. {task_id} is resolved from the plan "
             "(default: outputs/{task_id}).",
    )
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    plan = _load_json(args.plan)
    task_id = plan.get("task_id") or graph.task_id

    steps_in_order = _records_for_task(_load_jsonl(args.steps), task_id)
    memories = _records_for_task(_load_jsonl(args.memories), task_id)
    session = _load_session(args.session, task_id)

    output_dir = args.output_dir.format(task_id=task_id)

    agent = _make_executor_state_agent()
    execute_rollback_plan_setup_only(
        agent, graph, plan, steps_in_order, memories, session, output_dir
    )


if __name__ == "__main__":
    _main()
