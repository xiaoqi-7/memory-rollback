# methods/ours/repair_ops.py
"""No-LLM, purely mechanical memory operations used by the rollback executor.

Where this sits in the pipeline
================================
``rollback_optimizer.py`` produces a plan (delete/quarantine/replay/preserve/
redundant/suspicious). This module performs the MECHANICAL half of executing
that plan against a live ``MemoryStore`` — the parts that involve no LLM call
and no decision-making whatsoever:

* applying the plan's memory dispositions (soft delete / quarantine),
* the read-side of a replay step (the existing relevant-memories query), and
* the mechanical replay of a ``memory_delete`` step whose target was NOT already
  handled by the plan's own dispositions.

Every function here REUSES ``MemoryStore``'s existing mutation/query methods
rather than inventing a new persistence path: ``delete_memory`` for a soft
delete, a direct ``status`` flip for quarantine (the store has no quarantine
verb), ``get_memories`` for the active read. Nothing here calls the LLM; the
LLM-driven re-derivation of claims / answers is ``selective_replay.py``'s job
(not built yet).

Status mutations are SOFT
=========================
Memories are NEVER hard-removed from the store. ``delete_memory`` flips
``status`` to ``"deleted"``; quarantine flips it to ``"quarantined"``. Both keep
the record in ``memory_store.memories`` so the disposition is inspectable after
the fact and so a memory the agent later cites still resolves to a record (with
a non-active status) rather than vanishing.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from .replay_types import (
    JSONRecord,
    JSONValue,
    PlanReplayResult,
    SyntheticPlanResult,
    TargetedToolDecision,
    ToolReplayResult,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; MemoryStore is used only in
    # annotations, which `from __future__ import annotations` never evaluates at
    # runtime, so importing it lazily here keeps this module import-light and
    # avoids a runtime dependency on the agent package's import path.
    from agent.memory_store import MemoryStore
    from .replay_runtime import ReplayRuntime

logger = logging.getLogger(__name__)

# The status a quarantined memory carries. MemoryStore.delete_memory already
# owns the "deleted" flip; there is no quarantine verb on the store, so the one
# place that literal lives is here.
_QUARANTINED_STATUS = "quarantined"


def apply_memory_dispositions(
    memory_store: MemoryStore,
    delete_memory_ids: List[str],
    quarantine_memory_ids: List[str],
) -> Dict[str, Any]:
    """Soft-delete / quarantine the plan's memory dispositions (no hard removal).

    ``delete_memory_ids`` are flipped to ``status="deleted"`` via the store's own
    :meth:`MemoryStore.delete_memory` (the existing soft-delete path — it sets
    ``status`` and ``last_modified_at`` and never pops the record). Each id in
    ``quarantine_memory_ids`` is flipped to ``status="quarantined"`` directly on
    its record (the store has no quarantine method, so this is the single place
    that flip is made — mirroring ``delete_memory``'s shape, not a new
    persistence path).

    Delete takes precedence: an id appearing in BOTH lists is deleted, not
    quarantined (the rollback optimizer's precedence already keeps the six plan
    lists disjoint, so this is only a defensive tie-break). An id with no
    matching record lands in ``not_found`` and is reported, not raised on, so one
    stale id never aborts applying the rest of the plan.

    Returns ``{"deleted": [...], "quarantined": [...], "not_found": [...]}`` with
    each list sorted for stable, diffable output.
    """
    deleted: List[str] = []
    quarantined: List[str] = []
    not_found: List[str] = []

    delete_set = set(delete_memory_ids)

    for memory_id in delete_set:
        if memory_store.get_memory(memory_id) is None:
            not_found.append(memory_id)
            continue
        # Reuse the store's existing soft-delete (status -> "deleted").
        memory_store.delete_memory(memory_id)
        deleted.append(memory_id)

    for memory_id in quarantine_memory_ids:
        if memory_id in delete_set:
            # Already handled as a delete above; delete wins the tie.
            continue
        memory = memory_store.get_memory(memory_id)
        if memory is None:
            not_found.append(memory_id)
            continue
        memory.status = _QUARANTINED_STATUS
        memory.last_modified_at = _now()
        quarantined.append(memory_id)

    return {
        "deleted": sorted(deleted),
        "quarantined": sorted(quarantined),
        "not_found": sorted(set(not_found)),
    }


def _now() -> Any:
    """Return a fresh ``last_modified_at`` value matching the store's convention.

    ``MemoryStore.delete_memory`` stamps ``datetime.now()``; the quarantine flip
    has no store method to borrow from, so mirror that exact stamp here.
    """
    from datetime import datetime

    return datetime.now()


def replay_memory_read(memory_store: MemoryStore) -> List[Dict[str, Any]]:
    """Return the store's current active-memory view (the read-side of a replay).

    Reuses :meth:`MemoryStore.get_memories`, which serializes every record the
    store holds, then keeps only the ``active`` ones — i.e. the memories visible
    to the agent AFTER ``apply_memory_dispositions`` has flipped the faulty /
    derived records to ``deleted`` / ``quarantined``. This is the existing query
    the store exposes; no new ranking or filtering logic is written here (a
    replay step's semantic re-derivation, including any relevance ranking, is
    ``selective_replay.py``'s job — this function only reports the active set a
    ``memory_read`` step would see).

    Returns a list of serialized memory dicts (``Memory.to_dict`` shape).
    """
    return [m for m in memory_store.get_memories() if m.get("status") == "active"]


def replay_memory_delete(
    memory_store: MemoryStore, invalidated_memory_ids: List[str]
) -> Dict[str, Any]:
    """Mechanically replay a ``memory_delete`` step's effect (no LLM).

    Used when a ``memory_delete`` step sits in the plan's ``replay_step_ids`` but
    its target was NOT already torn down by the plan's own
    ``delete_memory_ids`` / ``quarantine_memory_ids`` (the rollback optimizer
    routes an already-covered delete to ``redundant_step_ids`` instead, so this
    is the genuinely-still-needed case). ``invalidated_memory_ids`` is the step's
    own ``invalidated_memory_ids`` field — the memory/memories it tears down.

    Applies the SAME soft-delete as :func:`apply_memory_dispositions` (status ->
    ``"deleted"`` via :meth:`MemoryStore.delete_memory`) for any target that is
    not already ``deleted`` or ``quarantined``. A target already in one of those
    terminal states is reported under ``quarantined`` (already quarantined) or
    simply skipped from ``deleted`` (already deleted) so the replay is idempotent
    and never double-flips a record. A target with no record lands in
    ``not_found``.

    Returns the same summary shape as :func:`apply_memory_dispositions`:
    ``{"deleted": [...], "quarantined": [...], "not_found": [...]}``.
    """
    deleted: List[str] = []
    quarantined: List[str] = []
    not_found: List[str] = []

    for memory_id in invalidated_memory_ids:
        memory = memory_store.get_memory(memory_id)
        if memory is None:
            not_found.append(memory_id)
            continue
        if memory.status == "deleted":
            # Already torn down (idempotent replay) — nothing to do.
            continue
        if memory.status == _QUARANTINED_STATUS:
            # Already removed from the active set by quarantine; record it as
            # such rather than flipping a quarantined record to deleted.
            quarantined.append(memory_id)
            continue
        memory_store.delete_memory(memory_id)
        deleted.append(memory_id)

    return {
        "deleted": sorted(deleted),
        "quarantined": sorted(quarantined),
        "not_found": sorted(set(not_found)),
    }


def _pair_target_observations(
    target_tool_steps: Sequence[JSONRecord],
    target_observation_steps: Sequence[JSONRecord],
) -> Dict[str, Optional[str]]:
    """Deterministically pair each target action with its old observation.

    Pairing is by trace adjacency only (the action's ``next_step_ids`` or the
    observation's ``prev_step_ids``); it never consults the LLM. Each old
    observation is claimed by at most one action.
    """
    observation_by_id = {
        str(step["step_id"]): step
        for step in target_observation_steps
        if isinstance(step.get("step_id"), str)
    }
    paired: Dict[str, Optional[str]] = {}
    claimed: set[str] = set()
    for action in target_tool_steps:
        action_id = str(action.get("step_id"))
        observation_id: Optional[str] = None
        for candidate in action.get("next_step_ids") or []:
            if candidate in observation_by_id and candidate not in claimed:
                observation_id = candidate
                break
        if observation_id is None:
            for candidate_id, candidate in observation_by_id.items():
                if candidate_id in claimed:
                    continue
                if action_id in (candidate.get("prev_step_ids") or []):
                    observation_id = candidate_id
                    break
        if observation_id is not None:
            claimed.add(observation_id)
        paired[action_id] = observation_id
    return paired


def replay_targeted_tools(
    runtime: "ReplayRuntime",
    plan: PlanReplayResult,
    target_tool_steps: Sequence[JSONRecord],
    target_observation_steps: Sequence[JSONRecord],
    next_new_step_num: int,
) -> ToolReplayResult:
    """Deterministically execute a plan's per-target tool decisions (no LLM).

    Each decision names exactly one superseded ``tool_action`` target and
    emits a fresh action/observation pair. The plan's
    decisions must EXACTLY cover the supplied target action IDs — no unbound
    call may be introduced and no target may be left undecided.

    Selected calls are always executed; prior call keys do not silently remove
    mandatory targets. New action/observation step IDs are allocated contiguously
    from ``next_new_step_num``; the returned ``step_id_remap`` maps each
    replaced target action/observation to its replacement and
    ``removed_step_ids`` remains empty for mandatory replay.
    """
    decisions: List[TargetedToolDecision] = list(plan.tool_decisions)
    target_action_ids = [
        str(step["step_id"]) for step in target_tool_steps
        if isinstance(step.get("step_id"), str)
    ]
    if len(target_action_ids) != len(set(target_action_ids)):
        raise ValueError("targeted tool replay target action IDs must be unique")
    decision_ids = [decision.target_action_step_id for decision in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError(
            "targeted tool replay received duplicate target action IDs"
        )
    if set(decision_ids) != set(target_action_ids):
        missing = sorted(set(target_action_ids) - set(decision_ids))
        unexpected = sorted(set(decision_ids) - set(target_action_ids))
        raise ValueError(
            "targeted tool replay decisions must exactly cover the target "
            f"action IDs: missing={missing}, unexpected={unexpected}"
        )

    decision_by_target = {
        decision.target_action_step_id: decision for decision in decisions
    }
    paired_observation = _pair_target_observations(
        target_tool_steps, target_observation_steps
    )

    new_steps: List[JSONRecord] = []
    observations: List[JSONValue] = []
    step_id_remap: Dict[str, str] = {}
    removed_step_ids: List[str] = []
    counter = next_new_step_num
    previous_observation_id: Optional[str] = None

    for action_id in target_action_ids:
        decision = decision_by_target[action_id]
        observation_id = paired_observation.get(action_id)

        tool_name = decision.tool_name
        args = decision.args if isinstance(decision.args, dict) else {}

        call = {"tool_name": tool_name, "args": dict(args)}
        new_action_id = f"s_n{counter:03d}"
        new_observation_id = f"s_n{counter + 1:03d}"
        counter += 2
        observation = runtime.execute_tool(tool_name, args)
        observations.append(observation)
        action_step: JSONRecord = {
            "step_id": new_action_id,
            "task_id": runtime.task_id,
            "step_type": "tool_action",
            "content": json.dumps(call, ensure_ascii=False),
            "next_step_ids": [new_observation_id],
            "tool_name": tool_name,
            "tool_args": dict(args),
            "used_ids": list(plan.used_ids),
        }
        if previous_observation_id is not None:
            action_step["prev_step_ids"] = [previous_observation_id]
            new_steps[-1]["next_step_ids"] = [new_action_id]
        new_steps.extend([
            action_step,
            {
                "step_id": new_observation_id,
                "task_id": runtime.task_id,
                "step_type": "tool_observation",
                "content": json.dumps(observation, ensure_ascii=False),
                "prev_step_ids": [new_action_id],
                "used_ids": [new_action_id],
            },
        ])
        previous_observation_id = new_observation_id
        step_id_remap[action_id] = new_action_id
        if observation_id is not None:
            step_id_remap[observation_id] = new_observation_id

    return ToolReplayResult(
        new_steps=new_steps,
        observations=observations,
        step_id_remap=step_id_remap,
        removed_step_ids=removed_step_ids,
        next_new_step_num=counter,
    )


def execute_synthetic_tool_calls(
    runtime: "ReplayRuntime",
    plan: SyntheticPlanResult,
    plan_step_id: str,
    next_new_step_num: int,
    *,
    read_only_only: bool = True,
) -> ToolReplayResult:
    """Execute unbound omission-repair calls deterministically, without an LLM."""
    new_steps: List[JSONRecord] = []
    observations: List[JSONValue] = []
    counter = next_new_step_num
    predecessor_id = plan_step_id
    for call in plan.tool_calls:
        if call.tool_name not in runtime.tools:
            raise ValueError(
                f"synthetic missing-action plan names unavailable tool {call.tool_name!r}"
            )
        if read_only_only and not runtime.is_read_only_tool(call.tool_name):
            raise ValueError(
                "synthetic missing-action tool is not read-only while "
                "synthetic read-only tool replay is enabled: "
                f"{call.tool_name!r}"
            )
    # Validate the complete batch before executing its first call. This keeps
    # a later unsafe call from partially executing an earlier safe one.
    for call in plan.tool_calls:
        action_id = f"s_n{counter:03d}"
        observation_id = f"s_n{counter + 1:03d}"
        counter += 2
        args = dict(call.args)
        observation = runtime.execute_tool(call.tool_name, args)
        observations.append(observation)
        if new_steps:
            new_steps[-1]["next_step_ids"] = [action_id]
        new_steps.extend([
            {
                "step_id": action_id,
                "task_id": runtime.task_id,
                "step_type": "tool_action",
                "content": json.dumps(
                    {"tool_name": call.tool_name, "args": args},
                    ensure_ascii=False,
                ),
                "prev_step_ids": [predecessor_id],
                "next_step_ids": [observation_id],
                "tool_name": call.tool_name,
                "tool_args": args,
                "used_ids": list(plan.used_ids),
            },
            {
                "step_id": observation_id,
                "task_id": runtime.task_id,
                "step_type": "tool_observation",
                "content": json.dumps(observation, ensure_ascii=False),
                "prev_step_ids": [action_id],
                "used_ids": [action_id],
            },
        ])
        predecessor_id = observation_id
    return ToolReplayResult(
        new_steps=new_steps,
        observations=observations,
        next_new_step_num=counter,
    )
