"""Typed values exchanged by the selective replay runtime and backend."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[JSONScalar, List["JSONValue"], Dict[str, "JSONValue"]]
JSONRecord = Dict[str, JSONValue]


@dataclass
class ReplayContext:
    step_id: str
    turn: int
    user_input_id: str
    user_prompt: str
    trigger_step_id: Optional[str] = None
    target_step_ids: List[str] = field(default_factory=list)
    active_memories: List[JSONRecord] = field(default_factory=list)
    history: List[JSONRecord] = field(default_factory=list)
    claim_text: Optional[str] = None
    claim_step_id: Optional[str] = None
    observations: List[JSONValue] = field(default_factory=list)
    observation_step_ids: List[str] = field(default_factory=list)
    final_answer_text: Optional[str] = None
    final_answer_step_id: Optional[str] = None
    protected_memory_outputs: List[JSONRecord] = field(default_factory=list)


ReplayContextObserver = Callable[[str, ReplayContext], None]


def replay_context_to_dict(context: ReplayContext) -> Dict[str, JSONValue]:
    """Return a complete, detached JSON-compatible replay context."""
    return {
        "step_id": context.step_id,
        "trigger_step_id": context.trigger_step_id,
        "target_step_ids": list(context.target_step_ids),
        "turn": context.turn,
        "user_input_id": context.user_input_id,
        "user_prompt": context.user_prompt,
        "active_memories": deepcopy(context.active_memories),
        "history": deepcopy(context.history),
        "claim_text": context.claim_text,
        "claim_step_id": context.claim_step_id,
        "observations": deepcopy(context.observations),
        "observation_step_ids": list(context.observation_step_ids),
        "final_answer_text": context.final_answer_text,
        "final_answer_step_id": context.final_answer_step_id,
        "protected_memory_outputs": deepcopy(context.protected_memory_outputs),
    }


@dataclass
class ReplayStepResult:
    content: JSONValue
    used_ids: List[str] = field(default_factory=list)


@dataclass
class SyntheticToolCall:
    tool_name: str
    args: JSONRecord = field(default_factory=dict)


@dataclass
class SyntheticPlanResult:
    content: str
    used_ids: List[str] = field(default_factory=list)
    tool_calls: List[SyntheticToolCall] = field(default_factory=list)


@dataclass
class TargetedToolDecision:
    """The mandatory replacement call for one selected tool target."""

    target_action_step_id: str
    tool_name: str
    args: JSONRecord = field(default_factory=dict)


@dataclass
class PlanReplayResult(ReplayStepResult):
    """A plan replay result carrying its per-target tool decisions.

    ``content`` holds the effective replacement calls (never internal target
    metadata); ``tool_decisions`` binds exactly one call to every target.
    """

    tool_decisions: List[TargetedToolDecision] = field(default_factory=list)


@dataclass
class ToolReplayResult:
    new_steps: List[JSONRecord] = field(default_factory=list)
    observations: List[JSONValue] = field(default_factory=list)
    step_id_remap: Dict[str, str] = field(default_factory=dict)
    removed_step_ids: List[str] = field(default_factory=list)
    next_new_step_num: int = 1


@dataclass
class TargetedMemoryDecision:
    """The mandatory replacement for one selected memory-mutation target."""

    target_step_id: str
    content: Optional[str] = None
    source: Optional[str] = None
    trust_score: Optional[float] = None
    used_ids: List[str] = field(default_factory=list)


@dataclass
class MemoryReconciliationDecision:
    """Validated target-constrained memory actions, before IDs are allocated.

    Each entry in ``targets`` is exactly one old selected memory-mutation step;
    the LLM must replace each supplied target and cannot introduce an unbound
    new/update/delete/consolidation action.
    """

    targets: List[TargetedMemoryDecision] = field(default_factory=list)


@dataclass
class MemoryReplayResult:
    """Reconciliation output using IDs allocated from the supplied counters.

    ``new_steps`` and ``new_memory_ids`` are ordered allocation sequences; the
    executor validates that both begin at their independently supplied next
    numbers and remain contiguous.
    """

    new_steps: List[JSONRecord] = field(default_factory=list)
    new_memory_ids: List[str] = field(default_factory=list)
    memory_id_remap: Dict[str, str] = field(default_factory=dict)
    step_id_remap: Dict[str, str] = field(default_factory=dict)
    removed_step_ids: List[str] = field(default_factory=list)
