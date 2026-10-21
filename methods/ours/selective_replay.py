"""Context resolution and production/fake operations for selective replay."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

from client.model_client import LLMResult, LLMTokenUsage

from . import replay_prompts
from .graph_schema import DependencyGraph
from .replay_runtime import ReplayRuntime
from .replay_types import (
    JSONRecord,
    JSONValue,
    MemoryReconciliationDecision,
    MemoryReplayResult,
    PlanReplayResult,
    ReplayContext,
    ReplayStepResult,
    SyntheticPlanResult,
    TargetedMemoryDecision,
    ToolReplayResult,
)


ORACLE_MEMORY_FIELDS = (
    "fact_key",
    "fact_value",
    "entity_id",
    "valid_from",
    "valid_to",
    "oracle_role",
)

_REPLAY_LLM_MAX_ATTEMPTS = 3


def _append_retry_feedback(user_prompt: str, feedback: str) -> str:
    if not feedback:
        return user_prompt
    return (
        f"{user_prompt}\n\n"
        "RETRY FEEDBACK FROM THE STRICT REPLAY VALIDATOR:\n"
        f"{feedback}\n"
        "Return a fresh JSON object that satisfies the replay-only contract. "
        "Do not include explanations or markdown."
    )


def _target_ids(steps: Sequence[JSONRecord]) -> List[str]:
    return [
        str(step.get("step_id"))
        for step in steps
        if isinstance(step.get("step_id"), str) and step.get("step_id")
    ]


def normalize_observation_value(value: JSONValue) -> JSONValue:
    """Decode JSON object/array strings while preserving all other values."""
    if not isinstance(value, str):
        return value

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value

    if isinstance(parsed, (dict, list)):
        return parsed
    return value


def _content_for(
    graph: DependencyGraph,
    step_id: str,
    current_content: Mapping[str, JSONValue],
    step_id_remap: Optional[Mapping[str, str]] = None,
) -> JSONValue:
    remapped = (step_id_remap or {}).get(step_id)
    if remapped is not None and remapped in current_content:
        return current_content[remapped]
    if step_id in current_content:
        return current_content[step_id]
    node = graph.node(step_id)
    return None if node is None else node.raw.get("content")


def _nearest_dependency(
    graph: DependencyGraph, step_id: str, semantic_type: str
) -> Optional[str]:
    from .rollback_executor import _context_predecessor_ids

    visited = {step_id}
    frontier = [step_id]
    while frontier:
        following: List[str] = []
        for current in frontier:
            for predecessor in _context_predecessor_ids(graph, current):
                if predecessor in visited:
                    continue
                visited.add(predecessor)
                node = graph.node(predecessor)
                if node is not None and node.semantic_type == semantic_type:
                    return predecessor
                following.append(predecessor)
        frontier = sorted(following)
    return None


def build_replay_context(
    runtime: ReplayRuntime,
    graph: DependencyGraph,
    step_id: str,
    session: Sequence[str],
    current_content: Mapping[str, JSONValue],
    step_id_remap: Mapping[str, str],
) -> ReplayContext:
    """Build the complete current-turn view using chain and cite dependencies."""
    node = graph.node(step_id)
    turn = node.turn if node is not None and node.turn is not None else 1
    claim_id = _nearest_dependency(graph, step_id, "claim")
    answer_id = _nearest_dependency(graph, step_id, "final_answer")

    # Observations are dependency-scoped.  A final answer normally cites them;
    # a memory step reaches them through its final-answer predecessor.
    visited = {step_id}
    frontier = [step_id]
    observation_ids: List[str] = []
    from .rollback_executor import _context_predecessor_ids
    while frontier:
        following: List[str] = []
        for current in frontier:
            for predecessor in _context_predecessor_ids(graph, current):
                if predecessor in visited:
                    continue
                visited.add(predecessor)
                predecessor_node = graph.node(predecessor)
                if predecessor_node is not None and predecessor_node.semantic_type == "tool_observation":
                    observation_ids.append(predecessor)
                else:
                    following.append(predecessor)
        frontier = sorted(following)
    observation_ids = sorted(dict.fromkeys(observation_ids))

    def text_for(identifier: Optional[str]) -> Optional[str]:
        if identifier is None:
            return None
        value = _content_for(graph, identifier, current_content, step_id_remap)
        return None if value is None else str(value)

    resolved_claim_id = (
        step_id_remap.get(claim_id, claim_id) if claim_id is not None else None
    )
    resolved_observation_ids = [
        step_id_remap.get(identifier, identifier)
        for identifier in observation_ids
    ]
    resolved_answer_id = (
        step_id_remap.get(answer_id, answer_id) if answer_id is not None else None
    )

    return ReplayContext(
        step_id=step_id,
        trigger_step_id=step_id,
        target_step_ids=[step_id],
        turn=turn,
        user_input_id=f"user_input_{turn}",
        user_prompt=session[turn - 1] if 0 < turn <= len(session) else "",
        active_memories=runtime.active_memories(),
        history=[{"user_input_id": item.user_input_id, "text": item.text} for item in runtime.history],
        claim_text=text_for(claim_id),
        claim_step_id=resolved_claim_id,
        observations=[
            normalize_observation_value(
                _content_for(graph, item, current_content, step_id_remap)
            )
            for item in observation_ids
        ],
        observation_step_ids=resolved_observation_ids,
        final_answer_text=text_for(answer_id),
        final_answer_step_id=resolved_answer_id,
    )


class ReplayBackend:
    """Production selective-replay backend using the repository LLM gateway."""

    _is_deterministic_fixture = False

    def __init__(self) -> None:
        self.contexts: List[ReplayContext] = []
        self.reconciled_turns: List[int] = []
        self.reset_llm_replay_cost()

    def reset_llm_replay_cost(self) -> None:
        """Reset model-call accounting for one replay_chain invocation."""
        self.llm_replay_call_count = 0
        self.llm_replay_calls_by_operation: Dict[str, int] = {}
        self.llm_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "usage_available_call_count": 0,
            "usage_unavailable_call_count": 0,
            "by_operation": {},
        }

    def _record_llm_replay_call(
        self, operation: str, usage: Optional[LLMTokenUsage] = None,
    ) -> None:
        self.llm_replay_call_count += 1
        self.llm_replay_calls_by_operation[operation] = (
            self.llm_replay_calls_by_operation.get(operation, 0) + 1
        )
        if usage is None:
            self.llm_token_usage["usage_unavailable_call_count"] += 1
            return
        self.llm_token_usage["usage_available_call_count"] += 1
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            self.llm_token_usage[field] += getattr(usage, field)
        by_operation = self.llm_token_usage["by_operation"]
        operation_usage = by_operation.setdefault(operation, {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
        })
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            operation_usage[field] += getattr(usage, field)
        operation_usage["call_count"] += 1

    def _call_llm(
        self,
        runtime: ReplayRuntime,
        system_prompt: str,
        user_prompt: str,
        operation: str = "replay_claim",
    ) -> str:
        # Import the module at call time so tests and embedders can reliably
        # patch the sole production model boundary.
        from client import model_client

        model = runtime.config.get("model")
        try:
            raw_response = model_client.run_llm(
                system_prompt,
                user_prompt,
                model=model if isinstance(model, str) else None,
                response_format={"type": "json_object"},
                return_usage=True,
            )
        except Exception as exc:
            raise RuntimeError(f"{operation} model call failed: {exc}") from exc

        if isinstance(raw_response, model_client.LLMResult):
            self._record_llm_replay_call(operation, raw_response.usage)
            return raw_response.text
        if isinstance(raw_response, str):
            self._record_llm_replay_call(operation)
            return raw_response
        raise TypeError(f"{operation} model call must return a string or LLMResult")

    def _step(self, kind: str, context: ReplayContext, old_step: JSONRecord) -> ReplayStepResult:
        self.contexts.append(context)
        content = self.content_by_step.get(context.step_id, f"fake {kind} for {context.step_id}")
        return ReplayStepResult(content=content, used_ids=self.used_ids_by_step.get(context.step_id, [context.user_input_id]))

    def replay_claim(self, runtime: ReplayRuntime, context: ReplayContext, old_step: JSONRecord) -> ReplayStepResult:
        if not self._is_deterministic_fixture:
            self.contexts.append(context)
            domain_prompt = str(
                runtime.config.get(
                    "domain_prompt",
                    "You are a helpful assistant.",
                )
            )
            system_prompt, user_prompt = (
                replay_prompts.build_claim_replay_prompt(
                    context,
                    domain_prompt,
                )
            )
            raw_response = self._call_llm(
                runtime,
                system_prompt,
                user_prompt,
                "replay_claim",
            )
            return replay_prompts.parse_claim_replay_response(
                raw_response,
                context,
            )
        return self._step("claim", context, old_step)

    def replay_plan(
        self, runtime: ReplayRuntime, context: ReplayContext,
        old_step: JSONRecord, target_tool_steps: Sequence[JSONRecord],
        target_observation_steps: Sequence[JSONRecord] = (),
    ) -> PlanReplayResult:
        if self._is_deterministic_fixture:
            return self._deterministic_plan(context, target_tool_steps)

        self.contexts.append(context)
        domain_prompt = str(
            runtime.config.get(
                "domain_prompt",
                "You are a helpful assistant.",
            )
        )
        tool_guidance = str(runtime.config.get("tool_guidance", ""))
        system_prompt, user_prompt = replay_prompts.build_plan_replay_prompt(
            context,
            domain_prompt,
            tool_guidance,
            runtime.tools,
            target_tool_steps,
            target_observation_steps,
        )
        feedback = ""
        target_ids = _target_ids(target_tool_steps)
        for attempt in range(_REPLAY_LLM_MAX_ATTEMPTS):
            prompt = _append_retry_feedback(user_prompt, feedback)
            raw_response = self._call_llm(
                runtime,
                system_prompt,
                prompt,
                "replay_plan",
            )
            try:
                return replay_prompts.parse_plan_replay_response(
                    raw_response,
                    context,
                    list(runtime.tools),
                    target_tool_steps,
                )
            except ValueError as exc:
                if attempt + 1 >= _REPLAY_LLM_MAX_ATTEMPTS:
                    raise
                feedback = (
                    f"Previous replay_plan output failed validation: {exc}. "
                    f"Allowed target_action_step_id values are exactly "
                    f"{target_ids!r}. Output exactly {len(target_ids)} targets, "
                    "one per allowed target, with no extra steps or unknown IDs."
                )
        raise RuntimeError("unreachable replay_plan retry state")

    def replan_missing_actions(
        self, runtime: ReplayRuntime, context: ReplayContext,
        feedback: str = "",
    ) -> SyntheticPlanResult:
        """Freely plan the narrow final-turn omission case over repaired state."""
        self.contexts.append(context)
        domain_prompt = str(runtime.config.get("domain_prompt", "You are a helpful assistant."))
        tool_guidance = str(runtime.config.get("tool_guidance", ""))
        system_prompt, user_prompt = replay_prompts.build_missing_action_replan_prompt(
            context, domain_prompt, tool_guidance, runtime.tools
        )
        retry_feedback = feedback
        for attempt in range(_REPLAY_LLM_MAX_ATTEMPTS):
            prompt = _append_retry_feedback(user_prompt, retry_feedback)
            raw_response = self._call_llm(
                runtime, system_prompt, prompt, "replan_missing_actions"
            )
            try:
                return replay_prompts.parse_missing_action_replan_response(
                    raw_response, context, list(runtime.tools)
                )
            except ValueError as exc:
                if attempt + 1 >= _REPLAY_LLM_MAX_ATTEMPTS:
                    raise
                retry_feedback = (
                    f"Previous missing-action replan output failed validation: "
                    f"{exc}. Use only available tool names {list(runtime.tools)!r}. "
                    "If no safe tool call is needed, return an empty steps list."
                )
        raise RuntimeError("unreachable replan_missing_actions retry state")

    def _deterministic_plan(
        self,
        context: ReplayContext,
        target_tool_steps: Sequence[JSONRecord],
    ) -> PlanReplayResult:
        """Map the fixture's call list onto targets under the target contract.

        The fixture's ``content_by_step[plan]`` is a JSON list of calls. Each
        call is bound POSITIONALLY to a superseded tool_action target as a
        mandatory replacement. The number of calls must exactly match the
        number of targets.
        """
        self.contexts.append(context)
        raw = self.content_by_step.get(context.step_id, "[]")
        try:
            calls = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            calls = []
        if not isinstance(calls, list):
            calls = []
        normalized_calls: List[JSONRecord] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            args = call.get("args", {})
            normalized_calls.append({
                "tool_name": str(call.get("tool_name", "")),
                "args": dict(args) if isinstance(args, dict) else {},
            })

        target_ids = [
            str(step.get("step_id")) for step in target_tool_steps
            if isinstance(step.get("step_id"), str)
        ]
        if len(normalized_calls) != len(target_ids):
            raise ValueError(
                "deterministic plan fixture call count must exactly equal tool "
                f"target count ({len(normalized_calls)} != {len(target_ids)})"
            )

        decisions: List["TargetedToolDecision"] = []
        from .replay_types import TargetedToolDecision
        for index, target_id in enumerate(target_ids):
            call = normalized_calls[index]
            decisions.append(TargetedToolDecision(
                target_action_step_id=target_id,
                tool_name=str(call["tool_name"]),
                args=dict(call["args"]),
            ))
        content = json.dumps(
            [
                {
                    "type": "tool",
                    "tool_name": decision.tool_name,
                    "args": decision.args,
                }
                for decision in decisions
            ],
            ensure_ascii=False,
        )
        used_ids = self.used_ids_by_step.get(
            context.step_id, [context.user_input_id]
        )
        return PlanReplayResult(
            content=content,
            used_ids=list(used_ids),
            tool_decisions=decisions,
        )

    def replay_final_answer(self, runtime: ReplayRuntime, context: ReplayContext, old_step: JSONRecord) -> ReplayStepResult:
        if self._is_deterministic_fixture:
            return self._step("final answer", context, old_step)

        self.contexts.append(context)
        domain_prompt = str(
            runtime.config.get(
                "domain_prompt",
                "You are a helpful assistant.",
            )
        )
        system_prompt, user_prompt = (
            replay_prompts.build_final_answer_replay_prompt(
                context,
                domain_prompt,
            )
        )
        raw_response = self._call_llm(
            runtime,
            system_prompt,
            user_prompt,
            "replay_final_answer",
        )
        return replay_prompts.parse_final_answer_replay_response(
            raw_response,
            context,
        )

    def reconcile_memories(self, runtime: ReplayRuntime, context: ReplayContext, old_steps: Sequence[JSONRecord], next_new_step_num: int, next_new_memory_num: int) -> MemoryReplayResult:
        self.contexts.append(context)
        self.reconciled_turns.append(context.turn)
        if not old_steps:
            return MemoryReplayResult()
        if not self._is_deterministic_fixture:
            domain_prompt = str(
                runtime.config.get(
                    "domain_prompt",
                    "You are a helpful assistant.",
                )
            )
            memory_guidance = str(runtime.config.get("memory_guidance", ""))
            system_prompt, user_prompt = (
                replay_prompts.build_memory_reconciliation_prompt(
                    context,
                    domain_prompt,
                    memory_guidance,
                    old_steps,
                )
            )
            feedback = ""
            target_ids = _target_ids(old_steps)
            observation_ids = list(context.observation_step_ids)
            for attempt in range(_REPLAY_LLM_MAX_ATTEMPTS):
                prompt = _append_retry_feedback(user_prompt, feedback)
                raw_response = self._call_llm(
                    runtime,
                    system_prompt,
                    prompt,
                    "reconcile_memories",
                )
                try:
                    decision = replay_prompts.parse_memory_reconciliation_response(
                        raw_response,
                        context,
                        old_steps,
                    )
                    break
                except ValueError as exc:
                    if attempt + 1 >= _REPLAY_LLM_MAX_ATTEMPTS:
                        raise
                    feedback = (
                        "Previous memory reconciliation output failed validation: "
                        f"{exc}. Required target_step_id values are exactly "
                        f"{target_ids!r}; output exactly one target for each. "
                        f"Current observation evidence IDs are {observation_ids!r}. "
                        "Use source='tool_observation' only when the target's "
                        "used list includes one of those current observation IDs; "
                        "otherwise use source='agent_summary' for repaired "
                        "summaries derived from claim/history/memories."
                    )
            else:
                raise RuntimeError("unreachable reconcile_memories retry state")
            return self._materialize_memory_decision(
                runtime,
                context,
                old_steps,
                decision,
                next_new_step_num,
                next_new_memory_num,
            )

        target_key = tuple(context.target_step_ids)
        configured = self.memory_results_by_targets.get(target_key)
        if configured is None:
            raise NotImplementedError(
                "FakeReplayBackend memory reconciliation requires an explicit "
                f"MemoryReplayResult fixture for targets {list(target_key)!r}"
            )

        result = deepcopy(configured)
        allocated_step_ids = []
        for offset, step in enumerate(result.new_steps):
            step_id = step.get("step_id")
            if not isinstance(step_id, str) or not step_id:
                step_id = f"s_n{next_new_step_num + offset:03d}"
                step["step_id"] = step_id
            allocated_step_ids.append(step_id)
        for offset, old_step in enumerate(old_steps):
            old_id = old_step.get("step_id")
            if (
                isinstance(old_id, str)
                and not result.step_id_remap.get(old_id)
                and offset < len(allocated_step_ids)
            ):
                result.step_id_remap[old_id] = allocated_step_ids[offset]
        pending_old_memory_ids = [
            old_id for old_id, new_id in result.memory_id_remap.items()
            if not new_id
        ]
        allocated_memory_ids = [
            f"m_n{next_new_memory_num + offset:03d}"
            for offset in range(len(pending_old_memory_ids))
        ]
        for old_id, new_id in zip(pending_old_memory_ids, allocated_memory_ids):
            result.memory_id_remap[old_id] = new_id
        if allocated_memory_ids:
            result.new_memory_ids = allocated_memory_ids
            for offset, step in enumerate(result.new_steps):
                old_generated = (
                    old_steps[offset].get("generated_memory_ids", [])
                    if offset < len(old_steps) else []
                )
                generated = [
                    result.memory_id_remap[old_id]
                    for old_id in old_generated
                    if old_id in result.memory_id_remap
                ]
                if generated:
                    step["generated_memory_ids"] = generated
        step_by_generated_memory = {
            memory_id: step
            for step in result.new_steps
            for memory_id in (step.get("generated_memory_ids") or [])
            if isinstance(memory_id, str)
        }
        records = []
        for memory_id in result.new_memory_ids:
            if runtime.memory_store.get_memory(memory_id) is not None:
                continue
            step = step_by_generated_memory.get(memory_id)
            if step is None:
                continue
            content = step.get("content")
            records.append({
                "memory_id": memory_id,
                "task_id": runtime.task_id or "",
                "content": "" if content is None else str(content),
                "source": "agent_summary",
                "last_modified_by": step.get("step_id", "fake_replay"),
                "status": "active",
            })
        if records:
            runtime.memory_store.load_memories(records)
        return result

    @staticmethod
    def _materialize_memory_decision(
        runtime: ReplayRuntime,
        context: ReplayContext,
        old_steps: Sequence[JSONRecord],
        decision: MemoryReconciliationDecision,
        next_new_step_num: int,
        next_new_memory_num: int,
    ) -> MemoryReplayResult:
        """Allocate trace/memory IDs per target, then apply the store effect.

        Each target is exactly one old memory-mutation step. Its replacement
        allocates one contiguous new step ID (and a new memory ID unless the
        target is a ``memory_delete``), fixes generated/invalidated IDs from the
        target's ORIGINAL metadata, and applies the corresponding MemoryStore
        effect. The target's original semantic type is never changed here.
        """
        active_by_id = {
            memory["memory_id"]: memory
            for memory in context.active_memories
            if isinstance(memory.get("memory_id"), str)
        }
        decision_by_target = {
            target.target_step_id: target for target in decision.targets
        }
        target_ids = [str(step.get("step_id")) for step in old_steps]
        decision_ids = [target.target_step_id for target in decision.targets]
        if (
            len(target_ids) != len(set(target_ids))
            or len(decision_ids) != len(set(decision_ids))
            or set(decision_ids) != set(target_ids)
        ):
            raise ValueError(
                "mandatory memory replacements must exactly cover selected targets"
            )

        # The replay contract still requires one replacement per selected
        # memory mutation. If the model proposes content already preserved in
        # the active store, materialize it instead of aborting the whole run;
        # downstream store application and cleanup keep the final state valid.

        new_steps: List[JSONRecord] = []
        new_memory_ids: List[str] = []
        memory_id_remap: Dict[str, str] = {}
        step_id_remap: Dict[str, str] = {}
        removed_step_ids: List[str] = []
        # (new_step_id, new_memory_id, target, old_step, resolved_invalidated)
        # for the apply pass.
        pending: List[
            tuple[str, Optional[str], "TargetedMemoryDecision", JSONRecord, List[str]]
        ] = []
        step_counter = next_new_step_num
        memory_counter = next_new_memory_num

        for old_step in old_steps:
            old_id = str(old_step.get("step_id"))
            target = decision_by_target[old_id]
            old_type = old_step.get("step_type")
            old_generated = [
                str(value) for value in (old_step.get("generated_memory_ids") or [])
                if isinstance(value, str)
            ]
            old_invalidated = [
                str(value) for value in (old_step.get("invalidated_memory_ids") or [])
                if isinstance(value, str)
            ]
            resolved_invalidated: List[str] = []
            for memory_id in old_invalidated:
                resolved = memory_id
                seen = set()
                while resolved not in seen:
                    seen.add(resolved)
                    mapped = memory_id_remap.get(resolved)
                    if not isinstance(mapped, str) or not mapped:
                        break
                    resolved = mapped
                resolved_invalidated.append(resolved)
            old_invalidated = list(dict.fromkeys(resolved_invalidated))

            new_step_id = f"s_n{step_counter:03d}"
            step_counter += 1
            produces_memory = old_type != "memory_delete"
            generated: List[str] = []
            new_memory_id: Optional[str] = None
            if produces_memory:
                new_memory_id = f"m_n{memory_counter:03d}"
                memory_counter += 1
                generated = [new_memory_id]
                new_memory_ids.append(new_memory_id)

            # Memory remaps come from the target's ORIGINAL generated/invalidated
            # IDs, never from LLM output: every memory the old step produced or
            # retired now resolves to this replacement's memory.
            if new_memory_id is not None:
                for memory_id in old_generated + old_invalidated:
                    memory_id_remap.setdefault(memory_id, new_memory_id)

            if old_type == "memory_delete":
                anchor = old_invalidated[0] if old_invalidated else None
                content = (
                    f"Deleted memory: {active_by_id[anchor]['content']}"
                    if anchor is not None and anchor in active_by_id
                    else str(target.content or "")
                )
            else:
                content = str(target.content or "")

            remapped_used_ids: List[str] = []
            for identifier in target.used_ids:
                resolved = identifier
                seen = set()
                while resolved not in seen:
                    seen.add(resolved)
                    mapped = memory_id_remap.get(resolved)
                    if not isinstance(mapped, str) or not mapped:
                        break
                    resolved = mapped
                remapped_used_ids.append(resolved)
            if old_type in {"memory_delete", "memory_update", "memory_consolidate"}:
                invalidated_set = set(old_invalidated)
                remapped_used_ids = [
                    identifier for identifier in remapped_used_ids
                    if identifier not in invalidated_set
                ]

            step: JSONRecord = {
                "step_id": new_step_id,
                "task_id": runtime.task_id or "",
                "step_type": old_type,
                "content": content,
                "generated_memory_ids": generated,
                "invalidated_memory_ids": list(old_invalidated),
                "used_ids": remapped_used_ids,
            }
            if produces_memory:
                step["source"] = target.source
                step["trust_score"] = target.trust_score
            if context.final_answer_step_id:
                step["prev_step_ids"] = [context.final_answer_step_id]
            new_steps.append(step)
            step_id_remap[old_id] = new_step_id
            pending.append((new_step_id, new_memory_id, target, old_step, list(old_invalidated)))

        # Validate the live store snapshot once more before mutating: every
        # memory a replace supersedes or deletes must still be active.
        affected_ids = {
            memory_id
            for _step_id, _memory_id, _target, old_step, old_invalidated in pending
            for memory_id in (
                old_invalidated if old_step.get("step_type") != "memory_write" else []
            )
            if isinstance(memory_id, str)
        }
        pending_new_ids = {
            memory_id for _step_id, memory_id, _target, _old_step, _old_invalidated in pending
            if memory_id is not None
        }
        if any(
            (
                memory_id not in pending_new_ids
                and (
                    runtime.memory_store.get_memory(memory_id) is None
                    or runtime.memory_store.get_memory(memory_id).status != "active"  # type: ignore[union-attr]
                )
            )
            for memory_id in affected_ids
        ):
            raise ValueError("memory store changed during reconciliation")

        for new_step_id, new_memory_id, target, old_step, old_invalidated in pending:
            old_type = old_step.get("step_type")
            if old_type == "memory_delete":
                for memory_id in old_invalidated:
                    old_memory = runtime.memory_store.get_memory(memory_id)
                    if old_memory is not None:
                        old_memory.status = "deleted"
                        old_memory.last_modified_by = new_step_id
                continue

            assert new_memory_id is not None
            old_generated = [
                str(value) for value in (old_step.get("generated_memory_ids") or [])
                if isinstance(value, str)
            ]
            if old_type == "memory_write":
                source = str(target.source or "agent_summary")
                supersedes = old_generated[0] if old_generated else None
                for old_id in old_generated:
                    old_memory = runtime.memory_store.get_memory(old_id)
                    if old_memory is not None:
                        old_memory.status = "superseded"
                        old_memory.last_modified_by = new_step_id
                derived_from: List[str] = []
                trust_score = target.trust_score
            elif old_type == "memory_update":
                old_id = old_invalidated[0] if old_invalidated else None
                old_memory = (
                    runtime.memory_store.get_memory(old_id)
                    if old_id is not None else None
                )
                if old_memory is not None:
                    old_memory.status = "superseded"
                    source = str(target.source or old_memory.source)
                    trust_score = (
                        target.trust_score
                        if target.trust_score is not None
                        else old_memory.trust_score
                    )
                else:
                    source = str(target.source or "agent_summary")
                    trust_score = target.trust_score
                supersedes = old_id
                derived_from = []
            else:  # memory_consolidate
                derived_from = list(old_invalidated)
                for old_id in derived_from:
                    old_memory = runtime.memory_store.get_memory(old_id)
                    if old_memory is not None:
                        old_memory.status = "superseded"
                source = str(target.source or "agent_summary")
                supersedes = None
                trust_score = target.trust_score
            record = {
                "memory_id": new_memory_id,
                "task_id": runtime.task_id or "",
                "content": target.content,
                "source": source,
                "last_modified_by": new_step_id,
                "derived_from": derived_from,
                "supersedes": supersedes,
                "trust_score": trust_score,
                "status": "active",
            }
            source_memories = []
            for old_id in list(old_invalidated) + list(old_generated):
                old_memory = runtime.memory_store.get_memory(old_id)
                if old_memory is not None:
                    source_memories.append(old_memory.to_dict())
            if source_memories:
                for field in ORACLE_MEMORY_FIELDS:
                    values = [
                        memory.get(field)
                        for memory in source_memories
                        if memory.get(field) is not None
                    ]
                    unique_values = {
                        json.dumps(value, sort_keys=True)
                        for value in values
                    }
                    if values and len(unique_values) == 1:
                        record[field] = values[0]
            runtime.memory_store.load_memories([record])

        return MemoryReplayResult(
            new_steps=new_steps,
            new_memory_ids=new_memory_ids,
            memory_id_remap=memory_id_remap,
            step_id_remap=step_id_remap,
            removed_step_ids=removed_step_ids,
        )


class FakeReplayBackend(ReplayBackend):
    """Explicit test backend using an injected callable or fixture data only."""

    def __init__(
        self,
        *,
        content_by_step: Optional[Mapping[str, JSONValue]] = None,
        used_ids_by_step: Optional[Mapping[str, Sequence[str]]] = None,
        memory_results_by_targets: Optional[
            Mapping[tuple[str, ...], MemoryReplayResult]
        ] = None,
        llm: Optional[Callable[..., Union[str, LLMResult]]] = None,
        missing_action_replan: Optional[
            Callable[[ReplayRuntime, ReplayContext], SyntheticPlanResult]
        ] = None,
    ) -> None:
        super().__init__()
        self.content_by_step = dict(content_by_step or {})
        self.used_ids_by_step = {
            key: list(value)
            for key, value in (used_ids_by_step or {}).items()
        }
        self.memory_results_by_targets = {
            tuple(key): deepcopy(value)
            for key, value in (memory_results_by_targets or {}).items()
        }
        self.llm = llm
        self.missing_action_replan = missing_action_replan

    @property
    def _is_deterministic_fixture(self) -> bool:
        return self.llm is None

    def _call_llm(
        self,
        runtime: ReplayRuntime,
        system_prompt: str,
        user_prompt: str,
        operation: str = "replay_claim",
    ) -> str:
        if self.llm is None:
            raise RuntimeError(
                "FakeReplayBackend requires an injected LLM callable"
            )
        raw_response = self.llm(
            system_prompt,
            user_prompt,
            model=runtime.model,
            response_format={"type": "json_object"},
        )
        if isinstance(raw_response, LLMResult):
            self._record_llm_replay_call(operation, raw_response.usage)
            return raw_response.text
        if isinstance(raw_response, str):
            self._record_llm_replay_call(operation)
            return raw_response
        raise TypeError("injected LLM callable must return a string or LLMResult")

    def replan_missing_actions(
        self, runtime: ReplayRuntime, context: ReplayContext,
        feedback: str = "",
    ) -> SyntheticPlanResult:
        if self.missing_action_replan is not None:
            self.contexts.append(context)
            result = self.missing_action_replan(runtime, context)
            if not isinstance(result, SyntheticPlanResult):
                raise TypeError("injected missing-action planner must return SyntheticPlanResult")
            return deepcopy(result)
        if self.llm is None:
            self.contexts.append(context)
            return SyntheticPlanResult(content="[]")
        return super().replan_missing_actions(runtime, context, feedback=feedback)
