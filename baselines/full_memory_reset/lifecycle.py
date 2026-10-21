"""Full-memory-reset baseline lifecycle.

This module mirrors the final-turn shape of ReactAgent without calling
ReactAgent.run() or AgentRunner.respond().
"""

from __future__ import annotations

import inspect
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent import prompts
from agent.memory_store import Memory, MemoryStore
from agent.react_agent import HistoryTurn, ReactAgent
from agent.trace_logger import TraceLogger, TraceStep
from baselines.common_tools.llm_tracker import BaselineLLMTracker


@dataclass
class FullResetLifecycleResult:
    response: str
    repaired_memories: List[Dict[str, Any]]
    new_final_turn_steps: List[Dict[str, Any]]
    tracker: BaselineLLMTracker
    reset_deleted_memory_ids: List[str]


def _parse_json(text: str) -> Any:
    if text is None:
        raise ValueError("Empty LLM response")
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]!r}")


def _tool_specs(tools: Dict[str, Any]) -> str:
    lines: List[str] = []
    for name, func in tools.items():
        if callable(func):
            try:
                sig = str(inspect.signature(func))
            except (TypeError, ValueError):
                sig = "(...)"
            doc = (inspect.getdoc(func) or "").strip().splitlines()
            summary = doc[0] if doc else ""
            lines.append(f"- {name}{sig}: {summary}")
        else:
            lines.append(f"- {name}(...)")
    return "\n".join(lines) if lines else "(no tools available)"


def _normalize_used(
    raw: Any,
    *,
    known_memory_ids: set[str],
    valid_step_ids: set[str],
    valid_history_ids: set[str],
) -> List[str]:
    allowed = known_memory_ids | valid_step_ids | valid_history_ids
    out = [uid for uid in (raw or []) if uid in allowed]
    return list(dict.fromkeys(out))


def _load_reset_memories(
    original_memories: List[Dict[str, Any]],
    *,
    delete_memory_ids: Optional[List[str]] = None,
    modified_by: str = "full_memory_reset",
) -> tuple[List[Dict[str, Any]], List[str]]:
    reset_memories = deepcopy(original_memories)
    now = datetime.now().isoformat()
    deleted_ids: List[str] = []
    delete_set = set(delete_memory_ids) if delete_memory_ids is not None else None
    for memory in reset_memories:
        if delete_set is not None and memory.get("memory_id") not in delete_set:
            continue
        if delete_set is None and memory.get("status") != "active":
            continue
        memory["status"] = "deleted"
        memory["last_modified_by"] = modified_by
        memory["last_modified_at"] = now
        deleted_ids.append(memory["memory_id"])
    return reset_memories, deleted_ids


def _log(
    trace_logger: TraceLogger,
    task_id: str,
    step_type: str,
    content: str,
    *,
    prev_step_ids: Optional[List[str]] = None,
    next_step_ids: Optional[List[str]] = None,
    invalidated_memory_ids: Optional[List[str]] = None,
    generated_memory_ids: Optional[List[str]] = None,
    used_ids: Optional[List[str]] = None,
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None,
) -> str:
    step = TraceStep(
        step_id="",
        task_id=task_id,
        step_type=step_type,
        content=content,
        prev_step_ids=list(prev_step_ids or []),
        next_step_ids=list(next_step_ids or []),
        invalidated_memory_ids=list(invalidated_memory_ids or []),
        generated_memory_ids=list(generated_memory_ids or []),
        used_ids=list(used_ids or []),
        tool_name=tool_name,
        tool_args=tool_args,
        timestamp=None,
    )
    trace_logger.log_step(step)
    return step.step_id


def _make_claim(
    *,
    tracker: BaselineLLMTracker,
    config: Dict[str, Any],
    history: List[HistoryTurn],
    prompt: str,
    memories: List[Memory],
    user_input_id: str,
) -> Dict[str, Any]:
    memory_block = "\n".join(f"[{m.memory_id}] {m.content}" for m in memories) or (
        "(no relevant memories)"
    )
    history_block = "\n".join(
        f"[{turn.user_input_id}] {turn.text}" for turn in history
    ) or "(no prior turns)"
    raw = tracker.call(
        "claim",
        prompts.claim_system(config.get("domain_prompt", "You are a helpful assistant.")),
        prompts.claim_user(prompt, memory_block, history_block, user_input_id),
        model=config.get("model"),
        response_format={"type": "json_object"},
    )
    data = _parse_json(raw)
    known = {m.memory_id for m in memories}
    history_ids = {turn.user_input_id for turn in history} | {user_input_id}
    raw_used = data.get("used", []) if isinstance(data, dict) else []
    used_memory_ids = [mid for mid in raw_used if mid in known]
    used_history_ids = list(dict.fromkeys([hid for hid in raw_used if hid in history_ids]))
    return {
        "claim": str(data.get("claim", "")).strip() if isinstance(data, dict) else "",
        "used_memory_ids": used_memory_ids,
        "used_history_ids": used_history_ids,
        "used": used_memory_ids + used_history_ids,
    }


def _plan_actions(
    *,
    tracker: BaselineLLMTracker,
    config: Dict[str, Any],
    tools: Dict[str, Any],
    history: List[HistoryTurn],
    prompt: str,
    claim: Dict[str, Any],
    memories: List[Memory],
    observations: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    claim_step_id: str,
    user_input_id: str,
) -> tuple[List[Dict[str, Any]], List[str]]:
    memory_block = "\n".join(f"[{m.memory_id}] {m.content}" for m in memories) or (
        "(no relevant memories)"
    )
    obs_block = (
        json.dumps(observations, ensure_ascii=False, default=str)
        if observations
        else "(no tool results yet this turn)"
    )
    error_block = (
        json.dumps(errors, ensure_ascii=False, default=str)
        if errors
        else "(no failed calls yet this turn)"
    )
    system = prompts.plan_system(
        config.get("domain_prompt", "You are a helpful assistant."),
        prompts.plan_guidance_block(config.get("tool_guidance", "")),
        _tool_specs(tools),
        memory_block,
        obs_block,
        error_block,
    )
    raw = tracker.call(
        "plan",
        system,
        prompts.plan_user(prompt, claim.get("claim", ""), claim_step_id, user_input_id),
        model=config.get("model"),
        response_format={"type": "json_object"},
    )
    data = _parse_json(raw)
    valid: List[Dict[str, Any]] = []
    for step in (data.get("steps", []) if isinstance(data, dict) else data) or []:
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool_name")
        if tool_name in tools:
            valid.append(
                {
                    "type": "tool",
                    "tool_name": tool_name,
                    "args": step.get("args", {}) or {},
                }
            )
    used_ids = _normalize_used(
        data.get("used", []) if isinstance(data, dict) else [],
        known_memory_ids={m.memory_id for m in memories},
        valid_step_ids={claim_step_id, user_input_id},
        valid_history_ids={turn.user_input_id for turn in history},
    )
    return valid, used_ids


def _generate_response(
    *,
    tracker: BaselineLLMTracker,
    config: Dict[str, Any],
    history: List[HistoryTurn],
    prompt: str,
    claim: Dict[str, Any],
    observations: List[Dict[str, Any]],
    memories: List[Memory],
    claim_step_id: str,
    observation_steps: List[str],
    user_input_id: str,
) -> tuple[str, List[str]]:
    memory_block = "\n".join(f"[{m.memory_id}] {m.content}" for m in memories) or (
        "(no relevant memories)"
    )
    obs_block = (
        "\n".join(
            f"[{obs_step}] {json.dumps(obs, ensure_ascii=False, default=str)}"
            for obs_step, obs in zip(observation_steps, observations)
        )
        if observations
        else "(no tool results)"
    )
    raw = tracker.call(
        "final_answer",
        prompts.response_system(config.get("domain_prompt", "You are a helpful assistant.")),
        prompts.response_user(
            prompt,
            claim.get("claim", ""),
            memory_block,
            obs_block,
            claim_step_id,
            user_input_id,
        ),
        model=config.get("model"),
        response_format={"type": "json_object"},
    )
    data = _parse_json(raw)
    if not isinstance(data, dict):
        data = {}
    used_ids = _normalize_used(
        data.get("used", []),
        known_memory_ids={m.memory_id for m in memories},
        valid_step_ids=set(observation_steps) | {claim_step_id, user_input_id},
        valid_history_ids={turn.user_input_id for turn in history},
    )
    return str(data.get("response", "")).strip(), used_ids


def _extract_memories(
    *,
    tracker: BaselineLLMTracker,
    config: Dict[str, Any],
    history: List[HistoryTurn],
    memory_store: MemoryStore,
    task_id: str,
    prompt: str,
    response: str,
    existing: List[Memory],
    relied_history: List[HistoryTurn],
    observations: List[Dict[str, Any]],
    claim: Dict[str, Any],
    claim_step_id: str,
    observation_steps: List[str],
    user_input_id: str,
) -> Dict[str, List[Any]]:
    existing_block = "\n".join(
        f"[{m.memory_id}] (source={m.source}, trust={m.trust_score}) {m.content}"
        for m in existing
    ) or "(no existing memories)"
    history_block = "\n".join(
        f"[{turn.user_input_id}] {turn.text}" for turn in relied_history
    ) or "(no prior turns relied on)"
    obs_block = (
        "\n".join(
            f"[{obs_step}] {json.dumps(obs, ensure_ascii=False, default=str)}"
            for obs_step, obs in zip(observation_steps, observations)
        )
        if observations
        else "(no tool results)"
    )
    claim_block = f"[{claim_step_id}] {claim.get('claim', '')}"
    raw = tracker.call(
        "extract_memories",
        prompts.extract_system(
            config.get("domain_prompt", "You are a helpful assistant."),
            prompts.memory_guidance_block(config.get("memory_guidance", "")),
        ),
        prompts.extract_user(
            prompt,
            response,
            history_block,
            obs_block,
            existing_block,
            claim_block,
            user_input_id,
        ),
        model=config.get("model"),
        response_format={"type": "json_object"},
    )
    data = _parse_json(raw)
    if not isinstance(data, dict):
        data = {}

    known = {m.memory_id for m in existing}
    valid_step_ids = set(observation_steps) | {claim_step_id, user_input_id}
    valid_history_ids = {turn.user_input_id for turn in relied_history}

    def used_for(item: Dict[str, Any]) -> List[str]:
        return _normalize_used(
            item.get("used", []),
            known_memory_ids=known,
            valid_step_ids=valid_step_ids,
            valid_history_ids=valid_history_ids,
        )

    created: List[tuple] = []
    updated: List[tuple] = []
    deleted: List[tuple] = []
    consolidated: List[tuple] = []
    acted: set[str] = set()

    for item in data.get("new", []) or []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        memory = Memory(
            memory_id="",
            task_id=task_id,
            content=content,
            source=str(item.get("source", "agent_summary")),
            last_modified_by="agent",
            trust_score=item.get("trust_score"),
            status="active",
        )
        memory_store.add_memory(memory)
        created.append((memory, used_for(item)))

    for item in data.get("updates", []) or []:
        if not isinstance(item, dict):
            continue
        memory_id = item.get("memory_id")
        if memory_id not in known or memory_id in acted:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        updates = {"content": content, "last_modified_by": "agent"}
        if item.get("trust_score") is not None:
            updates["trust_score"] = item.get("trust_score")
        try:
            new_id = memory_store.update_memory(memory_id, updates)
        except (KeyError, ValueError):
            continue
        acted.add(memory_id)
        memory = memory_store.get_memory(new_id)
        if memory is not None:
            updated.append((memory, used_for(item)))

    for item in data.get("deletes", []) or []:
        if not isinstance(item, dict):
            continue
        memory_id = item.get("memory_id")
        if memory_id not in known or memory_id in acted:
            continue
        try:
            memory_store.delete_memory(memory_id)
        except KeyError:
            continue
        acted.add(memory_id)
        memory = memory_store.get_memory(memory_id)
        if memory is not None:
            deleted.append((memory, used_for(item)))

    for item in data.get("consolidations", []) or []:
        if not isinstance(item, dict):
            continue
        source_ids = [
            mid for mid in (item.get("memory_ids") or []) if mid in known and mid not in acted
        ]
        source_ids = list(dict.fromkeys(source_ids))
        content = str(item.get("content", "")).strip()
        if len(source_ids) < 2 or not content:
            continue
        try:
            new_id = memory_store.consolidate_memories(
                source_ids,
                content=content,
                trust_score=item.get("trust_score"),
            )
        except (KeyError, ValueError):
            continue
        acted.update(source_ids)
        memory = memory_store.get_memory(new_id)
        if memory is not None:
            consolidated.append((memory, used_for(item)))

    return {
        "created": created,
        "updated": updated,
        "deleted": deleted,
        "consolidated": consolidated,
    }


def run_full_memory_reset_lifecycle(
    *,
    task_id: str,
    session: List[str],
    original_memories: List[Dict[str, Any]],
    old_prefix_steps: List[Dict[str, Any]],
    tools: Dict[str, Any],
    config: Dict[str, Any],
    delete_memory_ids: Optional[List[str]] = None,
    modified_by: str = "full_memory_reset",
) -> FullResetLifecycleResult:
    if not session:
        raise ValueError(f"session for task {task_id!r} is empty")

    tracker = BaselineLLMTracker()
    reset_memories, reset_deleted_ids = _load_reset_memories(
        original_memories,
        delete_memory_ids=delete_memory_ids,
        modified_by=modified_by,
    )
    memory_store = MemoryStore()
    memory_store.load_memories(reset_memories)

    trace_logger = TraceLogger()
    trace_logger.load(deepcopy(old_prefix_steps))
    trace_start = len(trace_logger.trace_steps)

    helper_agent = ReactAgent(tools, config)
    helper_agent.trace_logger = trace_logger
    helper_agent.memory_store = memory_store
    helper_agent.task_id = task_id
    helper_agent.turn_index = max(0, len(session) - 1)
    helper_agent.history = [
        HistoryTurn(user_input_id=f"user_input_{index + 1}", text=text)
        for index, text in enumerate(session[:-1])
    ]
    helper_agent.restore_executed_from_trace()

    prompt = session[-1]
    turn_index = len(session)
    user_input_id = f"user_input_{turn_index}"
    history = helper_agent.history

    used_memories = memory_store.find_relevant_memories(
        prompt,
        limit=config.get("retrieve_limit", 10),
        task_id=task_id,
    )
    read_step = _log(
        trace_logger,
        task_id,
        "memory_read",
        content=f"Retrieved {len(used_memories)} relevant memories.",
        used_ids=[memory.memory_id for memory in used_memories],
    )

    claim = _make_claim(
        tracker=tracker,
        config=config,
        history=history,
        prompt=prompt,
        memories=used_memories,
        user_input_id=user_input_id,
    )
    claim_step = _log(
        trace_logger,
        task_id,
        "claim",
        content=claim.get("claim", ""),
        prev_step_ids=[read_step],
        used_ids=claim.get("used", []),
    )

    history_by_id = {turn.user_input_id: turn for turn in history}
    relied_history_ids = [
        hid for hid in claim.get("used_history_ids", []) if hid in history_by_id
    ]
    relied_history = [history_by_id[hid] for hid in relied_history_ids]

    max_plan_iters = config.get("max_plan_iters", 3)
    tool_calls: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    observation_steps: List[str] = []
    executed: set[str] = set()
    prev_step_for_plan = claim_step

    for _ in range(max(1, max_plan_iters)):
        plans, plan_used = _plan_actions(
            tracker=tracker,
            config=config,
            tools=tools,
            history=history,
            prompt=prompt,
            claim=claim,
            memories=used_memories,
            observations=observations,
            errors=errors,
            claim_step_id=claim_step,
            user_input_id=user_input_id,
        )
        fresh_plans: List[Dict[str, Any]] = []
        seen_this_pass: set[str] = set()
        for step in plans:
            key = helper_agent._call_key(step["tool_name"], step.get("args", {}) or {})
            if key in seen_this_pass or key in executed:
                continue
            if helper_agent._is_read_only(step["tool_name"]) and key in helper_agent.executed:
                continue
            seen_this_pass.add(key)
            fresh_plans.append(step)
        if not fresh_plans:
            break

        plan_step = _log(
            trace_logger,
            task_id,
            "plan",
            content=json.dumps(fresh_plans, ensure_ascii=False),
            prev_step_ids=[prev_step_for_plan],
            used_ids=plan_used,
        )
        prev_step_for_plan = plan_step
        spawned_action_steps: List[str] = []
        for step in fresh_plans:
            tool_name = step["tool_name"]
            tool_args = step.get("args", {}) or {}
            key = helper_agent._call_key(tool_name, tool_args)
            executed.add(key)
            action_step = _log(
                trace_logger,
                task_id,
                "tool_action",
                content=f"Call {tool_name}",
                prev_step_ids=[plan_step],
                tool_name=tool_name,
                tool_args=tool_args,
            )
            spawned_action_steps.append(action_step)
            obs = helper_agent.execute_tool(tool_name, **tool_args)
            obs_step = _log(
                trace_logger,
                task_id,
                "tool_observation",
                content=json.dumps(obs, ensure_ascii=False, default=str),
                prev_step_ids=[action_step],
                tool_name=tool_name,
            )
            observation_steps.append(obs_step)
            if "error" in obs:
                errors.append(
                    {"tool_name": tool_name, "args": tool_args, "error": obs["error"]}
                )
                continue
            if helper_agent._is_read_only(tool_name):
                helper_agent.executed.add(key)
            tool_calls.append(
                {"tool_name": tool_name, "args": tool_args, "observation": obs}
            )
            observations.append(obs)
        trace_logger.set_next_step_ids(plan_step, spawned_action_steps)

    response, response_used = _generate_response(
        tracker=tracker,
        config=config,
        history=history,
        prompt=prompt,
        claim=claim,
        observations=observations,
        memories=used_memories,
        claim_step_id=claim_step,
        observation_steps=observation_steps,
        user_input_id=user_input_id,
    )
    final_prev = observation_steps if observation_steps else [claim_step]
    final_step = _log(
        trace_logger,
        task_id,
        "final_answer",
        content=response,
        prev_step_ids=final_prev,
        used_ids=response_used,
    )

    existing_memories = [
        memory for memory in memory_store.list_memories(task_id=task_id)
        if memory.status == "active"
    ]
    changes = _extract_memories(
        tracker=tracker,
        config=config,
        history=history,
        memory_store=memory_store,
        task_id=task_id,
        prompt=prompt,
        response=response,
        existing=existing_memories,
        relied_history=relied_history,
        observations=observations,
        claim=claim,
        claim_step_id=claim_step,
        observation_steps=observation_steps,
        user_input_id=user_input_id,
    )

    for memory, used_ids in changes["created"]:
        _log(
            trace_logger,
            task_id,
            "memory_write",
            content=memory.content,
            prev_step_ids=[final_step],
            generated_memory_ids=[memory.memory_id],
            used_ids=used_ids,
        )
        memory.last_modified_by = user_input_id
    for memory, used_ids in changes["updated"]:
        step_id = _log(
            trace_logger,
            task_id,
            "memory_update",
            content=memory.content,
            prev_step_ids=[final_step],
            invalidated_memory_ids=[memory.supersedes] if memory.supersedes else [],
            generated_memory_ids=[memory.memory_id],
            used_ids=used_ids,
        )
        memory.last_modified_by = step_id
    for memory, used_ids in changes["deleted"]:
        step_id = _log(
            trace_logger,
            task_id,
            "memory_delete",
            content=f"Deleted memory: {memory.content}",
            prev_step_ids=[final_step],
            invalidated_memory_ids=[memory.memory_id],
            used_ids=used_ids,
        )
        memory.last_modified_by = step_id
    for memory, used_ids in changes["consolidated"]:
        step_id = _log(
            trace_logger,
            task_id,
            "memory_consolidate",
            content=memory.content,
            prev_step_ids=[final_step],
            invalidated_memory_ids=list(memory.derived_from),
            generated_memory_ids=[memory.memory_id],
            used_ids=used_ids,
        )
        memory.last_modified_by = step_id

    return FullResetLifecycleResult(
        response=response,
        repaired_memories=memory_store.get_memories(),
        new_final_turn_steps=[
            step.to_dict() for step in trace_logger.trace_steps[trace_start:]
        ],
        tracker=tracker,
        reset_deleted_memory_ids=reset_deleted_ids,
    )
