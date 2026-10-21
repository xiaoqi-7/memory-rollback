import inspect
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from client.model_client import run_llm

from . import prompts
from .memory_store import Memory, MemoryStore
from .trace_logger import TraceLogger, TraceStep


@dataclass
class HistoryTurn:
    """A prior user turn retained in-session for cross-turn citation.

    Carries the turn's real ``user_input_id`` alongside its text so the SAME
    utterance keeps ONE string identity whether it is the current turn (minted
    as ``user_input_{turn_index}`` in ``run``) or later referenced from
    ``self.history``. There is no position-derived ``h{n}`` alias: a turn cited
    from history is cited by the exact id it carried while current, so cross-turn
    provenance (e.g. a memory's ``last_modified_by``) lines up, and history is
    free to be reordered or have turns removed without breaking citations.
    """

    user_input_id: str
    text: str


def _parse_json(text: str) -> Any:
    """Best-effort parse of an LLM response into a JSON object.

    Tolerates markdown code fences and leading/trailing prose by extracting
    the first balanced JSON object/array found in the text.
    """
    if text is None:
        raise ValueError("Empty LLM response")

    cleaned = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to the first {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]!r}")


class ReactAgent:
    """Agent that follows the REACT loop: retrieve, reason, act, respond.

    The agent is LLM-driven. Each task session step retrieves relevant
    memories, forms a claim about user intent, plans tool calls, executes
    them against the domain simulator, generates a final answer, and writes
    durable memories back to the store. Every step is recorded in the trace.
    """

    def __init__(
        self,
        tools: Dict[str, Callable],
        config: Dict[str, Any],
    ) -> None:
        """Create a ReactAgent with a memory store, trace logger, and tools.

        `tools` is a mapping of tool name -> callable (the domain entry from
        the TOOLS registry).
        """
        self.memory_store = MemoryStore()
        self.trace_logger = TraceLogger()
        # Tolerate a list of names (no callables) for lightweight/test use.
        if isinstance(tools, dict):
            self.tools: Dict[str, Callable] = tools
        else:
            self.tools = {name: None for name in (tools or [])}
        self.config = config
        self.model: Optional[str] = config.get("model")
        self.domain_prompt: str = config.get(
            "domain_prompt", "You are a helpful assistant."
        )
        # Domain-specific guidance on what is worth remembering, injected into
        # the memory-reconciliation prompt. Each config supplies its own so the
        # agent isn't hard-coded to one domain's vocabulary.
        self.memory_guidance: str = config.get("memory_guidance", "")
        # Domain-specific guidance on which user requests map to which tools,
        # injected into the planning prompt. Like memory_guidance, each config
        # supplies its own so the planner's trigger rules aren't hard-coded.
        self.tool_guidance: str = config.get("tool_guidance", "")
        # Tool names that take a real, user-visible action (create/book/cancel
        # etc.). These are NEVER deduplicated across turns — every imperative
        # must reach ``execute_tool`` so the planner's own repeat rules decide
        # whether to act (the user may legitimately ask for a second ticket
        # with the same details). A config may list them explicitly via
        # ``side_effecting_tools``; otherwise a name heuristic is used.
        self.side_effecting_tools: set = {
            str(name).lower() for name in config.get("side_effecting_tools", [])
        }
        # Session-scoped set of read-only (tool_name, args) calls already
        # executed this session. A duplicate read-only call planned on a later
        # turn is DROPPED rather than re-run, forcing the agent to answer from
        # the memory written from the first call's observation instead of
        # re-fetching it. Side-effecting calls are never recorded here. Reset
        # per session in ``clear``.
        self.executed: set = set()
        # Set per-run so trace steps and memories carry the right task_id.
        self.task_id: Optional[str] = None
        # 1-based index of the current user turn in the session. Used to mint
        # the user-input id (e.g. "user_input_1") that a freshly written memory
        # records as its author. Reset per session in ``clear``.
        self.turn_index: int = 0
        # Prior user turns in this session, each retaining its real
        # user_input_id (see HistoryTurn). A turn the agent relies on is promoted
        # into a durable memory (source="user_history") and dropped from this
        # list, so the store stays the single source of truth.
        self.history: List[HistoryTurn] = []

    # ------------------------------------------------------------------ #
    # Tool introspection
    # ------------------------------------------------------------------ #
    def _tool_specs(self) -> str:
        """Render available tools (name, signature, summary) for the planner."""
        lines: List[str] = []
        for name, func in self.tools.items():
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

    # ------------------------------------------------------------------ #
    # REACT sub-steps
    # ------------------------------------------------------------------ #
    def find_relevant_memories(self, user_prompt: str, limit: int = 10) -> List[Memory]:
        memories = self.memory_store.find_relevant_memories(
            query=user_prompt,
            limit=limit,
            task_id=self.task_id,
        )
        # Hand the agent sanitized copies so fault-injection bookkeeping never
        # leaks into the reasoning context. Copy to avoid mutating the store.
        sanitized: List[Memory] = []
        for memory in memories:
            clone = deepcopy(memory)
            clone.last_modified_by = None
            clone.fault_type = None
            if clone.source == "injected_fault":
                clone.source = "injected"
            sanitized.append(clone)
        return sanitized

    def make_claim(
        self, prompt: str, memories: List[Memory], user_input_id: str = ""
    ) -> Dict[str, Any]:
        """Form a concise claim about user intent from prompt + memories.

        Considers prior user turns (``self.history``) alongside durable
        memories, and reports which turns it relied on so the caller can
        promote them into the store. ``user_input_id`` is the CURRENT turn's
        own input id; it is shown to the LLM next to the request and is a valid
        citation target, so a claim that leaned directly on the user's current
        wording can cite it.
        """
        memory_block = "\n".join(
            f"[{m.memory_id}] {m.content}" for m in memories
        ) or "(no relevant memories)"
        history_block = "\n".join(
            f"[{turn.user_input_id}] {turn.text}" for turn in self.history
        ) or "(no prior turns)"

        system = prompts.claim_system(self.domain_prompt)
        user = prompts.claim_user(prompt, memory_block, history_block, user_input_id)

        raw = run_llm(system, user, model=self.model, response_format={"type": "json_object"})
        data = _parse_json(raw)
        if not isinstance(data, dict):
            data = {}
        known = {m.memory_id for m in memories}
        # The LLM emits a single flat "used" list mixing memory ids and history
        # ids; split it in code. History is now cited by each turn's real
        # user_input_id (the same id it carried while current), not a positional
        # "h{n}", so run() can resolve a cited id directly to its HistoryTurn to
        # promote the relied-on turns. The memory-id part feeds the combined list
        # returned for the claim step's used_ids. The current turn's own
        # user_input_id is also citable; it is grouped with history ids so it
        # survives into the combined used list (run() resolves only PAST-turn ids
        # against self.history, so the current id never spuriously promotes a
        # turn that is not in history yet).
        history_ids = {turn.user_input_id for turn in self.history}
        if user_input_id:
            history_ids = history_ids | {user_input_id}
        raw_used = data.get("used", [])
        used = [mid for mid in raw_used if mid in known]
        used_history = [hid for hid in raw_used if hid in history_ids]
        used_history = list(dict.fromkeys(used_history))
        return {
            "claim": str(data.get("claim", "")).strip(),
            "used_memory_ids": used,
            "used_history_ids": used_history,
            "used": used + used_history,
        }

    @staticmethod
    def _normalize_used(
        raw: Any,
        *,
        known_memory_ids: set,
        valid_step_ids: set,
        valid_history_ids: set,
    ) -> List[str]:
        """Validate a flat "used" list of mixed-type ids.

        Keeps memory ids that were actually retrieved, step ids that name a
        real claim/tool_observation step this turn, and history ids that name a
        prior user turn present in scope (by its real ``user_input_id``); drops
        anything else and de-duplicates while preserving order.
        """
        allowed = known_memory_ids | valid_step_ids | valid_history_ids
        out = [uid for uid in (raw or []) if uid in allowed]
        return list(dict.fromkeys(out))

    def plan_actions(
        self,
        prompt: str,
        claim: Dict[str, Any],
        memories: Optional[List[Memory]] = None,
        observations: Optional[List[Dict[str, Any]]] = None,
        errors: Optional[List[Dict[str, Any]]] = None,
        claim_step_id: Optional[str] = None,
        user_input_id: Optional[str] = None,
    ) -> tuple:
        """Plan an ordered list of tool actions to satisfy the claim.

        Returns ``(steps, used_ids)``: the validated tool steps and the flat
        list of ids the plan relied on, taken EXACTLY from the LLM's own "used"
        output. ``claim_step_id`` is shown to the planner next to the claim and
        is a valid citation target; if the plan LLM cites it, it appears in
        ``used_ids`` — otherwise it does not. ``user_input_id`` is the CURRENT
        turn's own input id, shown next to the request and likewise citable when
        the plan leaned directly on the user's current wording. ``run()`` logs
        ``used_ids`` verbatim and never force-adds the claim step itself.

        ``memories`` are the durable facts retrieved for this turn; the planner
        uses them to fill argument values the user established on an earlier turn
        but did not repeat (e.g. a destination or product id). ``observations``
        are the tool results gathered so far on this turn; passing them lets the
        planner chain a follow-up call whose arguments depend on an earlier
        call's output (e.g. ``check_shipping`` needs the ``product_id`` that
        ``search_product`` returned). ``errors`` are failed calls from earlier
        passes this turn; surfacing them lets the planner self-correct (e.g. a
        "product not found" error from passing a name where an id was needed
        prompts it to call the search tool first and retry with the resolved
        id). Read-only tools (search/lookup/policy) are planned whenever the
        request maps to them — deciding such a tool is unnecessary is the user's
        call, not the agent's. Side-effecting tools (create_ticket, book, etc.)
        are planned only when the user's current message is an explicit
        instruction to take that action now, not when they are merely asking
        whether to, considering it, clarifying a detail, or retracting it.
        """
        if not self.tools:
            return [], []

        claim_text = claim.get("claim", "") if isinstance(claim, dict) else str(claim)
        memory_block = "\n".join(
            f"[{m.memory_id}] {m.content}" for m in (memories or [])
        ) or "(no relevant memories)"
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
        guidance_block = prompts.plan_guidance_block(self.tool_guidance)

        system = prompts.plan_system(
            self.domain_prompt,
            guidance_block,
            self._tool_specs(),
            memory_block,
            obs_block,
            error_block,
        )
        user = prompts.plan_user(prompt, claim_text, claim_step_id or "", user_input_id or "")

        raw = run_llm(system, user, model=self.model, response_format={"type": "json_object"})
        data = _parse_json(raw)
        steps = data.get("steps", []) if isinstance(data, dict) else data
        valid: List[Dict[str, Any]] = []
        for step in steps or []:
            if not isinstance(step, dict):
                continue
            tool_name = step.get("tool_name")
            if tool_name in self.tools:
                valid.append(
                    {
                        "type": "tool",
                        "tool_name": tool_name,
                        "args": step.get("args", {}) or {},
                    }
                )
        # The plan's cited reliance, taken exactly from the LLM's own "used"
        # output: memory ids validated against what was retrieved, history ids
        # against the real user_input_ids in scope, and the claim step id only if
        # the LLM actually cited it. run() logs this verbatim — it does NOT
        # prepend the claim step itself.
        known = {m.memory_id for m in (memories or [])}
        valid_step_ids = {sid for sid in (claim_step_id, user_input_id) if sid}
        raw_used = data.get("used", []) if isinstance(data, dict) else []
        used_ids = self._normalize_used(
            raw_used,
            known_memory_ids=known,
            valid_step_ids=valid_step_ids,
            valid_history_ids={turn.user_input_id for turn in self.history},
        )
        return valid, used_ids

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute a named tool and return its observation as a dict."""
        func = self.tools.get(tool_name)
        if not callable(func):
            return {"error": f"unknown tool: {tool_name}"}
        try:
            result = func(**kwargs)
        except TypeError as exc:
            return {"error": f"bad arguments for {tool_name}: {exc}"}
        except Exception as exc:  # surfaced to the caller, logged as observation
            return {"error": str(exc)}
        return result if isinstance(result, dict) else {"result": result}

    def generate_response(
        self,
        prompt: str,
        claim: Dict[str, Any],
        observations: List[Dict[str, Any]],
        memories: Optional[List[Memory]] = None,
        claim_step_id: Optional[str] = None,
        observation_steps: Optional[List[str]] = None,
        user_input_id: Optional[str] = None,
    ) -> tuple:
        """Generate the agent's final natural-language answer.

        ``memories`` are the durable facts retrieved for this turn. They let the
        agent ground an answer from prior turns when no tool ran this turn (e.g.
        a product already established earlier), instead of falsely claiming it
        found nothing.

        ``claim_step_id`` and ``observation_steps`` (this turn's
        tool_observation step ids, index-aligned with ``observations``) are
        shown to the LLM next to the claim and each tool result so it can cite
        the exact step ids it relied on. ``user_input_id`` is the CURRENT turn's
        own input id, shown next to the request and likewise citable when the
        answer leaned directly on the user's current wording.

        Returns ``(response_text, used_ids)``: the answer plus the ids the LLM
        itself cited (memory ids validated against what was retrieved, plus the
        claim step id and any tool_observation step ids it named). ``run()``
        logs this verbatim — it does NOT force-add the claim step or the full
        observation_steps list.
        """
        claim_text = claim.get("claim", "") if isinstance(claim, dict) else str(claim)
        observation_steps = observation_steps or []
        if observations:
            obs_block = "\n".join(
                f"[{obs_step}] {json.dumps(obs, ensure_ascii=False, default=str)}"
                for obs_step, obs in zip(observation_steps, observations)
            )
        else:
            obs_block = "(no tool results)"
        memory_block = "\n".join(
            f"[{m.memory_id}] {m.content}" for m in (memories or [])
        ) or "(no relevant memories)"

        system = prompts.response_system(self.domain_prompt)
        user = prompts.response_user(
            prompt, claim_text, memory_block, obs_block, claim_step_id or "",
            user_input_id or "",
        )
        raw = run_llm(system, user, model=self.model, response_format={"type": "json_object"})
        data = _parse_json(raw)
        if not isinstance(data, dict):
            data = {}
        response_text = str(data.get("response", "")).strip()
        known = {m.memory_id for m in (memories or [])}
        valid_step_ids = set(observation_steps)
        if claim_step_id:
            valid_step_ids.add(claim_step_id)
        if user_input_id:
            valid_step_ids.add(user_input_id)
        used_ids = self._normalize_used(
            data.get("used", []),
            known_memory_ids=known,
            valid_step_ids=valid_step_ids,
            valid_history_ids={turn.user_input_id for turn in self.history},
        )
        return response_text, used_ids

    def extract_memories(
        self,
        prompt: str,
        response: str,
        existing: Optional[List[Memory]] = None,
        history_turns: Optional[List[str]] = None,
        observations: Optional[List[Dict[str, Any]]] = None,
        claim: Optional[Dict[str, Any]] = None,
        claim_step_id: Optional[str] = None,
        observation_steps: Optional[List[str]] = None,
        user_input_id: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        """Reconcile durable memories after an interaction.

        Asks the LLM to maintain the session's memory by (a) creating new
        durable facts, (b) updating existing memories whose content is now
        stale or should be refined, (c) deleting memories that are
        contradicted or no longer valid, and (d) consolidating two or more
        related memories into a single summary. ``history_turns`` are prior
        user turns the agent relied on this turn; durable facts distilled from
        them are created with ``source="user_history"``. ``observations`` are
        the tool results gathered this turn; durable facts grounded in them are
        created with ``source="tool_observation"``. ``claim`` (with its
        ``claim_step_id``) is shown to the LLM as input context and is a valid
        citation target; ``observation_steps`` are this turn's tool_observation
        step ids (index-aligned with ``observations`` and rendered next to each
        observation), also citable. ``user_input_id`` is the CURRENT turn's own
        input id, shown next to the request and likewise citable when an entry
        leaned directly on the user's current wording (e.g. a deletion prompted
        by the user saying "forget X" this turn). Each entry's ``used`` list
        (memory ids, history ids, the claim step, tool_observation steps, and/or
        the current user request id it drew on) — for new, updates,
        consolidations, AND deletes — is validated by the LLM's own output and
        returned ALONGSIDE the Memory object: provenance is not carried on the
        Memory. The changes are applied to the store (updates supersede the
        old memory; consolidations supersede their sources; deletes are marked,
        not removed) and the results are returned grouped by action, with
        per-memory citations paired in:
        ``{"created": [(memory, used_ids), ...],
        "updated": [(memory, used_ids), ...],
        "deleted": [(memory, used_ids), ...],
        "consolidated": [(memory, used_ids), ...]}``.
        """
        existing = existing or []
        history_turns = history_turns or []
        observations = observations or []
        observation_steps = observation_steps or []
        claim_text = (
            claim.get("claim", "") if isinstance(claim, dict) else str(claim or "")
        )
        existing_block = "\n".join(
            f"[{m.memory_id}] (source={m.source}, trust={m.trust_score}) {m.content}"
            for m in existing
        ) or "(no existing memories)"
        history_block = "\n".join(
            f"[{turn.user_input_id}] {turn.text}" for turn in history_turns
        ) or "(no prior turns relied on)"
        if observations:
            obs_block = "\n".join(
                f"[{obs_step}] {json.dumps(obs, ensure_ascii=False, default=str)}"
                for obs_step, obs in zip(observation_steps, observations)
            )
        else:
            obs_block = "(no tool results)"
        claim_block = (
            f"[{claim_step_id}] {claim_text}" if claim_step_id else claim_text
        ) or "(no claim)"

        guidance_block = prompts.memory_guidance_block(self.memory_guidance)

        system = prompts.extract_system(self.domain_prompt, guidance_block)
        user = prompts.extract_user(
            prompt, response, history_block, obs_block, existing_block,
            claim_block, user_input_id or "",
        )

        raw = run_llm(system, user, model=self.model, response_format={"type": "json_object"})
        data = _parse_json(raw)
        if not isinstance(data, dict):
            data = {}

        known = {m.memory_id for m in existing}
        # Step ids the LLM may cite in a "used" list: the claim step, this
        # turn's tool_observation steps, and the current turn's own user input
        # id (so an entry — including a deletion — that leaned directly on the
        # user's current wording can cite it).
        valid_step_ids = set(observation_steps)
        if claim_step_id:
            valid_step_ids.add(claim_step_id)
        if user_input_id:
            valid_step_ids.add(user_input_id)
        # History ids the LLM may cite: the real user_input_id of each relied-on
        # prior turn (the same id it carried while current).
        valid_history_ids = {turn.user_input_id for turn in history_turns}

        def used_for(item: Dict[str, Any]) -> List[str]:
            return self._normalize_used(
                item.get("used", []),
                known_memory_ids=known,
                valid_step_ids=valid_step_ids,
                valid_history_ids=valid_history_ids,
            )

        acted: set = set()  # ids already updated/deleted/consolidated this pass
        # Each created/updated/deleted/consolidated entry is paired with the
        # used_ids the LLM cited for it, so provenance travels alongside the
        # Memory rather than riding on the domain object. For deletes the
        # citation records WHY the deletion happened (the retracting user input,
        # a contradicting memory, a tool result proving the old value wrong),
        # which invalidated_memory_ids alone cannot capture.
        created: List[tuple] = []
        updated: List[tuple] = []
        deleted: List[tuple] = []
        consolidated: List[tuple] = []

        # New memories.
        for item in data.get("new", []) or []:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            memory = Memory(
                memory_id="",  # assigned by the store
                task_id=self.task_id or "",
                content=content,
                source=str(item.get("source", "agent_summary")),
                last_modified_by="agent",
                trust_score=item.get("trust_score"),
                status="active",
            )
            self.memory_store.add_memory(memory)
            created.append((memory, used_for(item)))

        # Updates: supersede the original with a refined active memory.
        for item in data.get("updates", []) or []:
            if not isinstance(item, dict):
                continue
            memory_id = item.get("memory_id")
            if memory_id not in known or memory_id in acted:
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            updates: Dict[str, Any] = {
                "content": content,
                "last_modified_by": "agent",
            }
            if item.get("trust_score") is not None:
                updates["trust_score"] = item.get("trust_score")
            try:
                new_id = self.memory_store.update_memory(memory_id, updates)
            except (KeyError, ValueError):
                continue
            acted.add(memory_id)
            new_memory = self.memory_store.get_memory(new_id)
            if new_memory is not None:
                updated.append((new_memory, used_for(item)))

        # Deletes: mark contradicted/invalid memories (skip any just updated).
        for item in data.get("deletes", []) or []:
            if not isinstance(item, dict):
                continue
            memory_id = item.get("memory_id")
            if memory_id not in known or memory_id in acted:
                continue
            try:
                self.memory_store.delete_memory(memory_id)
            except KeyError:
                continue
            acted.add(memory_id)
            deleted_memory = self.memory_store.get_memory(memory_id)
            if deleted_memory is not None:
                deleted.append((deleted_memory, used_for(item)))

        # Consolidations: merge two or more memories into a single summary that
        # supersedes its sources (recorded via the new memory's derived_from).
        for item in data.get("consolidations", []) or []:
            if not isinstance(item, dict):
                continue
            source_ids = [
                mid
                for mid in (item.get("memory_ids") or [])
                if mid in known and mid not in acted
            ]
            # De-duplicate while preserving order; need at least two to merge.
            source_ids = list(dict.fromkeys(source_ids))
            if len(source_ids) < 2:
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            try:
                new_id = self.memory_store.consolidate_memories(
                    source_ids,
                    content=content,
                    trust_score=item.get("trust_score"),
                )
            except (KeyError, ValueError):
                continue
            acted.update(source_ids)
            new_memory = self.memory_store.get_memory(new_id)
            if new_memory is not None:
                consolidated.append((new_memory, used_for(item)))

        return {
            "created": created,
            "updated": updated,
            "deleted": deleted,
            "consolidated": consolidated,
        }

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        """Clear the memory store and trace logger for a new task session."""
        self.memory_store = MemoryStore()
        self.trace_logger = TraceLogger()
        self.task_id = None
        self.turn_index = 0
        self.history = []
        self.executed = set()

    @staticmethod
    def _call_key(tool_name: str, args: Dict[str, Any]) -> str:
        """Stable key for a (tool, args) pair used by the duplicate-call guard."""
        return json.dumps(
            [tool_name, args or {}], sort_keys=True, default=str
        )

    def _is_read_only(self, tool_name: str) -> bool:
        """Whether a tool only fetches information (safe to deduplicate).

        Side-effecting tools (create/book/cancel/etc.) must always reach
        ``execute_tool`` so the planner's own repeat rules decide whether to
        act, so they are never deduplicated across turns. A config can name them
        explicitly via ``side_effecting_tools``; otherwise a verb heuristic on
        the tool name is used, defaulting to read-only when unsure.
        """
        name = tool_name.lower()
        if name in self.side_effecting_tools:
            return False
        read_only_prefixes = ("search_", "lookup_", "check_", "compare_")
        if name.startswith(read_only_prefixes):
            return True
        side_effect_verbs = (
            "create", "open", "book", "reserve", "cancel", "file", "raise",
            "place", "submit", "update", "delete", "pay", "refund", "schedule",
        )
        return not any(verb in name for verb in side_effect_verbs)

    def restore_executed_from_trace(self) -> None:
        """Rebuild the session-scoped read-only call guard from loaded trace.

        In faulty-mode replay the cut trace steps include the tool calls the
        agent made before the replay boundary. ``self.executed`` records the
        read-only ``(tool, args)`` calls already run this session so a later
        turn that re-plans the same read-only call is DROPPED — forcing the
        agent to answer from the memory written from the first call instead of
        re-fetching it. After ``clear`` resets it, replaying without rebuilding
        it would let a replayed turn re-run a read-only call already made before
        the boundary, getting a fresh observation and bypassing the
        memory-dependency path the benchmark exercises. Only successful
        read-only calls are restored, matching ``run`` which records a call only
        after its observation came back error-free.
        """
        steps = self.trace_logger.trace_steps
        # A tool_observation links back to its tool_action via
        # prev_step_ids=[action_step]; an observation carrying an "error" key
        # marks that action as failed, so it was never recorded in run().
        errored_action_ids: set = set()
        for step in steps:
            if step.step_type != "tool_observation":
                continue
            try:
                obs = json.loads(step.content)
            except (json.JSONDecodeError, TypeError):
                obs = None
            if isinstance(obs, dict) and "error" in obs:
                for prev_id in step.prev_step_ids:
                    if prev_id.startswith("s_"):
                        errored_action_ids.add(prev_id)

        for step in steps:
            if step.step_type != "tool_action" or not step.tool_name:
                continue
            if step.step_id in errored_action_ids:
                continue
            if not self._is_read_only(step.tool_name):
                continue
            self.executed.add(self._call_key(step.tool_name, step.tool_args or {}))

    def _log(
        self,
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
        turn_id: Optional[str] = None,
        user_input_id: Optional[str] = None,
        semantic_slots: Optional[Dict[str, Any]] = None,
        oracle_refs: Optional[List[str]] = None,
        failure_label: Optional[str] = None,
    ) -> str:
        """Append a trace step and return its assigned step_id.

        ``prev_step_ids`` carries the trace step ids this step directly follows
        from and ``next_step_ids`` the trace step ids that directly follow from
        it; together they form one bidirectional graph over STEPS ONLY (no
        memory ids). The logger back-links successors into a predecessor's
        ``next_step_ids`` as they are recorded.

        ``invalidated_memory_ids``/``generated_memory_ids`` carry memory lineage
        on memory-action steps: the memory ids this step invalidated (superseded,
        deleted, or merged away) and the memory ids it produced. They are kept
        distinct from the pure step-sequencing fields above.

        ``used_ids`` is a flat list of the mixed-type ids this step semantically
        relied on — memory ids ("m_*"), the claim/tool_observation step ids
        ("s_*") it drew on, and prior-user-turn ids ("user_input_{n}") — kept
        distinct from the structural step/memory links above.
        """
        step = TraceStep(
            step_id="",  # assigned by the logger
            task_id=self.task_id,
            step_type=step_type,
            content=content,
            prev_step_ids=list(prev_step_ids or []),
            next_step_ids=list(next_step_ids or []),
            invalidated_memory_ids=list(invalidated_memory_ids or []),
            generated_memory_ids=list(generated_memory_ids or []),
            used_ids=used_ids or [],
            tool_name=tool_name,
            tool_args=tool_args,
            timestamp=None,
            turn_id=turn_id,
            user_input_id=user_input_id,
            semantic_slots=semantic_slots,
            oracle_refs=oracle_refs,
            failure_label=failure_label,
        )
        self.trace_logger.log_step(step)
        return step.step_id

    def run(self, task_id: str, user_prompt: str) -> Dict[str, Any]:
        """Run the full REACT loop for a single prompt in a task session.

        Returns a dict with `response`, `tool_calls`, `used_memories`,
        `new_memories`, `tool_observations`, and `claim`.
        """
        self.task_id = task_id
        prompt = user_prompt
        # 1-based turn index for this session; mints the user-input id that a
        # freshly written memory records as its author.
        self.turn_index += 1
        user_input_id = f"user_input_{self.turn_index}"
        turn_id = f"turn_{self.turn_index:03d}"

        # 1) Retrieve relevant memories.
        used_memories = self.find_relevant_memories(
            prompt, limit=self.config.get("retrieve_limit", 10)
        )
        read_step = self._log(
            "memory_read",
            content=f"Retrieved {len(used_memories)} relevant memories.",
            used_ids=[m.memory_id for m in used_memories],
            turn_id=turn_id,
            user_input_id=user_input_id,
        )

        # 2) Form a claim about user intent.
        claim = self.make_claim(prompt, used_memories, user_input_id=user_input_id)
        claim_step = self._log(
            "claim",
            content=claim.get("claim", ""),
            prev_step_ids=[read_step],
            used_ids=claim.get("used", []),
            turn_id=turn_id,
            user_input_id=user_input_id,
        )

        # Capture the prior user turns the claim relied on, by their real
        # user_input_id (the same id each turn carried while current). They are
        # distilled into durable facts during memory reconciliation (step 6) and
        # then dropped from history so the store stays the single source of
        # truth. Resolving by id (not position) keeps this correct even if
        # history is later reordered.
        history_by_id = {turn.user_input_id: turn for turn in self.history}
        relied_history_ids = [
            hid for hid in claim.get("used_history_ids", []) if hid in history_by_id
        ]
        relied_history = [history_by_id[hid] for hid in relied_history_ids]

        # 3-4) Plan and execute tool actions iteratively. Re-planning with the
        # observations gathered so far lets the agent chain calls whose
        # arguments depend on an earlier call's output (e.g. check_shipping
        # needs the product_id that search_product returns) and recall argument
        # values from memory. Each pass plans the calls still needed; the loop
        # stops when a pass plans nothing new or the iteration cap is reached.
        max_plan_iters = self.config.get("max_plan_iters", 3)
        tool_calls: List[Dict[str, Any]] = []
        observations: List[Dict[str, Any]] = []
        # Errored observations are kept out of the final answer but fed back to
        # the planner so it can self-correct (e.g. a "product not found" error
        # from passing a name where an id was needed prompts it to call the
        # search tool first and retry with the resolved id).
        errors: List[Dict[str, Any]] = []
        observation_steps: List[str] = []
        executed: set = set()  # (tool, args) run THIS turn (all tools), to avoid loops
        prev_step_for_plan = claim_step
        for _ in range(max(1, max_plan_iters)):
            plans, plan_used = self.plan_actions(
                prompt,
                claim,
                memories=used_memories,
                observations=observations,
                errors=errors,
                claim_step_id=claim_step,
                user_input_id=user_input_id,
            )
            # Filter the planned calls before executing any:
            # - drop exact duplicates within this pass (planner listed the same
            #   call twice),
            # - drop calls already run THIS turn (a re-plan echoing prior calls),
            # - drop read-only calls already run earlier in this SESSION
            #   (``self.executed``). The duplicate read-only call is dropped, not
            #   re-run, so the agent has no fresh observation for it and must
            #   answer from the memory written from the first call — that is the
            #   memory-dependency path we want to exercise. Side-effecting calls
            #   are NOT session-deduplicated: a later imperative to repeat the
            #   action (e.g. open another ticket) must reach execute_tool, where
            #   the planner's own repeat rules govern it.
            fresh_plans: List[Dict[str, Any]] = []
            seen_this_pass: set = set()
            for step in plans:
                key = self._call_key(step["tool_name"], step.get("args", {}) or {})
                if key in seen_this_pass or key in executed:
                    continue
                if self._is_read_only(step["tool_name"]) and key in self.executed:
                    continue
                seen_this_pass.add(key)
                fresh_plans.append(step)
            # Nothing new to do this pass: skip logging an empty "[]" plan step
            # and stop the loop.
            if not fresh_plans:
                break
            plan_step = self._log(
                "plan",
                content=json.dumps(fresh_plans, ensure_ascii=False),
                prev_step_ids=[prev_step_for_plan],
                used_ids=plan_used,
                turn_id=turn_id,
                user_input_id=user_input_id,
            )
            prev_step_for_plan = plan_step
            # action_step ids spawned from THIS plan pass; written back to the
            # plan step's next_step_ids after the inner loop so the plan links
            # forward to every tool_action it produced (a single pass can spawn
            # more than one).
            spawned_action_steps: List[str] = []
            for step in fresh_plans:
                tool_name = step["tool_name"]
                tool_args = step.get("args", {}) or {}
                key = self._call_key(tool_name, tool_args)
                executed.add(key)
                action_step = self._log(
                    "tool_action",
                    content=f"Call {tool_name}",
                    prev_step_ids=[plan_step],
                    tool_name=tool_name,
                    tool_args=tool_args,
                    turn_id=turn_id,
                    user_input_id=user_input_id,
                )
                spawned_action_steps.append(action_step)
                obs = self.execute_tool(tool_name, **tool_args)
                obs_step = self._log(
                    "tool_observation",
                    content=json.dumps(obs, ensure_ascii=False, default=str),
                    prev_step_ids=[action_step],
                    tool_name=tool_name,
                    tool_args=tool_args,
                    turn_id=turn_id,
                    user_input_id=user_input_id,
                )
                observation_steps.append(obs_step)
                if "error" in obs:
                    # Keep the failure out of the final answer, but feed it back
                    # to the next planning pass so the planner can self-correct.
                    errors.append({"tool_name": tool_name, "args": tool_args, "error": obs["error"]})
                    continue
                # Record successful read-only calls at session scope so a later
                # turn that re-plans the same call is dropped (forcing reuse of
                # the memory written from this observation). Side-effecting calls
                # are left out so a later imperative can repeat the action.
                if self._is_read_only(tool_name):
                    self.executed.add(key)
                tool_calls.append({"tool_name": tool_name, "args": tool_args, "observation": obs})
                observations.append(obs)
            # The inner loop has now created every tool_action for this plan
            # pass; record them as the plan step's successors. Done here rather
            # than at plan creation because the action steps did not exist yet
            # when the plan step was logged.
            self.trace_logger.set_next_step_ids(plan_step, spawned_action_steps)

        # 5) Generate the final response.
        response, response_used = self.generate_response(
            prompt,
            claim,
            observations,
            memories=used_memories,
            claim_step_id=claim_step,
            observation_steps=observation_steps,
            user_input_id=user_input_id,
        )
        final_step = self._log(
            "final_answer",
            content=response,
            prev_step_ids=observation_steps if observation_steps is not None else [claim_step],
            used_ids=response_used,
            turn_id=turn_id,
            user_input_id=user_input_id,
        )

        # 6) Reconcile session memories: create, update, and delete.
        existing_memories = [
            m
            for m in self.memory_store.list_memories(task_id=self.task_id)
            if m.status == "active"
        ]
        changes = self.extract_memories(
            prompt,
            response,
            existing_memories,
            history_turns=relied_history,
            observations=observations,
            claim=claim,
            claim_step_id=claim_step,
            observation_steps=observation_steps,
            user_input_id=user_input_id,
        )
        # created/updated/deleted/consolidated all arrive as (memory, used_ids)
        # tuples so provenance stays off the Memory object — deletes now carry
        # the citation justifying WHY the deletion happened too.
        created_pairs = changes["created"]
        updated_pairs = changes["updated"]
        deleted_pairs = changes["deleted"]
        consolidated_pairs = changes["consolidated"]
        new_memories = [memory for memory, _ in created_pairs]
        updated_memories = [memory for memory, _ in updated_pairs]
        deleted_memories = [memory for memory, _ in deleted_pairs]
        consolidated_memories = [memory for memory, _ in consolidated_pairs]

        # Cited turns have now been distilled into memories; drop them so the
        # store stays the single source of truth. Remove by matching
        # user_input_id rather than position, so this stays correct regardless of
        # any reordering of history.
        history_was_written = any(
            m.source == "user_history" for m in new_memories
        )
        if history_was_written:
            drop_ids = set(relied_history_ids)
            self.history = [
                turn for turn in self.history if turn.user_input_id not in drop_ids
            ]

        # Attribute each memory change to its origin via last_modified_by.
        # A brand-new memory written this turn is authored by the user input
        # that prompted it (e.g. "user_input_1"); updates, deletes, and
        # consolidations are authored by the trace step that performed them, so
        # last_modified_by points back into the trace for rollback/repair.
        # The memory_* steps follow the final answer in the session chain.
        # used_ids comes from the citation tuple, not from the Memory object.
        for memory, used_ids in created_pairs:
            self._log(
                "memory_write",
                content=memory.content,
                prev_step_ids=[final_step],
                invalidated_memory_ids=[],
                generated_memory_ids=[memory.memory_id],
                used_ids=used_ids,
                turn_id=turn_id,
                user_input_id=user_input_id,
            )
            memory.last_modified_by = user_input_id
        for memory, used_ids in updated_pairs:
            step_id = self._log(
                "memory_update",
                content=memory.content,
                prev_step_ids=[final_step],
                invalidated_memory_ids=[memory.supersedes] if memory.supersedes else [],
                generated_memory_ids=[memory.memory_id],
                used_ids=used_ids,
                turn_id=turn_id,
                user_input_id=user_input_id,
            )
            memory.last_modified_by = step_id
        for memory, used_ids in deleted_pairs:
            step_id = self._log(
                "memory_delete",
                content=f"Deleted memory: {memory.content}",
                prev_step_ids=[final_step],
                invalidated_memory_ids=[memory.memory_id],
                generated_memory_ids=[],
                used_ids=used_ids,
                turn_id=turn_id,
                user_input_id=user_input_id,
            )
            memory.last_modified_by = step_id
        for memory, used_ids in consolidated_pairs:
            step_id = self._log(
                "memory_consolidate",
                content=memory.content,
                prev_step_ids=[final_step],
                invalidated_memory_ids=list(memory.derived_from),
                generated_memory_ids=[memory.memory_id],
                used_ids=used_ids,
                turn_id=turn_id,
                user_input_id=user_input_id,
            )
            memory.last_modified_by = step_id

        # Record this turn so later turns can rely on (and promote) it. Store its
        # real user_input_id (the one minted at the top of this turn) alongside
        # the text, so the SAME id identifies this utterance when it is cited
        # from history on a later turn — no positional "h{n}" rename.
        self.history.append(HistoryTurn(user_input_id=user_input_id, text=prompt))

        return {
            "task_id": task_id,
            "response": response,
            "tool_calls": tool_calls,
            "used_memories": [m.memory_id for m in used_memories],
            "promoted_memories": [
                m.memory_id for m in new_memories if m.source == "user_history"
            ],
            "new_memories": [m.memory_id for m in new_memories],
            "updated_memories": [m.memory_id for m in updated_memories],
            "deleted_memories": [m.memory_id for m in deleted_memories],
            "consolidated_memories": [m.memory_id for m in consolidated_memories],
            "tool_observations": observations,
            "claim": claim,
        }
