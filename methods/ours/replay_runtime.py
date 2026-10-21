"""Small replay-only runtime, independent of :class:`ReactAgent`'s loop."""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Mapping, MutableSet, Optional

from agent.memory_store import MemoryStore
from agent.react_agent import HistoryTurn

from .replay_types import JSONRecord, JSONValue

LLMCallable = Callable[..., str]
ToolCallable = Callable[..., object]


class ReplayRuntime:
    def __init__(
        self,
        *,
        model: Optional[str],
        config: Mapping[str, object],
        tools: Mapping[str, ToolCallable],
        llm: Optional[LLMCallable],
        memory_store: MemoryStore,
        history: List[HistoryTurn],
        executed: MutableSet[str],
        task_id: Optional[str],
        turn_index: int,
    ) -> None:
        self.model = model
        self.config = config
        self.tools = tools
        self.llm = llm
        self.memory_store = memory_store
        self.history = history
        self.executed = executed
        self.task_id = task_id
        self.turn_index = turn_index

    @classmethod
    def from_agent(cls, agent: object) -> "ReplayRuntime":
        return cls(
            model=getattr(agent, "model", None),
            config=getattr(agent, "config", {}),
            tools=getattr(agent, "tools", {}),
            llm=None,
            memory_store=getattr(agent, "memory_store"),
            history=getattr(agent, "history"),
            executed=getattr(agent, "executed"),
            task_id=getattr(agent, "task_id", None),
            turn_index=getattr(agent, "turn_index", 0),
        )

    def active_memories(self) -> List[JSONRecord]:
        return [
            memory.to_dict()
            for memory in self.memory_store.list_memories(self.task_id)
            if memory.status == "active"
        ]

    @staticmethod
    def call_key(tool_name: str, args: Mapping[str, object]) -> str:
        return json.dumps([tool_name, dict(args)], sort_keys=True, default=str)

    def is_read_only_tool(self, tool_name: str) -> bool:
        name = tool_name.lower()
        configured = {
            str(item).lower()
            for item in self.config.get("side_effecting_tools", [])  # type: ignore[union-attr]
        }
        if name in configured:
            return False
        read_only_prefixes = ("search_", "lookup_", "check_", "compare_")
        if name.startswith(read_only_prefixes):
            return True
        verbs = (
            "create", "open", "book", "reserve", "cancel", "file", "raise",
            "place", "submit", "update", "delete", "pay", "refund", "schedule",
        )
        return not any(verb in name for verb in verbs)

    def execute_tool(self, tool_name: str, args: Mapping[str, object]) -> JSONRecord:
        func = self.tools.get(tool_name)
        if not callable(func):
            return {"error": f"unknown tool: {tool_name}"}
        try:
            value = func(**dict(args))
        except TypeError as exc:
            return {"error": f"bad arguments for {tool_name}: {exc}"}
        except Exception as exc:
            return {"error": str(exc)}
        result: JSONRecord
        if isinstance(value, dict):
            result = value  # type: ignore[assignment]
        else:
            result = {"result": value}  # type: ignore[dict-item]
        if "error" not in result and self.is_read_only_tool(tool_name):
            self.executed.add(self.call_key(tool_name, args))
        return result
