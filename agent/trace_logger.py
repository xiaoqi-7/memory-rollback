'''
log trace
to given file path  like
{
  "step_id": "s_003",
  "task_id": "task_001",
  "step_type": "memory_read | claim | plan | tool_action | tool_observation | final_answer | memory_write",
  "content": "The user prefers luxury hotels and has no budget limit.",
  "prev_step_ids": ["s_002"],
  "next_step_ids": ["s_004"],
  "invalidated_memory_ids": ["m_001"],
  "generated_memory_ids": ["m_002"],
  "used_ids": ["m_f001", "s_002", "user_input_2"],
  "tool_name": null,
  "tool_args": null,
  "timestamp": "2026-01-01T10:05:00"
}

Step sequencing and memory lineage are kept in separate fields:

`prev_step_ids` / `next_step_ids` are PURE step-chain links: the trace step ids
("s_*") this step directly follows from, and the trace step ids that directly
follow from it. They never carry memory ids. Together they form one
bidirectional graph over steps that can be traversed in either direction for
rollback/repair.

`invalidated_memory_ids` / `generated_memory_ids` carry memory lineage on
memory-action steps: the memory ids a step invalidated (superseded by an update,
deleted, or merged away by a consolidation) and the memory ids it produced.

`used_ids` expresses *semantic reliance* distinctly from the structural links
above: a flat list of the mixed-type ids this step actually relied on — memory
ids ("m_*"), the claim/tool_observation step ids ("s_*") it drew on, and
prior-user-turn ids ("user_input_{n}") for turns not yet promoted into a memory.
Each prior turn keeps the SAME user_input id it carried while it was the current
turn, so a cross-turn citation lines up with that turn's own identity (and with
a memory's last_modified_by) rather than a position-derived alias.
'''


from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TraceStep:
    """Structured trace step representing a single agent action or
    observation.
    """

    step_id: str
    task_id: Optional[str]
    step_type: str
    content: str
    prev_step_ids: List[str]
    next_step_ids: List[str]
    invalidated_memory_ids: List[str]
    generated_memory_ids: List[str]
    used_ids: List[str]
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    timestamp: Optional[datetime]
    turn_id: Optional[str] = None
    user_input_id: Optional[str] = None
    semantic_slots: Optional[Dict[str, Any]] = None
    oracle_refs: Optional[List[str]] = None
    failure_label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TraceStep to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert datetime objects to ISO format strings
        if result.get("timestamp"):
            result["timestamp"] = result["timestamp"].isoformat()
        return result


class TraceLogger:
    """Abstract trace logger used to record and persist trace steps."""
    def __init__(self) -> None:
        """Initialize the trace logger."""
        self.trace_steps: List[TraceStep] = []
        self.newest_trace_steps_id: int = 0

    def _extract_trace_step_number(self, step_id: str) -> int:
        """Extract the numeric part from step_id like 's_003'."""
        try:
            if step_id.startswith("s_"):
                return int(step_id[2:])
            return 0
        except (ValueError, IndexError):
            return 0

    def _generate_step_id(self) -> str:
        """Generate the next step_id in sequence."""
        self.newest_trace_steps_id += 1
        return f"s_{self.newest_trace_steps_id:03d}"

    def log_step(self, step: TraceStep) -> None:
        """Record a trace step, maintaining the bidirectional step graph.

        Any predecessor step ids listed in ``step.prev_step_ids`` get this new
        step's id appended to their ``next_step_ids``, so the forward links stay
        consistent with the backward links the caller supplies.
        """
        step.step_id = self._generate_step_id()
        if step.timestamp is None:
            step.timestamp = datetime.now()

        self.trace_steps.append(step)

        by_id = {s.step_id: s for s in self.trace_steps}
        for prev_id in step.prev_step_ids:
            if prev_id.startswith("s_"):
                prev = by_id.get(prev_id)
                if prev is not None and step.step_id not in prev.next_step_ids:
                    prev.next_step_ids.append(step.step_id)

    def set_next_step_ids(self, step_id: str, next_step_ids: List[str]) -> None:
        """Replace a previously-logged step's ``next_step_ids``.

        Used when a step's successors are only known after it was logged (e.g. a
        plan step whose spawned tool_action children are created in a later
        inner loop). Only step ids ("s_*") are kept.
        """
        for step in self.trace_steps:
            if step.step_id == step_id:
                step.next_step_ids = [
                    sid for sid in (next_step_ids or []) if str(sid).startswith("s_")
                ]
                return
        raise KeyError(f"Trace step not found: {step_id}")

    # def save(self, path: str) -> None:
    #     """Persist the recorded trace to `path`."""
    #     raise NotImplementedError()

    def load(self, traces: List[Dict[str, Any]]) -> List[TraceStep]:
        """Load trace steps from a list of dictionaries and return them."""
        loaded_steps: List[TraceStep] = []

        for trace in traces:
            timestamp = trace.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            step = TraceStep(
                step_id=trace.get("step_id", ""),
                task_id=trace.get("task_id"),
                step_type=trace.get("step_type", ""),
                content=trace.get("content", ""),
                prev_step_ids=trace.get("prev_step_ids", []) or [],
                next_step_ids=trace.get("next_step_ids", []) or [],
                invalidated_memory_ids=trace.get("invalidated_memory_ids", []) or [],
                generated_memory_ids=trace.get("generated_memory_ids", []) or [],
                used_ids=trace.get("used_ids", []) or [],
                tool_name=trace.get("tool_name"),
                tool_args=trace.get("tool_args"),
                timestamp=timestamp,
                turn_id=trace.get("turn_id"),
                user_input_id=trace.get("user_input_id"),
                semantic_slots=trace.get("semantic_slots"),
                oracle_refs=trace.get("oracle_refs"),
                failure_label=trace.get("failure_label"),
            )

            num = self._extract_trace_step_number(step.step_id)
            if num > self.newest_trace_steps_id:
                self.newest_trace_steps_id = num

            loaded_steps.append(step)

        self.trace_steps = loaded_steps
        return loaded_steps

    def get_trace_steps(self) -> List[Dict[str, Any]]:
        return [step.to_dict() for step in self.trace_steps]
