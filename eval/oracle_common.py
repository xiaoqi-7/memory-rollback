"""Shared deterministic-oracle utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


@dataclass
class OracleResult:
    task_id: str
    domain: str
    run_type: str
    oracle_pass: bool
    tool_call_pass: bool = True
    tool_arg_pass: bool = True
    answer_fact_pass: bool = True
    forbidden_fact_absent: bool = True
    used_required_memory: bool = True
    fault_is_retrieved_or_used: bool = True
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_observation_content(content: str) -> Any:
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None


def trace_steps_of_type(
    trace_rows: Iterable[dict[str, Any]], step_type: str
) -> list[dict[str, Any]]:
    return [row for row in trace_rows if row.get("step_type") == step_type]


def final_answer(trace_rows: Iterable[dict[str, Any]]) -> str:
    finals = trace_steps_of_type(trace_rows, "final_answer")
    return str(finals[-1].get("content", "")) if finals else ""


def all_final_answers(trace_rows: Iterable[dict[str, Any]]) -> str:
    finals = trace_steps_of_type(trace_rows, "final_answer")
    return "\n".join(str(row.get("content", "")) for row in finals)


def used_id_set(trace_rows: Iterable[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in trace_rows:
        out.update(str(item) for item in row.get("used_ids", []) or [])
        out.update(str(item) for item in row.get("generated_memory_ids", []) or [])
        out.update(str(item) for item in row.get("invalidated_memory_ids", []) or [])
    return out


def visible_fact_present(text: str, value: Any) -> bool:
    normalized_text = text.lower().replace(",", "")
    if value is None:
        return True
    if isinstance(value, (list, tuple, set)):
        return all(visible_fact_present(text, item) for item in value)
    if isinstance(value, (int, float)):
        candidates = {str(value)}
        if float(value).is_integer():
            candidates.add(str(int(value)))
        candidates.add(f"{float(value):.2f}")
        return any(candidate.lower().replace(",", "") in normalized_text for candidate in candidates)
    normalized_value = str(value).lower().replace(",", "")
    if len(normalized_value.strip()) <= 2:
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_value)}(?![a-z0-9])"
        return re.search(pattern, normalized_text) is not None
    return normalized_value in normalized_text


def _memory_contains_fact(memory: dict[str, Any], value: Any) -> bool:
    return visible_fact_present(str(memory.get("content", "")), value)


def tool_actions(trace_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return trace_steps_of_type(trace_rows, "tool_action")


def tool_observations(trace_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    observations = []
    for row in trace_steps_of_type(trace_rows, "tool_observation"):
        copied = dict(row)
        copied["parsed_content"] = parse_observation_content(str(row.get("content", "")))
        observations.append(copied)
    return observations


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _fact_identity(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def required_answer_facts(spec: dict[str, Any]) -> list[Any]:
    """Return answer facts that must be visible in the final answer."""
    required_final = _as_list(spec.get("required_final_facts"))
    if required_final:
        return required_final

    expected = _as_list(spec.get("expected_answer_facts"))
    required: list[Any] = []
    seen: set[str] = set()
    for group_name in ("memory_dependencies", "fault_sensitive_facts"):
        for item in _as_list(spec.get(group_name)):
            if not isinstance(item, dict) or "fact_value" not in item:
                continue
            value = item.get("fact_value")
            identity = _fact_identity(value)
            if identity in seen:
                continue
            seen.add(identity)
            required.append(value)
    return required or expected


def _match_expected_tool_call(action: dict[str, Any], expected: Any) -> bool:
    if isinstance(expected, str):
        return action.get("tool_name") == expected
    if not isinstance(expected, dict):
        return False
    if expected.get("tool_name") and action.get("tool_name") != expected["tool_name"]:
        return False
    expected_args = expected.get("args") or expected.get("tool_args") or {}
    actual_args = action.get("tool_args") or {}
    for key, value in expected_args.items():
        if actual_args.get(key) != value:
            return False
    return True


def evaluate_oracle_spec(
    task: dict[str, Any],
    memories: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    run_type: str,
) -> OracleResult:
    """Evaluate only the final answer content against task oracle facts."""
    task_id = str(task.get("task_id", ""))
    domain = str(task.get("domain", ""))
    spec = task.get("oracle")
    result = OracleResult(task_id=task_id, domain=domain, run_type=run_type, oracle_pass=False)
    if not isinstance(spec, dict):
        result.failure_reasons.append("task has no oracle spec")
        result.answer_fact_pass = False
        return result

    answer_text = final_answer(traces)

    for fact in required_answer_facts(spec):
        if not visible_fact_present(answer_text, fact):
            result.answer_fact_pass = False
            result.failure_reasons.append(f"expected answer fact not visible: {fact!r}")

    for fact in _as_list(spec.get("forbidden_answer_facts")):
        if visible_fact_present(answer_text, fact):
            result.forbidden_fact_absent = False
            result.failure_reasons.append(f"forbidden answer fact visible: {fact!r}")

    result.oracle_pass = (
        result.answer_fact_pass and result.forbidden_fact_absent
    )
    return result
