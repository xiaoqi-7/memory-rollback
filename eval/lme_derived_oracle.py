"""Deterministic oracle for LME-derived tasks.

This mirrors the oracle shipped with ``lme-v2-adapter`` for the LME-derived
dataset.  LME tasks carry an embedded ``task["oracle"]`` spec whose evaluator
can be phrase-set based or multiple-choice based; the generic controlled
benchmark oracle is too coarse for these answer formats.
"""

from __future__ import annotations

import re
import string
from typing import Any

from eval.oracle_common import (
    OracleResult,
    final_answer,
    tool_actions,
    used_id_set,
    visible_fact_present,
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _boxed_or_text(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1] if matches else text


def _normalize(text: Any) -> str:
    value = str(text or "").lower()
    value = value.replace("-", " ")
    value = value.translate(str.maketrans("", "", string.punctuation.replace(";", "")))
    return re.sub(r"\s+", " ", value).strip()


def _answer_terms(answer: Any) -> list[str]:
    if isinstance(answer, list):
        return [str(item) for item in answer]
    return [
        part.strip()
        for part in re.split(r"[;,]", str(answer or ""))
        if part.strip()
    ]


def _letters(text: str) -> set[str]:
    target = _boxed_or_text(text).upper()
    return set(re.findall(r"\b[A-Z]\b", target))


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


def _memory_contains_fact(memory: dict[str, Any], value: Any) -> bool:
    return visible_fact_present(str(memory.get("content", "")), value)


def _requires_ordered_match(oracle: dict[str, Any]) -> bool:
    evaluator = str(oracle.get("evaluator") or oracle.get("eval_function") or "")
    return bool(oracle.get("requires_order")) or evaluator.startswith(
        "norm_phrase_set_match_ordered"
    )


def _ordered_phrase_set_match(answer_text: str, expected: Any) -> bool:
    target = _normalize(_boxed_or_text(answer_text))
    position = 0
    for term in _answer_terms(expected):
        normalized = _normalize(term)
        found = target.find(normalized, position)
        if found < 0:
            return False
        position = found + len(normalized)
    return True


def _lme_answer_pass(answer_text: str, oracle: dict[str, Any]) -> bool:
    evaluator = str(oracle.get("evaluator") or oracle.get("eval_function") or "")
    expected = oracle.get("answer")
    target = _normalize(_boxed_or_text(answer_text))

    if _requires_ordered_match(oracle):
        return _ordered_phrase_set_match(answer_text, expected)

    if evaluator.startswith("norm_phrase_set_match"):
        return all(_normalize(term) in target for term in _answer_terms(expected))

    if evaluator.startswith("mc_choice_set_match"):
        expected_letters = set(re.findall(r"[A-Z]", str(expected or "").upper()))
        return bool(expected_letters) and _letters(answer_text) == expected_letters

    if evaluator.startswith("mc_choice_match"):
        expected_letter = str(expected or "").strip().upper()
        return bool(expected_letter) and expected_letter in _letters(answer_text)

    return False


def evaluate(
    task: dict[str, Any],
    memories: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    run_type: str,
) -> OracleResult:
    task_id = str(task.get("task_id", ""))
    domain = str(task.get("domain", ""))
    oracle = task.get("oracle")
    result = OracleResult(task_id=task_id, domain=domain, run_type=run_type, oracle_pass=False)
    if not isinstance(oracle, dict):
        result.failure_reasons.append("task has no oracle spec")
        result.tool_call_pass = False
        result.tool_arg_pass = False
        result.answer_fact_pass = False
        return result

    actions = tool_actions(traces)
    answer_text = final_answer(traces)
    used_ids = used_id_set(traces)

    for expected in _as_list(oracle.get("expected_tool_calls")):
        if not any(_match_expected_tool_call(action, expected) for action in actions):
            result.tool_call_pass = False
            result.tool_arg_pass = False
            result.failure_reasons.append(f"missing expected tool call: {expected!r}")

    evaluator = str(oracle.get("evaluator") or "")
    if evaluator.startswith("llm_") or evaluator == "unsupported":
        result.answer_fact_pass = False
        result.failure_reasons.append(f"unsupported evaluator: {evaluator}")
    elif not _lme_answer_pass(answer_text, oracle):
        result.answer_fact_pass = False
        result.failure_reasons.append(
            f"LME evaluator mismatch: expected {oracle.get('answer')!r} with {evaluator}"
        )

    for fact in _as_list(oracle.get("forbidden_answer_facts")):
        if visible_fact_present(answer_text, fact):
            result.forbidden_fact_absent = False
            result.failure_reasons.append(f"forbidden answer fact visible: {fact!r}")

    required_roles = set(str(role) for role in _as_list(oracle.get("required_memory_roles")))
    if required_roles:
        used_roles = {
            str(memory.get("oracle_role"))
            for memory in memories
            if memory.get("memory_id") in used_ids and memory.get("oracle_role") is not None
        }
        for dependency in _as_list(oracle.get("memory_dependencies")):
            if not isinstance(dependency, dict):
                continue
            role = dependency.get("role")
            if role is None:
                continue
            if any(
                memory.get("memory_id") in used_ids
                and _memory_contains_fact(memory, dependency.get("fact_value"))
                for memory in memories
            ):
                used_roles.add(str(role))
        missing_roles = sorted(required_roles - used_roles)
        if missing_roles:
            result.used_required_memory = False
            result.failure_reasons.append(f"required memory roles not used: {missing_roles}")

    required_memory_ids = set(str(mid) for mid in _as_list(oracle.get("required_memory_ids")))
    missing_memory_ids = sorted(required_memory_ids - used_ids)
    if missing_memory_ids:
        result.used_required_memory = False
        result.failure_reasons.append(f"required memory ids not used: {missing_memory_ids}")

    fault_memory_ids = set(str(mid) for mid in _as_list(oracle.get("fault_memory_ids")))
    if run_type == "faulty" and fault_memory_ids and not (fault_memory_ids & used_ids):
        result.fault_is_retrieved_or_used = False
        result.failure_reasons.append("fault memory was not retrieved or used")

    result.oracle_pass = (
        result.tool_call_pass
        and result.tool_arg_pass
        and result.answer_fact_pass
        and result.forbidden_fact_absent
        and result.used_required_memory
        and result.fault_is_retrieved_or_used
    )
    return result
