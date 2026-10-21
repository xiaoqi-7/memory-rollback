"""Schema validation for the LLM-judge rollback plan baseline."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


METHOD = "llm_judge_repair"

LIST_FIELDS = (
    "delete_memory_ids",
    "quarantine_memory_ids",
    "invalidate_claim_ids",
    "replay_step_ids",
    "preserve_step_ids",
    "redundant_step_ids",
    "suspicious_step_ids",
)

STEP_DISPOSITION_FIELDS = (
    "replay_step_ids",
    "preserve_step_ids",
    "redundant_step_ids",
    "suspicious_step_ids",
)


class PlanValidationError(ValueError):
    """Raised when an LLM plan violates the rollback-plan contract."""


def _require_string_list(raw: Mapping[str, Any], field: str) -> list[str]:
    value = raw.get(field, [])
    if not isinstance(value, list):
        raise PlanValidationError(f"{field} must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise PlanValidationError(f"{field} must contain only strings")
    return list(value)


def _dedupe_in_order(values: Sequence[str], order: Sequence[str]) -> list[str]:
    wanted = set(values)
    return [item for item in order if item in wanted]


def normalize_llm_plan(
    raw: Mapping[str, Any],
    *,
    task_id: str,
    case_id: str,
    faulty_memory_ids: Sequence[str],
    memories: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate and normalize an LLM-generated rollback plan."""
    if not isinstance(raw, Mapping):
        raise PlanValidationError("LLM response must be a JSON object")

    memory_order = [
        memory["memory_id"]
        for memory in memories
        if isinstance(memory.get("memory_id"), str)
    ]
    step_order = [
        step["step_id"] for step in steps if isinstance(step.get("step_id"), str)
    ]
    claim_ids = {
        step["step_id"]
        for step in steps
        if step.get("step_type") == "claim"
        and isinstance(step.get("step_id"), str)
    }
    memory_ids = set(memory_order)
    step_ids = set(step_order)

    values = {field: _require_string_list(raw, field) for field in LIST_FIELDS}

    for field in ("delete_memory_ids", "quarantine_memory_ids"):
        unknown = sorted(set(values[field]) - memory_ids)
        if unknown:
            raise PlanValidationError(
                f"{field} contains unknown memory id(s): {unknown}"
            )

    required_delete_ids = [
        memory_id for memory_id in faulty_memory_ids if memory_id in memory_ids
    ]
    missing_required_delete_ids = sorted(
        set(required_delete_ids) - set(values["delete_memory_ids"])
    )
    if missing_required_delete_ids:
        raise PlanValidationError(
            "delete_memory_ids must include every source faulty memory id; "
            f"missing: {missing_required_delete_ids}"
        )

    for field in STEP_DISPOSITION_FIELDS:
        unknown = sorted(set(values[field]) - step_ids)
        if unknown:
            raise PlanValidationError(
                f"{field} contains unknown step id(s): {unknown}"
            )

    unknown_claims = sorted(set(values["invalidate_claim_ids"]) - claim_ids)
    if unknown_claims:
        raise PlanValidationError(
            "invalidate_claim_ids must contain only claim step ids; invalid: "
            f"{unknown_claims}"
        )

    invalidate_set = set(values["invalidate_claim_ids"])
    replay_or_suspicious = set(values["replay_step_ids"]) | set(
        values["suspicious_step_ids"]
    )
    if not invalidate_set <= replay_or_suspicious:
        missing = sorted(invalidate_set - replay_or_suspicious)
        raise PlanValidationError(
            "invalidate_claim_ids must be a subset of replay_step_ids union "
            f"suspicious_step_ids; missing: {missing}"
        )

    owner_by_step: dict[str, str] = {}
    for field in STEP_DISPOSITION_FIELDS:
        for step_id in values[field]:
            owner = owner_by_step.setdefault(step_id, field)
            if owner != field:
                raise PlanValidationError(
                    f"step id {step_id!r} appears in both {owner} and {field}"
                )

    normalized = {
        "method": METHOD,
        "task_id": task_id,
        "case_id": case_id,
        "faulty_memory_ids": _dedupe_in_order(faulty_memory_ids, list(faulty_memory_ids)),
        "delete_memory_ids": _dedupe_in_order(
            values["delete_memory_ids"], memory_order
        ),
        "quarantine_memory_ids": _dedupe_in_order(
            values["quarantine_memory_ids"], memory_order
        ),
        "invalidate_claim_ids": _dedupe_in_order(
            values["invalidate_claim_ids"], step_order
        ),
        "replay_step_ids": _dedupe_in_order(values["replay_step_ids"], step_order),
        "preserve_step_ids": _dedupe_in_order(
            values["preserve_step_ids"], step_order
        ),
        "redundant_step_ids": _dedupe_in_order(
            values["redundant_step_ids"], step_order
        ),
        "suspicious_step_ids": _dedupe_in_order(
            values["suspicious_step_ids"], step_order
        ),
        "reason": raw.get("reason") if isinstance(raw.get("reason"), str) else "",
    }
    return normalized
