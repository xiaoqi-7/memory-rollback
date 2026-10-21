"""LLM judge that emits a rollback-plan-like JSON object."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from baselines.common_tools.llm_tracker import BaselineLLMTracker
from baselines.llm_judge_repair.plan_schema import (
    PlanValidationError,
    normalize_llm_plan,
)
from baselines.llm_judge_repair.prompts import (
    SYSTEM_PROMPT,
    build_prompt_payload,
    build_user_prompt,
)


def _parse_json_object(text: str) -> Dict[str, Any]:
    if text is None:
        raise ValueError("empty LLM response")
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")
    return data


def generate_llm_judge_plan(
    *,
    task_id: str,
    case_id: str,
    faulty_memory_ids: List[str],
    memories: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    session: List[str],
    model: str | None = None,
) -> Dict[str, Any]:
    tracker = BaselineLLMTracker()
    prompt_payload = build_prompt_payload(
        task_id=task_id,
        case_id=case_id,
        faulty_memory_ids=faulty_memory_ids,
        memories=memories,
        steps=steps,
        session=session,
    )
    attempts: list[dict[str, Any]] = []
    validation_error: str | None = None

    for attempt_no in (1, 2):
        user_prompt = build_user_prompt(prompt_payload, validation_error)
        raw_text = tracker.call(
            "llm_judge_plan",
            SYSTEM_PROMPT,
            user_prompt,
            model=model,
            response_format={"type": "json_object"},
        )
        attempt = {
            "attempt": attempt_no,
            "raw_text": raw_text,
            "validation_error": None,
        }
        try:
            raw_plan = _parse_json_object(raw_text)
            normalized = normalize_llm_plan(
                raw_plan,
                task_id=task_id,
                case_id=case_id,
                faulty_memory_ids=faulty_memory_ids,
                memories=memories,
                steps=steps,
            )
        except (json.JSONDecodeError, PlanValidationError, ValueError) as exc:
            validation_error = str(exc)
            attempt["validation_error"] = validation_error
            attempts.append(attempt)
            continue

        attempts.append(attempt)
        return {
            "repair_failed": False,
            "plan": normalized,
            "prompt_artifact": {
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "input_metadata": {
                    "task_id": task_id,
                    "case_id": case_id,
                    "faulty_memory_ids": faulty_memory_ids,
                    "memory_count": len(memories),
                    "trace_step_count": len(steps),
                },
            },
            "raw_response_artifact": {"attempts": attempts},
            "tracker": tracker,
        }

    return {
        "repair_failed": True,
        "error": validation_error or "LLM judge failed to produce a valid plan",
        "plan": None,
        "prompt_artifact": {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": build_user_prompt(prompt_payload, validation_error),
            "input_metadata": {
                "task_id": task_id,
                "case_id": case_id,
                "faulty_memory_ids": faulty_memory_ids,
                "memory_count": len(memories),
                "trace_step_count": len(steps),
            },
        },
        "raw_response_artifact": {"attempts": attempts},
        "tracker": tracker,
    }
