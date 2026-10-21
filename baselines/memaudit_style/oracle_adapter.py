"""Deterministic-oracle adapter for MemAudit counterfactual judgments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from eval.oracle_common import (
    final_answer,
    required_answer_facts,
    visible_fact_present,
)
from eval.run_oracle import DOMAIN_ORACLES


_TASK_ID_RE = re.compile(r"(task_\d+)$")


def _task_id_candidates(task_id: str) -> List[str]:
    candidates = [task_id]
    match = _TASK_ID_RE.search(task_id or "")
    if match:
        candidates.append(match.group(1))
    return list(dict.fromkeys(candidates))


def load_task_record(config: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    data_source = config.get("data_source")
    if not isinstance(data_source, str) or not data_source:
        raise ValueError("agent config is missing data_source")
    path = Path(data_source).resolve()
    tasks_path = path / "tasks_full_session.jsonl"
    candidates = _task_id_candidates(task_id)
    wanted = set(candidates)
    exact_task_id = candidates[0] if candidates else task_id
    matches: List[Dict[str, Any]] = []
    with tasks_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            row_task_id = row.get("task_id")
            row_original_task_id = row.get("original_task_id")
            if row_task_id in wanted or row_original_task_id in wanted:
                matches.append(row)
    if matches:
        def rank(row: Dict[str, Any]) -> tuple[int, int, int]:
            row_task_id = row.get("task_id")
            has_oracle = isinstance(row.get("oracle"), dict)
            return (
                0 if row_task_id == exact_task_id else 1,
                0 if has_oracle else 1,
                0 if row_task_id in wanted else 1,
            )

        return sorted(matches, key=rank)[0]
    raise ValueError(f"task_id {task_id!r} not found in {tasks_path}")


def load_ground_truth_record(
    *,
    config: Dict[str, Any],
    case_id: str | None,
    task_id: str,
) -> Dict[str, Any] | None:
    if not case_id:
        return None
    data_source = config.get("data_source")
    if not isinstance(data_source, str) or not data_source:
        raise ValueError("agent config is missing data_source")
    path = Path(data_source).resolve() / "ground_truth.jsonl"
    if not path.exists():
        return None
    wanted_task_ids = set(_task_id_candidates(task_id))
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("case_id") == case_id:
                return row
            if row.get("task_id") in wanted_task_ids and case_id.endswith(
                "/" + "+".join(row.get("faulty_memory_ids", []) or [])
            ):
                return row
    return None



def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def score_oracle_trace(
    *,
    task: Dict[str, Any],
    trace_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    spec = task.get("oracle")
    if not isinstance(spec, dict):
        return {
            "oracle_score": 0.0,
            "required_fact_hits": 0,
            "required_fact_total": 0,
            "forbidden_fact_absent_count": 0,
            "forbidden_fact_total": 0,
            "required_fact_results": [],
            "forbidden_fact_results": [],
            "failure_reasons": ["task has no oracle spec"],
        }

    answer_text = final_answer(trace_steps)
    required_results = []
    for fact in required_answer_facts(spec):
        present = visible_fact_present(answer_text, fact)
        required_results.append({"fact": fact, "present": present})

    forbidden_results = []
    for fact in _as_list(spec.get("forbidden_answer_facts")):
        present = visible_fact_present(answer_text, fact)
        forbidden_results.append({"fact": fact, "present": present, "absent": not present})

    required_total = len(required_results)
    required_hits = sum(1 for row in required_results if row["present"])
    forbidden_total = len(forbidden_results)
    forbidden_absent = sum(1 for row in forbidden_results if row["absent"])

    components = []
    if required_total:
        components.append(required_hits / required_total)
    if forbidden_total:
        components.append(forbidden_absent / forbidden_total)
    oracle_score = sum(components) / len(components) if components else 1.0

    return {
        "oracle_score": oracle_score,
        "required_fact_hits": required_hits,
        "required_fact_total": required_total,
        "forbidden_fact_absent_count": forbidden_absent,
        "forbidden_fact_total": forbidden_total,
        "required_fact_results": required_results,
        "forbidden_fact_results": forbidden_results,
    }

def judge_recovery(
    *,
    domain: str,
    task: Dict[str, Any],
    memories: List[Dict[str, Any]],
    trace_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if domain not in DOMAIN_ORACLES:
        raise ValueError(f"no deterministic oracle for domain {domain!r}")
    result = DOMAIN_ORACLES[domain](task, memories, trace_steps, "counterfactual")
    payload = result.to_dict()
    payload["label"] = "success" if payload.get("oracle_pass") is True else "failure"
    return payload
