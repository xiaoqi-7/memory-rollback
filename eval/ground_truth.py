"""Ground-truth compatibility helpers for eval.

The current benchmark data uses ``task_id`` as the stable identity and keeps a
compact legacy schema.  Older eval code expected a normalized schema keyed by
``case_id``.  These helpers normalize both shapes into one internal row format
without rewriting the dataset on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_jsonl(path: Path, label: str = "JSONL") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid {label} JSON in {path} line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path} line {line_number}: {label} row must be an object"
                    )
                rows.append(value)
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    return rows


def row_task_id(row: Mapping[str, Any], path: Path | None = None) -> str:
    """Return the eval identity, preferring task_id over legacy case_id."""
    value = row.get("task_id") or row.get("case_id")
    if not isinstance(value, str) or not value:
        where = f" in {path}" if path is not None else ""
        raise ValueError(f"ground truth row{where} needs a non-empty task_id")
    return value


def result_task_id(case: Mapping[str, Any]) -> str:
    """Return the result identity, preferring case.task_id over case.case_id."""
    value = case.get("task_id") or case.get("case_id")
    if not isinstance(value, str) or not value:
        raise ValueError("result.case needs a non-empty task_id")
    return value


def memory_ids(value: Any, field: str) -> list[str]:
    """Accept either legacy list[str] or normalized list[{memory_id: ...}]."""
    if not isinstance(value, list):
        raise ValueError(f"memory_ground_truth.{field} must be a list")
    ids: list[str] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            memory_id = item
        elif isinstance(item, dict):
            memory_id = item.get("memory_id")
        else:
            raise ValueError(
                f"memory_ground_truth.{field}[{index}] must be a string or object"
            )
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError(
                f"memory_ground_truth.{field}[{index}].memory_id must be non-empty"
            )
        ids.append(memory_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"memory_ground_truth.{field} contains duplicate memory IDs")
    return ids


def affected_step_ids(row: Mapping[str, Any]) -> list[str]:
    """Return affected step IDs from either normalized or legacy fields."""
    if isinstance(row.get("trace_ground_truth"), dict):
        affected = row["trace_ground_truth"].get("affected_steps")
        if not isinstance(affected, list):
            raise ValueError("trace_ground_truth.affected_steps must be a list")
        ids: list[str] = []
        for index, step in enumerate(affected):
            if isinstance(step, str):
                step_id = step
            elif isinstance(step, dict):
                step_id = step.get("faulty_step_id") or step.get("step_id")
            else:
                raise ValueError(
                    f"trace_ground_truth.affected_steps[{index}] must be a string or object"
                )
            if not isinstance(step_id, str) or not step_id:
                raise ValueError(
                    f"trace_ground_truth.affected_steps[{index}] step id must be non-empty"
                )
            ids.append(step_id)
        return ids

    affected = row.get("affected_steps")
    if not isinstance(affected, list) or any(not isinstance(item, str) for item in affected):
        raise ValueError("affected_steps must be a list of strings")
    if len(affected) != len(set(affected)):
        raise ValueError("affected_steps contains duplicate step IDs")
    return list(affected)


def normalize_ground_truth_row(row: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Normalize one ground-truth row while preserving original fields."""
    normalized = dict(row)
    task_id = row_task_id(normalized, path)
    normalized["task_id"] = task_id
    normalized["case_id"] = task_id

    memory_truth = normalized.get("memory_ground_truth")
    if not isinstance(memory_truth, dict):
        raise ValueError(f"ground truth {task_id} needs memory_ground_truth")
    memory_truth = dict(memory_truth)
    for field in (
        "source_faulty_memories",
        "derived_faulty_memories",
        "benign_active_memory_ids",
    ):
        memory_truth[field] = memory_ids(memory_truth.get(field), field)
    normalized["memory_ground_truth"] = memory_truth

    # Validate either schema without inventing a step type. Controlled rows
    # only carry affected step IDs, whose types must be recovered from the
    # referenced faulty trace. LME rows already carry authoritative types.
    affected_step_ids(normalized)
    trace_truth = normalized.get("trace_ground_truth")
    if isinstance(trace_truth, dict):
        normalized["trace_ground_truth"] = dict(trace_truth)
    return normalized


def load_ground_truth_index(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path, "ground truth"):
        normalized = normalize_ground_truth_row(row, path)
        task_id = normalized["task_id"]
        if task_id in indexed:
            raise ValueError(f"duplicate ground truth task_id {task_id!r} in {path}")
        indexed[task_id] = normalized
    return indexed


def load_task_index(data_root: Path, domain: str) -> dict[str, dict[str, Any]]:
    tasks_path = data_root / domain / "tasks_full_session.jsonl"
    indexed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(tasks_path, "task"):
        task_id = row.get("task_id")
        if isinstance(task_id, str) and task_id:
            indexed[task_id] = row
    return indexed


def fresh_probe_input(task: Mapping[str, Any], task_id: str) -> str:
    session = task.get("session")
    if not isinstance(session, list) or not session:
        raise ValueError(f"task {task_id} has no non-empty session for recurrence probe")
    probe = session[-1]
    if not isinstance(probe, str) or not probe.strip():
        raise ValueError(f"task {task_id} session[-1] is not a non-empty user input")
    return probe.strip()


def expected_output_text(row: Mapping[str, Any]) -> str:
    expected = row.get("expected_output")
    if isinstance(expected, str):
        return expected
    if isinstance(expected, dict):
        value = expected.get("clean_final_answer")
        if isinstance(value, str):
            return value
    return ""
