"""Deterministic oracle for travel tasks."""

from __future__ import annotations

from typing import Any

from eval.oracle_common import OracleResult, evaluate_oracle_spec


def evaluate(
    task: dict[str, Any],
    memories: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    run_type: str,
) -> OracleResult:
    return evaluate_oracle_spec(task, memories, traces, run_type)
