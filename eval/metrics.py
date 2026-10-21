"""Deterministic evaluation for repaired-result artifacts.

This module intentionally performs no model calls and does not inspect or mutate
the repair pipeline.  It only compares ``repaired_result.json`` files with the
controlled ground-truth JSONL.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.ground_truth import (
    affected_step_ids,
    load_ground_truth_index,
    memory_ids as ground_truth_memory_ids,
    result_task_id,
)


COUNT_FIELDS = (
    "replayed_target_step_count",
    "total_original_steps",
    "replayed_tool_call_count",
    "synthetic_tool_call_count",
    "total_repair_tool_call_count",
    "total_original_tool_call_count",
    "total_llm_call_count",
)
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")
RATIO_FIELDS = (
    "replayed_step_ratio",
    "replayed_original_tool_call_ratio",
    "total_repair_tool_call_ratio",
)
RECURRENCE_MODES = ("fresh_memory",)
RECOVERY_LABELS = ("success", "failure", "uncertain")
RECURRENCE_LABELS = ("no_recurrence", "recurrence", "uncertain")
HumanJudgmentKey = Tuple[str, str, str, Optional[str]]


class EvaluationError(ValueError):
    """Raised when an evaluation input is malformed or inconsistent."""


def _location(source: Optional[Path], case_id: Optional[str] = None) -> str:
    parts = []
    if source is not None:
        parts.append(str(source))
    if case_id:
        parts.append("case_id={!r}".format(case_id))
    return " ({}): ".format(", ".join(parts)) if parts else ": "


def _require_mapping(value: Any, name: str, source: Optional[Path],
                     case_id: Optional[str] = None) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(
            "{}{}must be an object".format(name, _location(source, case_id))
        )
    return value


def _require_string(value: Any, name: str, source: Optional[Path],
                    case_id: Optional[str] = None) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(
            "{}{}must be a non-empty string".format(
                name, _location(source, case_id)
            )
        )
    return value


def _require_id_list(value: Any, name: str, source: Optional[Path],
                     case_id: Optional[str] = None) -> List[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise EvaluationError(
            "{}{}must be a list of strings".format(
                name, _location(source, case_id)
            )
        )
    if len(value) != len(set(value)):
        raise EvaluationError(
            "{}{}contains duplicate IDs".format(name, _location(source, case_id))
        )
    return value


def _require_count(value: Any, name: str, source: Optional[Path],
                   case_id: Optional[str] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(
            "{}{}must be a non-negative integer".format(
                name, _location(source, case_id)
            )
        )
    return value


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def load_human_judgments(
        path: Path) -> Dict[HumanJudgmentKey, Mapping[str, Any]]:
    """Load and validate human judgments, rejecting every duplicate key."""
    path = Path(path)
    judgments = {}  # type: Dict[HumanJudgmentKey, Mapping[str, Any]]
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise EvaluationError("cannot read human judgments {}: {}".format(path, exc))
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            location = "{} line {}".format(path, line_number)
            try:
                judgment = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    "invalid human judgment JSONL in {}: {}".format(
                        location, exc.msg
                    )
                )
            if not isinstance(judgment, dict):
                raise EvaluationError(
                    "human judgment in {} must be an object".format(location)
                )
            method = judgment.get("method")
            if (not isinstance(method, str) or not method.strip()):
                raise EvaluationError(
                    "human judgment method in {} must be a non-empty string".format(
                        location
                    )
                )
            case_id = judgment.get("case_id")
            if (not isinstance(case_id, str) or not case_id.strip()):
                raise EvaluationError(
                    "human judgment case_id in {} must be a non-empty string".format(
                        location
                    )
                )
            evaluation = judgment.get("evaluation")
            if evaluation not in ("recovery", "recurrence"):
                raise EvaluationError(
                    "human judgment evaluation in {} must be recovery or recurrence".format(
                        location
                    )
                )

            if evaluation == "recovery":
                if "mode" in judgment:
                    raise EvaluationError(
                        "recovery judgment in {} must not contain mode".format(location)
                    )
                label = judgment.get("label")
                if label not in RECOVERY_LABELS:
                    raise EvaluationError(
                        "invalid recovery label in {}: {!r}".format(location, label)
                    )
                key = (method, case_id, evaluation, None)
            else:
                if "mode" not in judgment:
                    raise EvaluationError(
                        "recurrence judgment in {} requires mode".format(location)
                    )
                mode = judgment.get("mode")
                if mode not in RECURRENCE_MODES:
                    raise EvaluationError(
                        "invalid recurrence mode in {}: {!r}".format(location, mode)
                    )
                label = judgment.get("label")
                if label not in RECURRENCE_LABELS:
                    raise EvaluationError(
                        "invalid recurrence label in {}: {!r}".format(location, label)
                    )
                key = (method, case_id, evaluation, mode)
            if key in judgments:
                kind = "recovery key" if evaluation == "recovery" else "recurrence key"
                raise EvaluationError(
                    "duplicate {} in {}: {!r}".format(kind, location, key)
                )
            judgments[key] = judgment
    return judgments


def attach_human_metrics(
        cases: Sequence[Dict[str, Any]],
        judgments: Mapping[HumanJudgmentKey, Mapping[str, Any]]) -> None:
    """Attach nullable human metric values to every evaluated case."""
    recovery_values = {"success": 1.0, "failure": 0.0, "uncertain": None}
    recurrence_values = {
        "recurrence": 1.0, "no_recurrence": 0.0, "uncertain": None,
    }
    for case in cases:
        method = case["method"]
        case_id = case["case_id"]
        recovery = judgments.get((method, case_id, "recovery", None))
        recovery_label = recovery.get("label") if recovery is not None else None
        case["recovery_success"] = {
            "label": recovery_label,
            "value": recovery_values.get(recovery_label),
        }
        recurrence = {}
        for mode in RECURRENCE_MODES:
            judgment = judgments.get((method, case_id, "recurrence", mode))
            label = judgment.get("label") if judgment is not None else None
            recurrence[mode] = {
                "label": label,
                "value": recurrence_values.get(label),
            }
        case["recurrence"] = recurrence


def aggregate_human_metrics(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate labels, conditioning recurrence on successful recovery.

    A recurrence probe answers whether a repaired state stays repaired.  Cases
    that did not recover in the first place therefore do not enter recurrence
    counts or denominators.  Uncertain and missing labels among successfully
    recovered cases still do not enter the rate denominator.
    """
    recovery_counts = {label: 0 for label in RECOVERY_LABELS}
    recovery_counts["missing"] = 0
    for case in cases:
        label = case["recovery_success"]["label"]
        recovery_counts[label if label is not None else "missing"] += 1
    recovery_denominator = recovery_counts["success"] + recovery_counts["failure"]
    recovery = {
        "numerator": recovery_counts["success"],
        "denominator": recovery_denominator,
        "value": _ratio(recovery_counts["success"], recovery_denominator),
    }
    recovery.update(recovery_counts)

    recurrence_rates = {}
    recovered_cases = [
        case for case in cases
        if case["recovery_success"]["label"] == "success"
    ]
    for mode in RECURRENCE_MODES:
        counts = {label: 0 for label in RECURRENCE_LABELS}
        counts["missing"] = 0
        for case in recovered_cases:
            label = case["recurrence"][mode]["label"]
            counts[label if label is not None else "missing"] += 1
        denominator = counts["recurrence"] + counts["no_recurrence"]
        metric = {
            "numerator": counts["recurrence"],
            "denominator": denominator,
            "value": _ratio(counts["recurrence"], denominator),
            "eligible_recovery_success_cases": len(recovered_cases),
            "excluded_non_success_cases": len(cases) - len(recovered_cases),
        }
        metric.update(counts)
        recurrence_rates[mode] = metric
    return {
        "recovery_success_rate": recovery,
        "recurrence_rate": recurrence_rates,
    }


def _removal_metric(expected_ids: Iterable[str], active_ids: Set[str],
                    positive_name: str) -> Dict[str, Any]:
    expected = set(expected_ids)
    positive = sorted(expected - active_ids)
    missed = sorted(expected & active_ids)
    denominator = len(expected)
    return {
        "numerator": len(positive),
        "denominator": denominator,
        "value": _ratio(len(positive), denominator),
        positive_name: positive,
        "missed_ids": missed,
    }


def _repair_removal_metric(expected_ids: Iterable[str], deactivated_ids: Set[str],
                           positive_name: str) -> Dict[str, Any]:
    """Credit only removals caused by the repair, not pre-existing inactivity."""
    expected = set(expected_ids)
    positive = sorted(expected & deactivated_ids)
    missed = sorted(expected - deactivated_ids)
    denominator = len(expected)
    return {
        "numerator": len(positive),
        "denominator": denominator,
        "value": _ratio(len(positive), denominator),
        positive_name: positive,
        "missed_ids": missed,
    }


def _lineage_reaches_active(memory_id: str, remap: Mapping[str, str],
                            active_ids: Set[str]) -> bool:
    """Return whether an ID or any remapped successor is active.

    ``visited`` makes malformed cyclic lineage safe.  Active membership is
    checked before cycle detection so an active ID on a cycle is preserved.
    """
    current = memory_id
    visited = set()  # type: Set[str]
    while True:
        if current in active_ids:
            return True
        if current in visited:
            return False
        visited.add(current)
        if current not in remap:
            return False
        current = remap[current]


def _preservation_metric(benign_ids: Iterable[str], active_ids: Set[str],
                         remap: Mapping[str, str]) -> Dict[str, Any]:
    benign = set(benign_ids)
    preserved = sorted(
        memory_id for memory_id in benign
        if _lineage_reaches_active(memory_id, remap, active_ids)
    )
    lost = sorted(benign - set(preserved))
    denominator = len(benign)
    return {
        "numerator": len(preserved),
        "denominator": denominator,
        "value": _ratio(len(preserved), denominator),
        "preserved_ids": preserved,
        "lost_ids": lost,
    }


def _repair_preservation_metric(benign_ids: Iterable[str], original_active_ids: Set[str],
                                active_ids: Set[str], remap: Mapping[str, str],
                                harmed_ids: Set[str]) -> Dict[str, Any]:
    """Measure whether repair harmed benign memories.

    The denominator remains all benign memory IDs from ground truth.  For
    memories active at repair start, preservation means the memory or its
    remapped successor is active after repair.  For memories already inactive at
    repair start, preservation means repair did not target them for deletion or
    quarantine; they should not count as lost merely because they were historical.
    """
    benign = set(benign_ids)
    preserved = []
    for memory_id in benign:
        if memory_id in original_active_ids:
            if _lineage_reaches_active(memory_id, remap, active_ids):
                preserved.append(memory_id)
        elif memory_id not in harmed_ids:
            preserved.append(memory_id)
    preserved = sorted(preserved)
    lost = sorted(benign - set(preserved))
    denominator = len(benign)
    return {
        "numerator": len(preserved),
        "denominator": denominator,
        "value": _ratio(len(preserved), denominator),
        "preserved_ids": preserved,
        "lost_ids": lost,
    }


def _claim_metric(true_ids: Iterable[str], predicted_ids: Iterable[str]) -> Dict[str, Any]:
    truth = set(true_ids)
    predicted = set(predicted_ids)
    true_positive = sorted(truth & predicted)
    false_positive = sorted(predicted - truth)
    false_negative = sorted(truth - predicted)
    tp, fp, fn = len(true_positive), len(false_positive), len(false_negative)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "true_positive_ids": true_positive,
        "false_positive_ids": false_positive,
        "false_negative_ids": false_negative,
    }


def _affected_step_records(row: Mapping[str, Any]) -> List[Any]:
    affected = row.get("affected_steps")
    if isinstance(affected, list):
        return list(affected)
    trace_truth = row.get("trace_ground_truth")
    if isinstance(trace_truth, dict) and isinstance(trace_truth.get("affected_steps"), list):
        return list(trace_truth["affected_steps"])
    return []


def _step_id_from_affected_record(record: Any) -> Optional[str]:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        step_id = record.get("faulty_step_id") or record.get("step_id")
        return step_id if isinstance(step_id, str) and step_id else None
    return None


def _load_controlled_faulty_step_types(
    *,
    case: Mapping[str, Any],
    ground_truth_path: Optional[Path],
) -> Dict[str, str]:
    """Load original faulty step types for legacy controlled ground truth rows."""
    if ground_truth_path is None:
        return {}
    registry_path = Path(ground_truth_path).resolve().parent.parent / "registry.jsonl"
    if not registry_path.is_file():
        return {}
    domain = case.get("domain")
    task_id = case.get("task_id")
    if not isinstance(domain, str) or not isinstance(task_id, str):
        return {}
    fault_types = case.get("fault_types")
    if not isinstance(fault_types, list):
        fault_types = []
    wanted_fault_types = tuple(sorted(item for item in fault_types if isinstance(item, str)))
    registry_rows = []
    try:
        with registry_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("domain") != domain or row.get("task_id") != task_id:
                    continue
                registry_rows.append(row)
    except (OSError, json.JSONDecodeError):
        return {}
    if not registry_rows:
        return {}
    registry_row = registry_rows[0]
    if len(registry_rows) > 1:
        for row in registry_rows:
            row_fault_types = row.get("fault_types")
            if not isinstance(row_fault_types, list):
                row_fault_types = []
            if tuple(sorted(item for item in row_fault_types if isinstance(item, str))) == wanted_fault_types:
                registry_row = row
                break
    paths = registry_row.get("paths")
    if not isinstance(paths, dict) or not isinstance(paths.get("faulty_tracesteps"), str):
        return {}
    trace_path = registry_path.parent / paths["faulty_tracesteps"]
    if not trace_path.is_file():
        return {}
    step_types = {}  # type: Dict[str, str]
    try:
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("task_id") != task_id:
                    continue
                step_id = row.get("step_id")
                step_type = row.get("step_type")
                if isinstance(step_id, str) and isinstance(step_type, str):
                    step_types[step_id] = step_type
    except (OSError, json.JSONDecodeError):
        return {}
    return step_types


def _affected_claim_ids(
    ground_truth: Mapping[str, Any],
    case: Mapping[str, Any],
    source: Optional[Path],
    ground_truth_path: Optional[Path],
) -> List[str]:
    records = _affected_step_records(ground_truth)
    typed_ids = []
    saw_explicit_type = False
    for record in records:
        if not isinstance(record, dict) or "step_type" not in record:
            continue
        saw_explicit_type = True
        step_id = _step_id_from_affected_record(record)
        if step_id is not None and record.get("step_type") == "claim":
            typed_ids.append(step_id)
    if saw_explicit_type:
        if len(typed_ids) != len(set(typed_ids)):
            raise EvaluationError(
                "affected claim IDs{}contain duplicates".format(_location(source))
            )
        return typed_ids

    all_ids = affected_step_ids(ground_truth)
    step_types = _load_controlled_faulty_step_types(
        case=case,
        ground_truth_path=ground_truth_path,
    )
    if not step_types:
        return all_ids
    claim_ids = [step_id for step_id in all_ids if step_types.get(step_id) == "claim"]
    if len(claim_ids) != len(set(claim_ids)):
        raise EvaluationError(
            "affected claim IDs{}contain duplicates".format(_location(source))
        )
    return claim_ids


def _computed_cost_ratios(counts: Mapping[str, int]) -> Dict[str, Optional[float]]:
    return {
        "replayed_step_ratio": _ratio(
            counts["replayed_target_step_count"], counts["total_original_steps"]
        ),
        "replayed_original_tool_call_ratio": _ratio(
            counts["replayed_tool_call_count"],
            counts["total_original_tool_call_count"],
        ),
        "total_repair_tool_call_ratio": _ratio(
            counts["total_repair_tool_call_count"],
            counts["total_original_tool_call_count"],
        ),
    }


def _validate_stored_ratio(cost: Mapping[str, Any], name: str,
                           computed: Optional[float], source: Optional[Path],
                           case_id: str) -> None:
    if name not in cost:
        return
    stored = cost[name]
    if stored is not None and (isinstance(stored, bool) or not isinstance(stored, (int, float))):
        raise EvaluationError(
            "replay_cost.{}{}must be numeric or null".format(
                name, _location(source, case_id)
            )
        )
    matches = stored is None and computed is None
    if stored is not None and computed is not None:
        matches = math.isclose(float(stored), computed, rel_tol=1e-9, abs_tol=1e-12)
    if not matches:
        raise EvaluationError(
            "replay_cost.{}{}does not match raw counts (stored={!r}, computed={!r})".format(
                name, _location(source, case_id), stored, computed
            )
        )


def _evaluate_cost(cost_value: Any, source: Optional[Path], case_id: str) -> Dict[str, Any]:
    cost = _require_mapping(cost_value, "replay_cost", source, case_id)
    counts = {}  # type: Dict[str, int]
    for field in COUNT_FIELDS:
        if field not in cost:
            raise EvaluationError(
                "replay_cost.{}{}is required".format(
                    field, _location(source, case_id)
                )
            )
        counts[field] = _require_count(
            cost[field], "replay_cost.{}".format(field), source, case_id
        )
    expected_repair_calls = (
        counts["replayed_tool_call_count"] + counts["synthetic_tool_call_count"]
    )
    if counts["total_repair_tool_call_count"] != expected_repair_calls:
        raise EvaluationError(
            "replay_cost.total_repair_tool_call_count{}must equal replayed + "
            "synthetic tool calls ({} != {})".format(
                _location(source, case_id),
                counts["total_repair_tool_call_count"], expected_repair_calls,
            )
        )

    usage = _require_mapping(
        cost.get("llm_token_usage"), "replay_cost.llm_token_usage", source, case_id
    )
    tokens = {}  # type: Dict[str, int]
    for field in TOKEN_FIELDS:
        if field not in usage:
            raise EvaluationError(
                "replay_cost.llm_token_usage.{}{}is required".format(
                    field, _location(source, case_id)
                )
            )
        tokens[field] = _require_count(
            usage[field], "replay_cost.llm_token_usage.{}".format(field),
            source, case_id,
        )
    if tokens["total_tokens"] != tokens["input_tokens"] + tokens["output_tokens"]:
        raise EvaluationError(
            "replay_cost.llm_token_usage.total_tokens{}must equal input + output tokens".format(
                _location(source, case_id)
            )
        )

    ratios = _computed_cost_ratios(counts)
    for name in RATIO_FIELDS:
        _validate_stored_ratio(cost, name, ratios[name], source, case_id)
    result = dict(counts)
    result.update(ratios)
    result["llm_token_usage"] = tokens
    return result


def evaluate_case(
    result: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    source_path: Optional[Path] = None,
    ground_truth_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate one repaired result against its matching ground truth."""
    source = Path(source_path) if source_path is not None else None
    result = _require_mapping(result, "result", source)
    ground_truth = _require_mapping(ground_truth, "ground_truth", source)
    case = _require_mapping(result.get("case"), "result.case", source)
    try:
        case_id = result_task_id(case)
    except ValueError as exc:
        raise EvaluationError("{}{}".format(exc, _location(source))) from exc
    truth_case_id = _require_string(
        ground_truth.get("task_id") or ground_truth.get("case_id"),
        "ground_truth.task_id", source, case_id
    )
    if case_id != truth_case_id:
        raise EvaluationError(
            "result.case.task_id{}does not match ground truth task_id {!r}".format(
                _location(source, case_id), truth_case_id
            )
        )
    method = _require_string(case.get("method"), "result.case.method", source, case_id)
    fault_types = _require_id_list(
        case.get("fault_types"), "result.case.fault_types", source, case_id
    )

    memory_truth = _require_mapping(
        ground_truth.get("memory_ground_truth"), "memory_ground_truth", source, case_id
    )
    memory_repair = _require_mapping(
        result.get("memory_repair"), "memory_repair", source, case_id
    )
    active_ids = set(_require_id_list(
        memory_repair.get("final_active_memory_ids"),
        "memory_repair.final_active_memory_ids", source, case_id,
    ))
    original_active_ids = set(_require_id_list(
        memory_repair.get("original_active_memory_ids"),
        "memory_repair.original_active_memory_ids", source, case_id,
    ))
    deactivated_by_repair_ids = set(_require_id_list(
        memory_repair.get("deactivated_by_repair_memory_ids"),
        "memory_repair.deactivated_by_repair_memory_ids", source, case_id,
    ))
    planned_delete_ids = set(_require_id_list(
        memory_repair.get("planned_delete_memory_ids"),
        "memory_repair.planned_delete_memory_ids", source, case_id,
    ))
    planned_quarantine_ids = set(_require_id_list(
        memory_repair.get("planned_quarantine_memory_ids"),
        "memory_repair.planned_quarantine_memory_ids", source, case_id,
    ))
    remap_value = _require_mapping(
        memory_repair.get("memory_id_remap"), "memory_repair.memory_id_remap",
        source, case_id,
    )
    remap = {}  # type: Dict[str, str]
    for old_id, new_id in remap_value.items():
        if not isinstance(old_id, str) or not isinstance(new_id, str):
            raise EvaluationError(
                "memory_repair.memory_id_remap{}must map string IDs to string IDs".format(
                    _location(source, case_id)
                )
            )
        remap[old_id] = new_id

    def memory_ids(field: str) -> List[str]:
        try:
            return ground_truth_memory_ids(memory_truth.get(field), field)
        except ValueError as exc:
            raise EvaluationError("{}{}".format(exc, _location(source, case_id))) from exc

    benign_ids = _require_id_list(
        memory_truth.get("benign_active_memory_ids"),
        "memory_ground_truth.benign_active_memory_ids", source, case_id,
    )

    try:
        true_claim_ids = _affected_claim_ids(
            ground_truth,
            case,
            source,
            ground_truth_path,
        )
    except ValueError as exc:
        raise EvaluationError("{}{}".format(exc, _location(source, case_id))) from exc
    claim_repair = _require_mapping(
        result.get("claim_repair"), "claim_repair", source, case_id
    )
    predicted_claim_ids = _require_id_list(
        claim_repair.get("applied_invalidated_claim_ids"),
        "claim_repair.applied_invalidated_claim_ids", source, case_id,
    )

    source_faulty_ids = memory_ids("source_faulty_memories")
    faulty_removal_credit_ids = (
        deactivated_by_repair_ids | planned_delete_ids | planned_quarantine_ids
    )
    if method in ("full_memory_reset", "delete_retrieved_memories"):
        # Full reset clears the active memory store rather than targeting every
        # historical faulty ID.  Delete-retrieved similarly acts only on the
        # retrieved active context.  Faulty memories that were already inactive
        # at repair start are therefore counted as removed if they remain absent
        # from the final active state; no-repair does not receive this credit.
        faulty_removal_credit_ids |= set(source_faulty_ids) - active_ids

    output = {
        "case_id": case_id,
        "task_id": case_id,
        "method": method,
        "domain": case.get("domain"),
        "fault_types": sorted(fault_types),
        "faulty_memory_removal_recall": _repair_removal_metric(
            source_faulty_ids,
            faulty_removal_credit_ids,
            "removed_ids"
        ),
        "benign_memory_preservation": (
            # Full reset intentionally clears the active memory store.  Even
            # benign memories that were already historical should not receive
            # preservation credit for this baseline, because the baseline does
            # not preserve normal persistent memory state.
            _preservation_metric(benign_ids, active_ids, remap)
            if method == "full_memory_reset"
            else _repair_preservation_metric(
                benign_ids, original_active_ids, active_ids, remap,
                deactivated_by_repair_ids | planned_delete_ids | planned_quarantine_ids,
            )
        ),
        "affected_claim_invalidation": _claim_metric(
            true_claim_ids, predicted_claim_ids
        ),
        "derived_memory_cleanup_recall": _removal_metric(
            memory_ids("derived_faulty_memories"), active_ids, "cleaned_ids"
        ),
        "replay_cost": _evaluate_cost(result.get("replay_cost"), source, case_id),
    }
    return output


def aggregate_cases(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Micro-aggregate case metrics and aggregate raw replay costs."""
    memory_names = (
        "faulty_memory_removal_recall",
        "benign_memory_preservation",
        "derived_memory_cleanup_recall",
    )
    aggregate = {"case_count": len(cases)}  # type: Dict[str, Any]
    for name in memory_names:
        numerator = sum(case[name]["numerator"] for case in cases)
        denominator = sum(case[name]["denominator"] for case in cases)
        aggregate[name] = {
            "numerator": numerator,
            "denominator": denominator,
            "value": _ratio(numerator, denominator),
        }

    tp = sum(case["affected_claim_invalidation"]["tp"] for case in cases)
    fp = sum(case["affected_claim_invalidation"]["fp"] for case in cases)
    fn = sum(case["affected_claim_invalidation"]["fn"] for case in cases)
    aggregate["affected_claim_invalidation"] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }

    totals = {field: 0 for field in COUNT_FIELDS}
    totals.update({field: 0 for field in TOKEN_FIELDS})
    for case in cases:
        cost = case["replay_cost"]
        for field in COUNT_FIELDS:
            totals[field] += cost[field]
        for field in TOKEN_FIELDS:
            totals[field] += cost["llm_token_usage"][field]
    means = {
        field: (totals[field] / len(cases) if cases else None)
        for field in tuple(COUNT_FIELDS) + TOKEN_FIELDS
    }
    aggregate["replay_cost"] = {
        "totals": totals,
        "mean_per_case": means,
        "ratios": _computed_cost_ratios(totals),
    }
    return aggregate


def load_ground_truth(path: Path) -> Dict[str, Mapping[str, Any]]:
    try:
        return load_ground_truth_index(Path(path))
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc


def discover_result_files(inputs: Sequence[Path]) -> List[Path]:
    found = set()  # type: Set[Path]
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            found.update(item.resolve() for item in path.rglob("repaired_result.json"))
        elif path.is_file():
            found.add(path.resolve())
        else:
            raise EvaluationError("result input does not exist: {}".format(path))
    if not found:
        raise EvaluationError("no repaired_result.json files found in result inputs")
    return sorted(found, key=lambda item: str(item))


def _load_result(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            "invalid JSON in {} line {} column {}: {}".format(
                path, exc.lineno, exc.colno, exc.msg
            )
        )
    except OSError as exc:
        raise EvaluationError("cannot read result {}: {}".format(path, exc))
    return _require_mapping(value, "result", path)


def evaluate_results(ground_truth_path: Path,
                     result_inputs: Sequence[Path],
                     human_judgments_path: Optional[Path] = None) -> Dict[str, Any]:
    ground_truth_path = Path(ground_truth_path).resolve()
    ground_truth = load_ground_truth(ground_truth_path)
    cases = []
    seen = {}  # type: Dict[Tuple[str, Tuple[str, ...], str], Path]
    for path in discover_result_files(result_inputs):
        result = _load_result(path)
        case_value = _require_mapping(result.get("case"), "result.case", path)
        try:
            case_id = result_task_id(case_value)
        except ValueError as exc:
            raise EvaluationError("{}{}".format(exc, _location(path))) from exc
        if case_id not in ground_truth:
            raise EvaluationError(
                "result task_id {!r} in {} does not exist in ground truth {}".format(
                    case_id, path, ground_truth_path
                )
            )
        method = _require_string(case_value.get("method"), "result.case.method", path, case_id)
        fault_types = _require_id_list(
            case_value.get("fault_types"), "result.case.fault_types", path, case_id
        )
        identity = (method, tuple(sorted(fault_types)), case_id)
        if identity in seen:
            raise EvaluationError(
                "duplicate result case {} in {} and {}".format(
                    identity, seen[identity], path
                )
            )
        seen[identity] = path
        cases.append(
            evaluate_case(
                result,
                ground_truth[case_id],
                path,
                ground_truth_path,
            )
        )
    cases.sort(key=lambda case: (
        case["method"], tuple(case["fault_types"]), case["case_id"]
    ))
    aggregate = aggregate_cases(cases)
    if human_judgments_path is not None:
        judgments = load_human_judgments(Path(human_judgments_path))
        attach_human_metrics(cases, judgments)
        aggregate.update(aggregate_human_metrics(cases))
    return {"cases": cases, "aggregate": aggregate}


def write_evaluation(evaluation: Mapping[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    cases_path = output_path.parent / "evaluation_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in evaluation["cases"]:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return cases_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--human-judgments", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evaluation = evaluate_results(
            args.ground_truth, args.results, args.human_judgments
        )
        cases_path = write_evaluation(evaluation, args.output)
    except EvaluationError as exc:
        raise SystemExit("evaluation error: {}".format(exc))
    print("wrote {}".format(args.output))
    print("wrote {}".format(cases_path))
    return 0


if __name__ == "__main__":
    main()
