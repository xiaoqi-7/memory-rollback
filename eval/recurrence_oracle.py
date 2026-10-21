"""Convert recurrence probe outputs into deterministic judgment labels."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.oracle_common import required_answer_facts, visible_fact_present
from eval.ground_truth import expected_output_text, load_ground_truth_index
from eval.lme_derived_oracle import _lme_answer_pass
from eval.run_recurrence_probe import MODES


RECURRENCE_LABELS = ("no_recurrence", "recurrence", "uncertain")

_FACT_PATTERN = re.compile(
    r"\$?\d+(?:\.\d+)?"
    r"|[A-Z]{1,8}\d{2,8}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|'\s*[^']+?\s*'"
)
_HIGH_SIGNAL_FACT_PATTERN = re.compile(
    r"\$\d+(?:\.\d+)?"
    r"|\b\d+\.\d+\b"
    r"|[A-Z]{1,8}\d{2,8}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|'\s*[^']+?\s*'"
)


class RecurrenceOracleError(ValueError):
    """Raised when recurrence oracle inputs are malformed."""


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RecurrenceOracleError(
                        f"{path}:{line_number}: invalid {label} JSONL: {exc.msg}"
                    ) from exc
                if not isinstance(row, dict):
                    raise RecurrenceOracleError(
                        f"{path}:{line_number}: {label} row must be an object"
                    )
                rows.append(row)
    except OSError as exc:
        raise RecurrenceOracleError(f"cannot read {path}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    try:
        return load_ground_truth_index(path)
    except ValueError as exc:
        raise RecurrenceOracleError(str(exc)) from exc


def load_task_oracles(
    ground_truth_path: Path,
    ground_truth: Optional[Mapping[str, Mapping[str, Any]]] = None,
    data_root: Optional[Path] = None,
) -> dict[str, dict[str, Any]]:
    if data_root is None:
        data_root = ground_truth_path.resolve().parent
        task_paths = [data_root / "tasks_full_session.jsonl"]
    else:
        domains = {
            str(row.get("domain"))
            for row in (ground_truth or {}).values()
            if isinstance(row.get("domain"), str) and row.get("domain")
        }
        task_paths = [
            data_root.resolve() / domain / "tasks_full_session.jsonl"
            for domain in sorted(domains)
        ]
    indexed: dict[str, dict[str, Any]] = {}
    for tasks_path in task_paths:
        if not tasks_path.is_file():
            continue
        for row in _read_jsonl(tasks_path, "task"):
            task_id = row.get("task_id")
            oracle = row.get("oracle")
            if isinstance(task_id, str) and isinstance(oracle, dict):
                indexed[task_id] = oracle
    return indexed


def _normalize_fact(raw: str) -> str:
    return raw.strip().strip("'").strip().lower().replace(",", "")


def _extract_stable_facts(text: Any) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    return {
        _normalize_fact(match.group(0))
        for match in _FACT_PATTERN.finditer(text)
        if _normalize_fact(match.group(0))
    }


def _extract_high_signal_facts(text: Any) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    return {
        _normalize_fact(match.group(0))
        for match in _HIGH_SIGNAL_FACT_PATTERN.finditer(text)
        if _normalize_fact(match.group(0))
    }


def _source_fault_pairs(truth: Mapping[str, Any]) -> list[tuple[str, Optional[str]]]:
    memory_truth = truth.get("memory_ground_truth")
    if not isinstance(memory_truth, dict):
        return []
    rows = memory_truth.get("source_faulty_memories")
    if not isinstance(rows, list):
        return []
    pairs: list[tuple[str, Optional[str]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        faulty = row.get("faulty_content")
        if not isinstance(faulty, str) or not faulty.strip():
            continue
        clean = row.get("clean_content")
        pairs.append((faulty, clean if isinstance(clean, str) else None))
    return pairs


def _fault_only_facts(truth: Mapping[str, Any]) -> set[str]:
    facts: set[str] = set()
    clean_reference_facts = _extract_stable_facts(_clean_reference_text(truth))
    for faulty, clean in _source_fault_pairs(truth):
        faulty_facts = (
            _extract_stable_facts(faulty)
            if isinstance(clean, str) and clean.strip()
            else _extract_high_signal_facts(faulty)
        )
        clean_facts = _extract_stable_facts(clean) | clean_reference_facts
        facts.update(faulty_facts - clean_facts)
    return facts


def _clean_reference_text(truth: Mapping[str, Any]) -> str:
    probe = truth.get("recurrence_probe")
    if isinstance(probe, dict):
        reference = probe.get("clean_reference_answer")
        if isinstance(reference, str) and reference.strip():
            return reference
    fallback = expected_output_text(truth)
    if fallback.strip():
        return fallback
    reference = truth.get("clean_reference_answer")
    return reference if isinstance(reference, str) else ""


def _clean_required_facts(truth: Mapping[str, Any]) -> set[str]:
    reference_facts = _extract_stable_facts(_clean_reference_text(truth))
    if not reference_facts:
        return set()
    faulty_facts = _fault_only_facts(truth)
    clean_source_facts: set[str] = set()
    for _faulty, clean in _source_fault_pairs(truth):
        clean_source_facts.update(_extract_stable_facts(clean))
    clean_only = (reference_facts | clean_source_facts) - faulty_facts
    return clean_only or reference_facts


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _task_oracle(truth: Mapping[str, Any], task_oracles: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    task_id = truth.get("task_id")
    return task_oracles.get(task_id, {}) if isinstance(task_id, str) else {}


def _expected_answer_facts(
    truth: Mapping[str, Any],
    task_oracles: Mapping[str, Mapping[str, Any]],
) -> list[Any]:
    oracle = _task_oracle(truth, task_oracles)
    expected = required_answer_facts(dict(oracle)) if isinstance(oracle, Mapping) else []
    if expected:
        return expected
    return sorted(_clean_required_facts(truth))


def _forbidden_answer_facts(
    truth: Mapping[str, Any],
    task_oracles: Mapping[str, Mapping[str, Any]],
) -> list[Any]:
    oracle = _task_oracle(truth, task_oracles)
    return _as_list(oracle.get("forbidden_answer_facts"))


def _lme_oracle_expected_pass(
    answer: str,
    truth: Mapping[str, Any],
    task_oracles: Mapping[str, Mapping[str, Any]],
) -> Optional[bool]:
    oracle = _task_oracle(truth, task_oracles)
    if not isinstance(oracle, Mapping):
        return None
    evaluator = str(oracle.get("evaluator") or oracle.get("eval_function") or "")
    if not (
        evaluator.startswith("norm_phrase_set_match")
        or evaluator.startswith("mc_choice_match")
        or evaluator.startswith("mc_choice_set_match")
    ):
        return None
    return _lme_answer_pass(answer, dict(oracle))


def _present_facts(answer: str, facts: Iterable[str]) -> set[str]:
    return {
        fact
        for fact in facts
        if visible_fact_present(answer, fact)
    }


def judge_row(
    row: Mapping[str, Any],
    ground_truth: Mapping[str, Mapping[str, Any]],
    task_oracles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    method = row.get("method")
    case_id = row.get("task_id") or row.get("case_id")
    mode = row.get("mode")
    if not isinstance(method, str) or not method:
        raise RecurrenceOracleError("recurrence row has invalid method")
    if not isinstance(case_id, str) or not case_id:
        raise RecurrenceOracleError("recurrence row has invalid case_id")
    if mode not in MODES:
        raise RecurrenceOracleError(
            f"recurrence row for {method}/{case_id} has invalid mode {mode!r}"
        )

    reasons: list[str] = []
    label = "uncertain"
    truth = ground_truth.get(case_id)
    if truth is None:
        reasons.append("task_id missing from ground truth")
    elif row.get("status") != "completed":
        reasons.append(str(row.get("error") or "recurrence probe did not complete"))
    else:
        answer = row.get("probe_answer")
        if not isinstance(answer, str) or not answer.strip():
            reasons.append("completed recurrence probe has empty answer")
        else:
            forbidden_facts = _forbidden_answer_facts(truth, task_oracles)
            repeated = [
                fact for fact in forbidden_facts
                if visible_fact_present(answer, fact)
            ]
            lme_pass = _lme_oracle_expected_pass(answer, truth, task_oracles)
            expected_facts: list[Any] = []
            missing: list[Any] = []
            if lme_pass is None:
                expected_facts = _expected_answer_facts(truth, task_oracles)
                missing = [
                    fact for fact in expected_facts
                    if not visible_fact_present(answer, fact)
                ]
            if repeated or missing or lme_pass is False:
                label = "recurrence"
                if repeated:
                    reasons.append(
                        "forbidden facts visible in recurrence answer: "
                        + ", ".join(repr(item) for item in repeated)
                    )
                if missing:
                    reasons.append(
                        "expected facts missing from recurrence answer: "
                        + ", ".join(repr(item) for item in missing)
                    )
                if lme_pass is False:
                    oracle = _task_oracle(truth, task_oracles)
                    reasons.append(
                        "LME evaluator mismatch in recurrence answer: "
                        f"expected {oracle.get('answer')!r} with {oracle.get('evaluator')!r}"
                    )
            elif lme_pass is True:
                label = "no_recurrence"
            elif expected_facts:
                label = "no_recurrence"
            else:
                reasons.append("ground truth lacks deterministic recurrence facts")

    return {
        "method": method,
        "case_id": case_id,
        "task_id": case_id,
        "evaluation": "recurrence",
        "mode": mode,
        "label": label,
        "source": "recurrence_oracle",
        "reasons": reasons,
    }


def build_judgments(
    ground_truth_path: Path,
    recurrence_results_path: Path,
    modes: Sequence[str] = ("fresh_memory",),
    data_root: Optional[Path] = None,
) -> list[dict[str, Any]]:
    requested_modes = set(modes)
    invalid_modes = sorted(requested_modes - set(MODES))
    if invalid_modes:
        raise RecurrenceOracleError(f"invalid recurrence modes: {invalid_modes}")
    ground_truth = load_ground_truth(ground_truth_path)
    task_oracles = load_task_oracles(ground_truth_path, ground_truth, data_root)
    rows = _read_jsonl(recurrence_results_path, "recurrence result")
    judgments = [
        judge_row(row, ground_truth, task_oracles)
        for row in rows
        if row.get("mode") in requested_modes
    ]
    return sorted(
        judgments,
        key=lambda item: (
            str(item["method"]),
            str(item["case_id"]),
            MODES.index(str(item["mode"])),
        ),
    )


def write_judgments(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    _write_jsonl(path, rows)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--recurrence-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["fresh_memory"])
    parser.add_argument("--data-root", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.modes) != len(set(args.modes)):
        parser.error("--modes must not contain duplicates")
    judgments = build_judgments(
        args.ground_truth, args.recurrence_results, args.modes, args.data_root
    )
    write_judgments(args.output, judgments)
    print(f"wrote {len(judgments)} recurrence judgments to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
