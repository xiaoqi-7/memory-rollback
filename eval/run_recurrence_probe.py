"""Run recurrence probes against repaired agent state without judging answers.

The output is deliberately small: one final answer (or one error) for each
``(method, case_id, mode)``.  Probe-time state remains in an in-memory AgentRunner and
is never written back to a repair case directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.run_agent import AgentRunner
from eval.ground_truth import (
    fresh_probe_input,
    load_ground_truth_index,
    load_task_index,
    result_task_id,
)


MODES = ("fresh_memory",)


@dataclass(frozen=True)
class ProbeCase:
    method: str
    case_id: str
    task_id: str
    domain: str
    probe_input: str
    result_path: Path
    memories_path: Path


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("%s: invalid JSON or unreadable file: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("%s: top-level JSON must be an object" % path)
    return value


def _read_jsonl(path: Path, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError("%s: %s artifact is unavailable: %s" % (path, label, exc)) from exc
    with handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "%s:%d: invalid %s JSONL: %s" % (path, line_no, label, exc)
                ) from exc
            if not isinstance(row, dict):
                raise ValueError("%s:%d: %s row must be an object" % (path, line_no, label))
            rows.append(row)
    return rows


def load_ground_truth(path: Path) -> Dict[str, Dict[str, Any]]:
    return load_ground_truth_index(path)


def discover_result_paths(inputs: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for raw in inputs:
        path = Path(raw).resolve()
        if path.is_dir():
            found.extend(sorted(path.rglob("repaired_result.json")))
        elif path.is_file():
            found.append(path)
        else:
            raise ValueError("%s: result input does not exist" % path)
    unique: List[Path] = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise ValueError("--results did not resolve to any repaired_result.json")
    return sorted(unique)


def _artifact_path(result_path: Path, artifacts: Mapping[str, Any], key: str) -> Path:
    default = "%s.jsonl" % key
    raw = artifacts.get(key, default)
    if not isinstance(raw, str) or not raw:
        raise ValueError("%s: artifacts.%s must be a non-empty string" % (result_path, key))
    path = Path(raw)
    return path if path.is_absolute() else (result_path.parent / path).resolve()


def load_cases(
    ground_truth_path: Path,
    result_inputs: Sequence[str],
    data_root: Optional[Path] = None,
) -> List[ProbeCase]:
    ground_truth = load_ground_truth(ground_truth_path)
    data_root = (data_root or ground_truth_path.resolve().parent.parent).resolve()
    task_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    cases: List[ProbeCase] = []
    seen_identities = set()
    for result_path in discover_result_paths(result_inputs):
        result = _read_json(result_path)
        case = result.get("case")
        if not isinstance(case, dict):
            raise ValueError("%s: result.case must be an object" % result_path)
        method = case.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("%s: result.case.method must be a non-empty string" % result_path)
        try:
            case_id = result_task_id(case)
        except ValueError as exc:
            raise ValueError("%s: %s" % (result_path, exc)) from exc
        identity = (method, case_id)
        if identity in seen_identities:
            raise ValueError(
                "%s: duplicate result method/case %s/%s" % (result_path, method, case_id)
            )
        seen_identities.add(identity)
        truth = ground_truth.get(case_id)
        if truth is None:
            raise ValueError("%s: task_id %s is missing from %s" % (
                result_path, case_id, ground_truth_path,
            ))
        task_id = case.get("task_id")
        domain = case.get("domain")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("%s: case %s task_id is empty" % (result_path, case_id))
        if not isinstance(domain, str) or not domain:
            raise ValueError("%s: case %s domain is empty" % (result_path, case_id))
        if domain not in task_cache:
            task_cache[domain] = load_task_index(data_root, domain)
        task = task_cache[domain].get(task_id)
        if task is None:
            raise ValueError(
                "%s: task_id %s is missing from %s"
                % (result_path, task_id, data_root / domain / "tasks_full_session.jsonl")
            )
        probe_input = fresh_probe_input(task, task_id)
        artifacts = result.get("artifacts") or {}
        if not isinstance(artifacts, dict):
            raise ValueError("%s: artifacts must be an object" % result_path)
        cases.append(ProbeCase(
            method=method,
            case_id=case_id,
            task_id=task_id,
            domain=domain,
            probe_input=probe_input,
            result_path=result_path,
            memories_path=_artifact_path(result_path, artifacts, "repaired_memories"),
        ))
    return sorted(cases, key=lambda item: (item.method, item.case_id))


def active_memories(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the final active records; absent status follows Memory's default."""
    return [dict(row) for row in rows if row.get("status", "active") == "active"]


def _validate_memory_rows(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    required = ("memory_id", "task_id", "content", "source")
    for index, row in enumerate(rows, 1):
        missing = [key for key in required if not isinstance(row.get(key), str)]
        if missing:
            raise ValueError("%s:%d: repaired memory missing %s" % (
                path, index, ", ".join(missing),
            ))


def _validate_trace_rows(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    for index, row in enumerate(rows, 1):
        if not isinstance(row.get("step_id"), str) or not isinstance(row.get("step_type"), str):
            raise ValueError("%s:%d: repaired trace needs string step_id and step_type" % (
                path, index,
            ))


def _domain_agent_config(requested_config: Path, domain: str) -> Path:
    """Accept an agent config directly, or infer it beside a repair config."""
    try:
        with requested_config.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("%s: invalid config: %s" % (requested_config, exc)) from exc
    if not isinstance(config, dict):
        raise ValueError("%s: config must contain a mapping" % requested_config)
    if config.get("available_tools_source") and config.get("domain"):
        if config.get("domain") != domain:
            raise ValueError("%s: config domain does not match %s" % (requested_config, domain))
        return requested_config
    inferred = requested_config.parent / (domain + ".yaml")
    if not inferred.is_file():
        raise ValueError(
            "%s: repair config has no agent runtime fields and inferred config is missing: %s"
            % (requested_config, inferred)
        )
    return inferred


class ProbeExecutor:
    """Production adapter around the repository's normal AgentRunner."""

    def __init__(
        self,
        config_path: Path,
        runner_factory: Callable[[str, str], AgentRunner] = AgentRunner,
    ) -> None:
        self.config_path = config_path.resolve()
        self.runner_factory = runner_factory

    def _new_runner(self, case: ProbeCase) -> AgentRunner:
        agent_config = _domain_agent_config(self.config_path, case.domain)
        return self.runner_factory(str(agent_config), "clean")

    def execute(self, case: ProbeCase, mode: str) -> str:
        runner = self._new_runner(case)  # one new runtime for every case/mode
        memories = _read_jsonl(case.memories_path, "repaired memories")
        _validate_memory_rows(memories, case.memories_path)
        if mode == "fresh_memory":
            runner.clear()
            runner.agent.memory_store.load_memories(active_memories(memories))
        else:  # defensive for programmatic callers; argparse already validates
            raise ValueError("unknown mode %s" % mode)

        result = runner.run_single(case.task_id, case.probe_input)
        answer = result.get("response") if isinstance(result, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("agent did not generate a final answer for the new probe turn")
        return answer.strip()


def _error_row(case: ProbeCase, mode: str, exc: Exception) -> Dict[str, str]:
    return {
        "method": case.method,
        "case_id": case.case_id,
        "task_id": case.task_id,
        "mode": mode,
        "status": "error",
        "error": "method=%s; task_id=%s; mode=%s; %s; result=%s; memories=%s"
        % (
            case.method, case.case_id, mode, exc, case.result_path,
            case.memories_path,
        ),
    }


def run_cases(
    cases: Sequence[ProbeCase],
    modes: Sequence[str],
    executor: Any,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    total = len(cases) * len(modes)
    current = 0
    for case in cases:
        for mode in modes:
            current += 1
            print(
                "[recurrence] {}/{} task_id={} mode={}".format(
                    current, total, case.task_id, mode
                ),
                flush=True,
            )
            try:
                answer = executor.execute(case, mode)
                rows.append({
                    "method": case.method,
                    "case_id": case.case_id,
                    "task_id": case.task_id,
                    "mode": mode,
                    "status": "completed",
                    "probe_answer": answer,
                })
                print(
                    "[recurrence] completed task_id={} mode={}".format(
                        case.task_id, mode
                    ),
                    flush=True,
                )
            except Exception as exc:  # isolate each (case, mode)
                rows.append(_error_row(case, mode, exc))
                print(
                    "[recurrence] error task_id={} mode={}: {}".format(
                        case.task_id, mode, exc
                    ),
                    flush=True,
                )
    return sorted(rows, key=lambda row: (row["method"], row["case_id"], row["mode"]))


def write_results_atomic(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repaired-state recurrence probes")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--results", required=True, nargs="+")
    parser.add_argument("--config", required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["fresh_memory"])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Benchmark data root containing per-domain tasks_full_session.jsonl files. "
            "Defaults to the parent of the ground-truth domain directory."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must not contain duplicates")
    cases = load_cases(Path(args.ground_truth).resolve(), args.results, args.data_root)
    rows = run_cases(cases, args.modes, ProbeExecutor(Path(args.config)))
    write_results_atomic(Path(args.output), rows)
    completed = sum(row["status"] == "completed" for row in rows)
    errors = len(rows) - completed
    print("wrote %d recurrence results (%d completed, %d error) to %s" % (
        len(rows), completed, errors, Path(args.output).resolve(),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
