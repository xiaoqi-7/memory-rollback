"""Unified baseline runner."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import importlib
import inspect
import json
import os
import re
import string
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from methods.ours.run_ours import (
    CaseOutputTransaction,
    _load_agent_domain,
    _load_and_validate_case_records,
    resolve_case_output_dir,
)

from baselines.common_tools.result_writer import (
    DEFAULT_BASELINE_OUTPUT_TEMPLATE,
    write_case_outputs,
)


BASELINES = {
    "no_repair": ("baselines.no_repair.baseline", "run_no_repair"),
    "full_memory_reset": (
        "baselines.full_memory_reset.baseline",
        "run_full_memory_reset",
    ),
    "delete_retrieved_memories": (
        "baselines.delete_retrieved_memories.baseline",
        "run_delete_retrieved_memories",
    ),
    "llm_judge_repair": (
        "baselines.llm_judge_repair.baseline",
        "run_llm_judge_repair",
    ),
    "memaudit_style": (
        "baselines.memaudit_style.baseline",
        "run_memaudit_style",
    ),
    "agenttrace_style": (
        "baselines.agenttrace_style.baseline",
        "run_agenttrace_style",
    ),
}


def _load_baseline(name: str):
    if name not in BASELINES:
        raise KeyError(name)
    module_name, function_name = BASELINES[name]
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


@dataclass(frozen=True)
class CaseSpec:
    memories: str
    steps: str
    session: str
    task_id: str
    faulty_memory_ids: List[str]
    agent_config: str
    domain: Optional[str] = None
    fault_types: Optional[List[str]] = None
    case_id: Optional[str] = None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _task_id_candidates(task_id: str) -> List[str]:
    candidates = [task_id]
    match = re.search(r"(task_\d+)$", task_id or "")
    if match:
        candidates.append(match.group(1))
    return list(dict.fromkeys(candidates))


def _filter_task_with_fallback(records: List[Dict[str, Any]], task_id: str) -> List[Dict[str, Any]]:
    wanted = set(_task_id_candidates(task_id))
    return [
        row for row in records
        if row.get("task_id") in wanted or row.get("original_task_id") in wanted
    ]


def _load_case_records_with_baseline_fallback(
    case: "CaseSpec",
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[Optional[str]]]:
    try:
        return _load_and_validate_case_records(
            case.memories,
            case.steps,
            case.session,
            case.task_id,
            case.faulty_memory_ids,
        )
    except ValueError as first_error:
        memories_path = Path(case.memories)
        steps_path = Path(case.steps)
        session_path = Path(case.session)
        all_memories = _load_jsonl(memories_path)
        all_steps = _load_jsonl(steps_path)
        all_sessions = _load_jsonl(session_path)
        memories = _filter_task_with_fallback(all_memories, case.task_id)
        steps = _filter_task_with_fallback(all_steps, case.task_id)
        session_records = _filter_task_with_fallback(all_sessions, case.task_id)
        if not memories or not steps or not session_records:
            raise first_error
        memory_by_id = {
            record.get("memory_id"): record
            for record in memories
            if isinstance(record.get("memory_id"), str)
        }
        missing = [
            memory_id for memory_id in case.faulty_memory_ids
            if memory_id not in memory_by_id
        ]
        if missing:
            raise first_error
        session = session_records[0].get("session", [])
        if not isinstance(session, list):
            raise ValueError(f"session for task {case.task_id!r} must be a list")
        fault_types = [
            memory_by_id[memory_id].get("fault_type")
            if isinstance(memory_by_id[memory_id].get("fault_type"), str)
            and memory_by_id[memory_id].get("fault_type")
            else None
            for memory_id in case.faulty_memory_ids
        ]
        return steps, memories, session, fault_types


def _resolve_registry_path(registry_path: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((registry_path.parent / path).as_posix())


def _registry_agent_config(row: Dict[str, Any]) -> str:
    run_ours = row.get("run_ours")
    if isinstance(run_ours, dict):
        agent_config = run_ours.get("agent_config")
        if isinstance(agent_config, str) and agent_config:
            return agent_config
    domain = row.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError(
            f"registry row {row.get('case_id')!r} has no domain or agent_config"
        )
    return f"configs/{domain}.yaml"


def _case_from_registry_row(
    row: Dict[str, Any],
    *,
    registry_path: Path,
) -> CaseSpec:
    paths = row.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"registry row {row.get('case_id')!r} has no paths object")
    required_paths = ("faulty_memories", "faulty_tracesteps", "session")
    missing_paths = [
        name for name in required_paths
        if not isinstance(paths.get(name), str) or not paths.get(name)
    ]
    if missing_paths:
        raise ValueError(
            f"registry row {row.get('case_id')!r} missing paths: {missing_paths}"
        )

    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"registry row {row.get('case_id')!r} has no task_id")
    faulty_memory_ids = row.get("faulty_memory_ids")
    if (
        not isinstance(faulty_memory_ids, list)
        or any(not isinstance(item, str) or not item for item in faulty_memory_ids)
    ):
        raise ValueError(
            f"registry row {row.get('case_id')!r} has invalid faulty_memory_ids"
        )

    fault_types = row.get("fault_types")
    if not isinstance(fault_types, list) or not fault_types:
        fault_type = row.get("fault_type")
        fault_types = [fault_type] if isinstance(fault_type, str) and fault_type else []
    if any(not isinstance(item, str) or not item for item in fault_types):
        raise ValueError(f"registry row {row.get('case_id')!r} has invalid fault_types")

    domain = row.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError(f"registry row {row.get('case_id')!r} has no domain")

    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("registry row has no case_id")

    return CaseSpec(
        memories=_resolve_registry_path(registry_path, paths["faulty_memories"]),
        steps=_resolve_registry_path(registry_path, paths["faulty_tracesteps"]),
        session=_resolve_registry_path(registry_path, paths["session"]),
        task_id=task_id,
        faulty_memory_ids=sorted(set(faulty_memory_ids)),
        agent_config=_registry_agent_config(row),
        domain=domain,
        fault_types=list(fault_types),
        case_id=case_id,
    )


def _load_registry_cases(registry: str, case_ids: Optional[List[str]],
                         all_cases: bool) -> List[CaseSpec]:
    registry_path = Path(registry)
    rows = _load_jsonl(registry_path)
    if all_cases:
        selected_rows = rows
    else:
        wanted = list(case_ids or [])
        if not wanted:
            raise ValueError("registry mode requires --case_id or --all_cases")
        by_case_id = {row.get("case_id"): row for row in rows}
        missing = [case_id for case_id in wanted if case_id not in by_case_id]
        if missing:
            raise ValueError(f"case_id(s) not found in registry: {missing}")
        selected_rows = [by_case_id[case_id] for case_id in wanted]
    return [
        _case_from_registry_row(row, registry_path=registry_path)
        for row in selected_rows
    ]


def _legacy_case_from_args(args: argparse.Namespace) -> CaseSpec:
    required = {
        "--memories": args.memories,
        "--steps": args.steps,
        "--session": args.session,
        "--task_id": args.task_id,
        "--faulty_memory_ids": args.faulty_memory_ids,
        "--agent_config": args.agent_config,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "legacy explicit mode requires " + ", ".join(missing)
        )
    return CaseSpec(
        memories=args.memories,
        steps=args.steps,
        session=args.session,
        task_id=args.task_id,
        faulty_memory_ids=sorted(set(args.faulty_memory_ids)),
        agent_config=args.agent_config,
    )


def _template_fields(template: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name is not None
    }


def _ensure_multi_case_output_template(output_dir: str, case_count: int) -> None:
    if case_count <= 1:
        return
    if not _template_fields(output_dir):
        raise ValueError(
            "running multiple registry cases requires an --output_dir template "
            "with placeholders such as {method}, {domain}, {task_id}, or "
            "{faulty_memory_ids}"
        )


def run_one_case(
    *,
    baseline_name: str,
    baseline_fn: Any,
    case: CaseSpec,
    output_template: str,
) -> Dict[str, Any]:
    domain = case.domain or _load_agent_domain(case.agent_config)
    fault_types = case.fault_types or []
    output_dir = resolve_case_output_dir(
        output_template,
        method=baseline_name,
        domain=domain,
        task_id=case.task_id,
        faulty_memory_ids=case.faulty_memory_ids,
        fault_types=fault_types,
    )
    case_id = case.case_id or resolve_case_output_dir(
        "{domain}/{fault_types}/{task_id}/{faulty_memory_ids}",
        method=baseline_name,
        domain=domain,
        task_id=case.task_id,
        faulty_memory_ids=case.faulty_memory_ids,
        fault_types=fault_types,
    )
    existing_result = Path(output_dir) / "repaired_result.json"
    if existing_result.is_file():
        print(f"=== baseline {baseline_name}: {case.task_id} ===")
        print(f"case_id: {case_id}")
        print(f"method: {baseline_name}")
        print(f"skip existing: {existing_result}")
        return {
            "payload": None,
            "output_dir": output_dir,
            "case_id": case_id,
            "skipped": True,
        }

    steps, memories, _session, loaded_fault_types = (
        _load_case_records_with_baseline_fallback(case)
    )
    if not fault_types:
        fault_types = loaded_fault_types

    with CaseOutputTransaction(output_dir) as staging_dir:
        baseline_kwargs = {
            "case_id": case_id,
            "domain": domain,
            "session": _session,
            "agent_config": case.agent_config,
        }
        if "output_dir" in inspect.signature(baseline_fn).parameters:
            baseline_kwargs["output_dir"] = staging_dir
        case_outputs = baseline_fn(
            case.task_id,
            steps,
            memories,
            case.faulty_memory_ids,
            **baseline_kwargs,
        )
        if case_outputs.get("outputs_written"):
            payload = case_outputs["payload"]
        else:
            payload = write_case_outputs(
                output_dir=staging_dir,
                plan=case_outputs["plan"],
                original_trace_steps=steps,
                original_memories=memories,
                replay_result=case_outputs["replay_result"],
                repaired_trace_steps=case_outputs["repaired_trace_steps"],
                repaired_memories=case_outputs["repaired_memories"],
                runtime_config=case_outputs.get("runtime_config"),
                cost_overrides=case_outputs.get("cost_overrides"),
            )

    print(f"=== baseline {baseline_name}: {case.task_id} ===")
    print(f"case_id: {payload['case']['case_id']}")
    print(f"method: {payload['case']['method']}")
    print(f"wrote: {output_dir}/repaired_result.json")
    print(f"wrote: {output_dir}/repaired_memories.jsonl")
    print(f"wrote: {output_dir}/repaired_tracesteps.jsonl")
    return {
        "payload": payload,
        "output_dir": output_dir,
        "case_id": payload["case"]["case_id"],
    }


def _case_output_metadata(
    *,
    baseline_name: str,
    case: CaseSpec,
    output_template: str,
) -> tuple[str, str]:
    domain = case.domain or _load_agent_domain(case.agent_config)
    fault_types = case.fault_types or []
    output_dir = resolve_case_output_dir(
        output_template,
        method=baseline_name,
        domain=domain,
        task_id=case.task_id,
        faulty_memory_ids=case.faulty_memory_ids,
        fault_types=fault_types,
    )
    case_id = case.case_id or resolve_case_output_dir(
        "{domain}/{fault_types}/{task_id}/{faulty_memory_ids}",
        method=baseline_name,
        domain=domain,
        task_id=case.task_id,
        faulty_memory_ids=case.faulty_memory_ids,
        fault_types=fault_types,
    )
    return output_dir, case_id


def _write_case_error(
    *,
    baseline_name: str,
    case: CaseSpec,
    output_template: str,
    exc: Exception,
) -> Dict[str, Any]:
    output_dir, case_id = _case_output_metadata(
        baseline_name=baseline_name,
        case=case,
        output_template=output_template,
    )
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    error_path = path / "error.json"
    payload = {
        "case": {
            "method": baseline_name,
            "task_id": case.task_id,
            "domain": case.domain,
            "fault_types": case.fault_types or [],
            "faulty_memory_ids": case.faulty_memory_ids,
            "case_id": case_id,
        },
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }
    error_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "payload": None,
        "output_dir": output_dir,
        "case_id": case_id,
        "error": True,
        "error_path": str(error_path),
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one baseline case.")
    parser.add_argument(
        "--baseline",
        required=True,
        choices=sorted(BASELINES),
        help="baseline name",
    )
    parser.add_argument("--memories", help="faulty memories JSONL path")
    parser.add_argument("--steps", help="faulty trace steps JSONL path")
    parser.add_argument(
        "--session", help="tasks_full_session JSONL path"
    )
    parser.add_argument("--task_id", help="task_id to filter to")
    parser.add_argument(
        "--faulty_memory_ids", nargs="+",
        help="faulty memory id(s) for this case",
    )
    parser.add_argument(
        "--agent_config",
        help="domain agent config such as configs/travel.yaml",
    )
    parser.add_argument(
        "--registry",
        help=(
            "registry JSONL path for --case_id/--all_cases mode; required "
            "because controlled_benchmark and adapted_lmev2_subset use "
            "different registries"
        ),
    )
    parser.add_argument(
        "--case_id",
        nargs="+",
        help="registry case_id(s) to run",
    )
    parser.add_argument(
        "--all_cases",
        action="store_true",
        help="run every case in --registry for the selected baseline",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_BASELINE_OUTPUT_TEMPLATE,
        help="case output template supporting {method}, {domain}, "
             "{fault_types}, {task_id}, and {faulty_memory_ids}. A literal "
             "path is preserved exactly.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue registry runs after a case failure and write error.json.",
    )
    args = parser.parse_args(argv)
    if args.all_cases and args.case_id:
        parser.error("--all_cases cannot be combined with --case_id")
    if (args.case_id or args.all_cases) and not args.registry:
        parser.error("--registry is required with --case_id or --all_cases")

    baseline_fn = _load_baseline(args.baseline)
    continue_on_error = args.continue_on_error or os.getenv("BASELINE_CONTINUE_ON_ERROR") == "1"
    try:
        if args.case_id or args.all_cases:
            cases = _load_registry_cases(args.registry, args.case_id, args.all_cases)
        else:
            cases = [_legacy_case_from_args(args)]
        _ensure_multi_case_output_template(args.output_dir, len(cases))
        results = []
        for case in cases:
            try:
                results.append(
                    run_one_case(
                        baseline_name=args.baseline,
                        baseline_fn=baseline_fn,
                        case=case,
                        output_template=args.output_dir,
                    )
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                result = _write_case_error(
                    baseline_name=args.baseline,
                    case=case,
                    output_template=args.output_dir,
                    exc=exc,
                )
                print(f"=== baseline {args.baseline}: {case.task_id} ===")
                print(f"case_id: {result['case_id']}")
                print(f"method: {args.baseline}")
                print(f"error: {type(exc).__name__}: {exc}")
                print(f"wrote: {result['error_path']}")
                print("continue_on_error: skipping failed case")
                results.append(result)
    except ValueError as exc:
        parser.error(str(exc))

    if len(results) > 1:
        skipped = sum(1 for result in results if result.get("skipped"))
        errors = sum(1 for result in results if result.get("error"))
        completed = len(results) - skipped - errors
        print(
            f"=== completed {completed} {args.baseline} baseline cases; "
            f"skipped {skipped} existing cases; errors {errors} cases ==="
        )


if __name__ == "__main__":
    main()
