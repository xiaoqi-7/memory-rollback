"""Run deterministic oracle recovery checks for repaired-result artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import customer_support_oracle, lme_derived_oracle, shopping_oracle, travel_oracle


DOMAIN_ORACLES = {
    "shopping": shopping_oracle.evaluate,
    "travel": travel_oracle.evaluate,
    "customer_support": customer_support_oracle.evaluate,
    "lme_derived": lme_derived_oracle.evaluate,
}
_SUPPLEMENT_ROOT = Path(__file__).resolve().parents[1]


class OracleInputError(ValueError):
    """Raised when oracle input files are missing or malformed."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise OracleInputError(
                        f"{path} line {line_number} must contain a JSON object"
                    )
                rows.append(value)
    except OSError as exc:
        raise OracleInputError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OracleInputError(
            f"invalid JSON in {path} line {exc.lineno}: {exc.msg}"
        ) from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except OSError as exc:
        raise OracleInputError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OracleInputError(
            f"invalid JSON in {path} line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise OracleInputError(f"{path} must contain a JSON object")
    return value


def discover_result_files(inputs: Sequence[Path]) -> list[Path]:
    found: set[Path] = set()
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            found.update(item.resolve() for item in path.rglob("repaired_result.json"))
        elif path.is_file():
            found.add(path.resolve())
        else:
            raise OracleInputError(f"result input does not exist: {path}")
    if not found:
        raise OracleInputError("no repaired_result.json files found in result inputs")
    return sorted(found, key=lambda item: str(item))


def _artifact_path(result_path: Path, result: Mapping[str, Any], key: str) -> Path:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise OracleInputError(f"{result_path} missing artifacts object")
    value = artifacts.get(key)
    if not isinstance(value, str) or not value:
        raise OracleInputError(f"{result_path} artifacts.{key} must be a path string")
    path = Path(value)
    if path.is_absolute():
        raise OracleInputError(f"{result_path} artifacts.{key} must be relative")
    return (result_path.parent / path).resolve()


def _task_index(data_root: Path, domain: str) -> dict[str, dict[str, Any]]:
    tasks_path = data_root / domain / "tasks_full_session.jsonl"
    rows = read_jsonl(tasks_path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if isinstance(task_id, str) and task_id:
            out[task_id] = row
    return out


def evaluate_result_file(
    result_path: Path,
    data_root: Path,
    task_cache: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    result = load_json(result_path)
    case = result.get("case")
    if not isinstance(case, dict):
        raise OracleInputError(f"{result_path} missing case object")
    domain = case.get("domain")
    task_id = case.get("task_id")
    method = case.get("method")
    if domain not in DOMAIN_ORACLES:
        raise OracleInputError(f"{result_path} has unknown domain {domain!r}")
    if not isinstance(task_id, str) or not task_id:
        raise OracleInputError(f"{result_path} case.task_id must be a non-empty string")
    if not isinstance(method, str) or not method:
        raise OracleInputError(f"{result_path} case.method must be a non-empty string")

    if domain not in task_cache:
        task_cache[domain] = _task_index(data_root, domain)
    task = task_cache[domain].get(task_id)
    if task is None:
        raise OracleInputError(
            f"{result_path} task_id {task_id!r} not found in {data_root / domain}"
        )

    memories = read_jsonl(_artifact_path(result_path, result, "repaired_memories"))
    traces = read_jsonl(_artifact_path(result_path, result, "repaired_tracesteps"))
    oracle_result = DOMAIN_ORACLES[domain](task, memories, traces, "repaired").to_dict()
    oracle_result.update(
        {
            "method": method,
            "case_id": task_id,
            "task_id": task_id,
            "result_path": str(result_path),
        }
    )
    return oracle_result


def judgment_from_result(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": row["method"],
        "case_id": row["case_id"],
        "task_id": row.get("task_id", row["case_id"]),
        "evaluation": "recovery",
        "label": "success" if row.get("oracle_pass") is True else "failure",
        "source": "oracle",
    }


def run_oracle(
    result_inputs: Sequence[Path],
    data_root: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    task_cache: dict[str, dict[str, dict[str, Any]]] = {}
    result_rows = [
        evaluate_result_file(path, data_root, task_cache)
        for path in discover_result_files(result_inputs)
    ]
    judgment_rows = [judgment_from_result(row) for row in result_rows]

    results_path = output_dir / "oracle_results.jsonl"
    judgments_path = output_dir / "oracle_judgments.jsonl"
    write_jsonl(results_path, result_rows)
    write_jsonl(judgments_path, judgment_rows)
    return results_path, judgments_path


def data_root_from_config(config_path: Path) -> Path:
    """Derive the benchmark root from a domain config data_source."""
    path = Path(config_path)
    if not path.is_absolute():
        supplement_path = _SUPPLEMENT_ROOT / path
        path = supplement_path if supplement_path.exists() else path
    try:
        with path.resolve().open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise OracleInputError(f"cannot read agent config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise OracleInputError(f"agent config {path} must contain a mapping")
    data_source = config.get("data_source")
    if not isinstance(data_source, str) or not data_source:
        raise OracleInputError(f"agent config {path} has no data_source")
    domain_dir = Path(data_source)
    if not domain_dir.is_absolute():
        domain_dir = _SUPPLEMENT_ROOT / domain_dir
    return domain_dir.resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Benchmark root containing per-domain task files, for example "
            "data/controlled_benchmark or data/adapted_lmev2_subset."
        ),
    )
    source.add_argument(
        "--config",
        type=Path,
        help="Domain agent config whose data_source determines the benchmark root.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = (
        args.data_root.resolve()
        if args.data_root is not None
        else data_root_from_config(args.config)
    )
    results_path, judgments_path = run_oracle(
        args.results,
        data_root,
        args.output_dir.resolve(),
    )
    print(f"oracle results: {results_path}")
    print(f"oracle judgments: {judgments_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
