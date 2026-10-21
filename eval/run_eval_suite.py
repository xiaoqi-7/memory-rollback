"""Run fresh-memory recurrence probes, optional oracle labels, and metrics.

Without oracle judgments the suite writes deterministic structural metrics.
With oracle judgments it writes final metrics including recovery/recurrence labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import (
    metrics,
    recurrence_oracle,
    run_oracle,
    run_recurrence_probe,
)


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _run_metrics(
    ground_truth: Path,
    results: Sequence[Path],
    output: Path,
    human_judgments: Optional[Path] = None,
) -> Path:
    evaluation = metrics.evaluate_results(ground_truth, results, human_judgments)
    return metrics.write_evaluation(evaluation, output)


def _run_recurrence(
    ground_truth: Path,
    results: Sequence[Path],
    config: Path,
    output: Path,
    modes: Sequence[str],
    data_root: Path,
    eligible_task_ids: Optional[set[str]] = None,
    resume_existing: bool = False,
) -> Dict[str, int]:
    cases = run_recurrence_probe.load_cases(
        ground_truth.resolve(), [str(path) for path in results], data_root
    )
    loaded_cases = len(cases)
    if eligible_task_ids is not None:
        cases = [case for case in cases if case.task_id in eligible_task_ids]
    resumed_rows = []
    resumed_completed_identities = set()
    if resume_existing and output.is_file():
        for row in run_oracle.read_jsonl(output):
            if row.get("status") != "completed":
                continue
            if (
                eligible_task_ids is not None
                and row.get("task_id") not in eligible_task_ids
            ):
                continue
            identity = (row.get("method"), row.get("case_id"), row.get("mode"))
            if all(isinstance(item, str) and item for item in identity):
                resumed_rows.append(row)
                resumed_completed_identities.add(identity)
    if resumed_completed_identities:
        cases = [
            case for case in cases
            if any(
                (case.method, case.case_id, mode) not in resumed_completed_identities
                for mode in modes
            )
        ]
    rows = run_recurrence_probe.run_cases(
        cases, modes, run_recurrence_probe.ProbeExecutor(config)
    )
    rows = [*resumed_rows, *rows]
    run_recurrence_probe.write_results_atomic(output, rows)
    completed = sum(row["status"] == "completed" for row in rows)
    return {
        "loaded_cases": loaded_cases,
        "probed_cases": len(cases),
        "resumed_completed": len(resumed_rows),
        "total": len(rows),
        "completed": completed,
        "errors": len(rows) - completed,
    }


def _run_recovery_oracle(
    results: Sequence[Path],
    data_root: Path,
    output_dir: Path,
) -> Path:
    _, judgments_path = run_oracle.run_oracle(results, data_root, output_dir)
    return judgments_path


def _success_task_ids(judgments_path: Path) -> set[str]:
    success: set[str] = set()
    for row in run_oracle.read_jsonl(judgments_path):
        if row.get("evaluation") != "recovery" or row.get("label") != "success":
            continue
        task_id = row.get("task_id") or row.get("case_id")
        if isinstance(task_id, str) and task_id:
            success.add(task_id)
    return success


def _append_recurrence_judgments(
    judgments_path: Path,
    ground_truth: Path,
    recurrence_results: Path,
    data_root: Path,
    modes: Sequence[str] = ("fresh_memory",),
) -> Path:
    recovery_rows = run_oracle.read_jsonl(judgments_path)
    recurrence_rows = recurrence_oracle.build_judgments(
        ground_truth, recurrence_results, modes, data_root
    )
    run_oracle.write_jsonl(judgments_path, [*recovery_rows, *recurrence_rows])
    return judgments_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-recurrence",
        action="store_true",
        help="Only run metrics; disables fresh-memory recurrence probes.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Skip the final metrics write after recurrence/oracle.",
    )
    parser.add_argument(
        "--use-oracle-judgments",
        action="store_true",
        help="Generate oracle recovery labels and use them for final metrics.",
    )
    parser.add_argument(
        "--resume-recurrence",
        action="store_true",
        help="Keep completed rows from an existing recurrence_results.jsonl and rerun missing/error rows.",
    )
    parser.add_argument(
        "--oracle-data-root",
        type=Path,
        help=(
            "Benchmark root containing per-domain task files. By default it "
            "is derived as the parent of the ground-truth domain directory."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.skip_recurrence and args.skip_final:
        parser.error("--skip-recurrence --skip-final would not run any suite step")

    ground_truth = args.ground_truth.resolve()
    data_root = (
        args.oracle_data_root.resolve()
        if args.oracle_data_root is not None
        else ground_truth.parent.parent
    )
    output_dir = args.output_dir.resolve()
    modes = ("fresh_memory",)
    judgment_path = output_dir / "oracle_judgments.jsonl"

    oracle_judgments = None
    eligible_recurrence_task_ids = None
    if args.use_oracle_judgments:
        oracle_judgments = _run_recovery_oracle(
            args.results,
            data_root,
            output_dir,
        )
        judgment_path = oracle_judgments
        eligible_recurrence_task_ids = _success_task_ids(judgment_path)

    recurrence_results = output_dir / "recurrence_results.jsonl"
    recurrence_summary = None
    if not args.skip_recurrence:
        recurrence_summary = _run_recurrence(
            ground_truth,
            args.results,
            args.config,
            recurrence_results,
            modes,
            data_root,
            eligible_recurrence_task_ids,
            args.resume_recurrence,
        )
        if oracle_judgments is not None:
            oracle_judgments = _append_recurrence_judgments(
                judgment_path,
                ground_truth,
                recurrence_results,
                data_root,
                modes,
            )
            judgment_path = oracle_judgments
    if args.use_oracle_judgments and args.skip_recurrence:
        oracle_judgments = judgment_path

    has_judgments = _nonempty_file(judgment_path)
    deterministic_metrics = output_dir / "deterministic" / "metrics.json"
    deterministic_cases = None
    final_metrics = output_dir / "final" / "metrics.json"
    final_cases = None
    if not args.skip_final:
        if has_judgments:
            final_cases = _run_metrics(
                ground_truth, args.results, final_metrics, judgment_path
            )
        else:
            deterministic_cases = _run_metrics(
                ground_truth, args.results, deterministic_metrics
            )

    print("\neval suite outputs")
    if recurrence_summary is not None:
        print(
            "recurrence results: {} ({} total, {} completed, {} error)".format(
                recurrence_results,
                recurrence_summary["total"],
                recurrence_summary["completed"],
                recurrence_summary["errors"],
            )
        )
        if "loaded_cases" in recurrence_summary:
            print(
                "recurrence probed cases: {} / {} loaded repaired cases".format(
                    recurrence_summary["probed_cases"],
                    recurrence_summary["loaded_cases"],
                )
            )
        if recurrence_summary.get("resumed_completed"):
            print(
                "recurrence resumed completed rows: {}".format(
                    recurrence_summary["resumed_completed"],
                )
            )
    if oracle_judgments is not None:
        print("oracle judgments: {}".format(oracle_judgments))
    if args.skip_final:
        print("metrics write skipped by --skip-final")
    elif has_judgments:
        print("oracle judgments found; wrote final metrics")
    else:
        print("oracle judgments not found or empty; wrote deterministic metrics")
    if deterministic_cases is not None:
        print("deterministic metrics: {}".format(deterministic_metrics))
        print("deterministic cases: {}".format(deterministic_cases))
    if final_cases is not None:
        print("final metrics: {}".format(final_metrics))
        print("final cases: {}".format(final_cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
