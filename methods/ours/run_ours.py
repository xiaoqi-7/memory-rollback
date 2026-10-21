# methods/ours/run_ours.py
# Top-level orchestrator for the "ours" repair pipeline.
#
# This is the multi-task driver: it loads one task's trace/memory/session
# records, runs the pipeline stage by stage, and writes each stage's artifact
# under a per-task output directory so runs across tasks never collide.
#
# Full replay dependencies can be injected, or constructed from agent config.

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import string
import tempfile
from typing import Any, Dict, List, Optional

import yaml

from .build_dependency_graph import build_dependency_graph, validate_graph, _load_jsonl
from .affected_subgraph import compute_affected_subgraph
from .ours_config import load_config, apply_overrides, OursConfig
from .support_checker import (
    _CHECKABLE_SEMANTIC_TYPES,
    build_support_report,
    find_derived_memory_candidates,
)
from .fault_provenance import find_producing_step, find_stale_window, walk_ancestor_chain
from .rollback_optimizer import build_rollback_plan
from .rollback_executor import execute_rollback_plan, _make_executor_state_agent
from .replay_runtime import ReplayRuntime
from .replay_types import (
    ReplayContext,
    ReplayContextObserver,
    replay_context_to_dict,
)
from .selective_replay import ReplayBackend
from agent.run_agent import AgentRunner


_DEFAULT_OUTPUT_TEMPLATE = (
    "outputs/{method}/{domain}/{fault_types}/{task_id}/{faulty_memory_ids}"
)
_OUTPUT_PLACEHOLDERS = {
    "method", "domain", "fault_types", "task_id", "faulty_memory_ids",
}
_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_case_segment(value: Any, name: str) -> str:
    """Return one safe deterministic path component.

    ``+`` is reserved for joining multiple individually-sanitized values, so
    it is intentionally not accepted from an input value itself.
    """
    if value is None:
        raise ValueError(f"{name} path segment must not be empty")
    raw = str(value)
    if not raw or raw in {".", ".."}:
        raise ValueError(f"{name} path segment must not be {raw!r}")
    raw = raw.replace("/", "_").replace("\\", "_")
    sanitized = _SAFE_SEGMENT_RE.sub("_", raw)
    if not sanitized or sanitized in {".", ".."}:
        raise ValueError(f"{name} path segment is unsafe: {value!r}")
    return sanitized


def _joined_case_segment(
    values: List[Any], name: str, *, missing: Optional[str] = None,
) -> str:
    normalized = [
        missing if missing is not None and (value is None or str(value) == "")
        else value
        for value in values
    ]
    sanitized = sorted({
        _sanitize_case_segment(value, name)
        for value in normalized
        if value is not None and str(value) != ""
    })
    if not sanitized:
        if missing is None:
            raise ValueError(f"{name} must contain at least one value")
        return _sanitize_case_segment(missing, name)
    return "+".join(sanitized)


def resolve_case_output_dir(
    output_template: str,
    method: str = "ours",
    domain: str = None,
    task_id: str = None,
    faulty_memory_ids: Optional[List[str]] = None,
    fault_types: Optional[List[Optional[str]]] = None,
) -> str:
    """Resolve a deterministic, sanitized case output path.

    A literal path has historically been useful in tests and manual debugging;
    when no placeholders are present it is returned byte-for-byte unchanged.
    """
    if not isinstance(output_template, str) or not output_template:
        raise ValueError("output template must be a non-empty string")
    if "{" not in output_template and "}" not in output_template:
        return output_template

    formatter = string.Formatter()
    fields = {
        field_name
        for _, field_name, _, _ in formatter.parse(output_template)
        if field_name is not None
    }
    unsupported = fields - _OUTPUT_PLACEHOLDERS
    if unsupported:
        raise ValueError(
            "unsupported output placeholder(s): " + ", ".join(sorted(unsupported))
        )

    values = {
        "method": _sanitize_case_segment(method or "ours", "method"),
        "domain": _sanitize_case_segment(domain, "domain"),
        "task_id": _sanitize_case_segment(task_id, "task_id"),
        "faulty_memory_ids": _joined_case_segment(
            faulty_memory_ids or [], "faulty_memory_ids"
        ),
        "fault_types": _joined_case_segment(
            fault_types or [], "fault_types", missing="unknown"
        ),
    }
    resolved = output_template.format(**values)

    # The static prefix before the first placeholder defines the configured
    # root. This catches templates whose literal suffix tries to traverse out
    # after otherwise-safe placeholder substitution.
    static_prefix = output_template.split("{", 1)[0]
    root = static_prefix.rstrip(os.sep) or "."
    if static_prefix and not static_prefix.endswith(os.sep):
        root = os.path.dirname(root) or "."
    root_abs = os.path.abspath(root)
    resolved_abs = os.path.abspath(resolved)
    if os.path.commonpath([root_abs, resolved_abs]) != root_abs:
        raise ValueError(
            f"resolved output directory escapes configured root {root!r}"
        )
    return resolved


class CaseOutputTransaction:
    """Stage a complete case beside its destination and atomically commit it."""

    def __init__(self, final_dir: str):
        self.final_dir = os.path.abspath(os.fspath(final_dir))
        self.staging_dir: Optional[str] = None
        self._committed = False

    def __enter__(self) -> str:
        parent = os.path.dirname(self.final_dir)
        os.makedirs(parent, exist_ok=True)
        name = os.path.basename(self.final_dir)
        self.staging_dir = tempfile.mkdtemp(
            prefix=f".{name}.staging-", dir=parent
        )
        return self.staging_dir

    def _remove_staging(self) -> None:
        if self.staging_dir and os.path.lexists(self.staging_dir):
            shutil.rmtree(self.staging_dir)

    def commit(self) -> None:
        if self._committed:
            return
        if not self.staging_dir or not os.path.isdir(self.staging_dir):
            raise RuntimeError("case output transaction is not active")

        backup_dir = None
        if os.path.lexists(self.final_dir):
            parent = os.path.dirname(self.final_dir)
            name = os.path.basename(self.final_dir)
            backup_dir = tempfile.mkdtemp(
                prefix=f".{name}.backup-", dir=parent
            )
            os.rmdir(backup_dir)
            os.replace(self.final_dir, backup_dir)

        try:
            os.replace(self.staging_dir, self.final_dir)
        except BaseException:
            if backup_dir is not None and os.path.lexists(backup_dir):
                os.replace(backup_dir, self.final_dir)
            raise
        else:
            self._committed = True
            if backup_dir is not None:
                shutil.rmtree(backup_dir)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None:
            self._remove_staging()
            return False
        try:
            self.commit()
        except BaseException:
            self._remove_staging()
            raise
        return False


# --------------------------------------------------------------------------- #
# Stage 1: dependency graph construction
# --------------------------------------------------------------------------- #
# We reuse build_dependency_graph.py's private _load_jsonl rather than copying a
# second JSONL reader: it is private but stable, and a single reader keeps the
# load semantics (blank-line skipping, utf-8) identical to the graph builder's
# own CLI. (The alternative -- promoting it to a shared io_utils.py -- was not
# taken because no third caller needs it yet; revisit if one appears.)


def run_dependency_graph(
    task_id: str,
    steps: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    session: List[str],
    output_dir: str,
):
    """Build, validate, serialize, and summarize the stage-1 dependency graph.

    Calls build_dependency_graph() then validate_graph() immediately, writes
    the graph to ``{output_dir}/dependency_graph.json`` via the dataclass's own
    ``to_dict()`` (the same shape build_dependency_graph.py's CLI writes -- one
    serializer, no hand-rolled second code path), prints a summary block, and
    returns the built graph for downstream stages.
    """
    graph = build_dependency_graph(task_id, steps, memories, session)
    warnings = validate_graph(graph)

    os.makedirs(output_dir, exist_ok=True)
    graph_json_path = os.path.join(output_dir, "dependency_graph.json")
    with open(graph_json_path, "w", encoding="utf-8") as fh:
        json.dump(graph.to_dict(), fh, indent=2, ensure_ascii=False)

    # --- summary ---
    n_step = sum(1 for n in graph.nodes.values() if n.node_kind == "step")
    n_mem = sum(1 for n in graph.nodes.values() if n.node_kind == "memory")
    n_user = sum(1 for n in graph.nodes.values() if n.node_kind == "user_input")
    # n_pairs counts distinct (src, dst) edges; a single pair can now carry a SET
    # of labels (e.g. chain + cite), so the per-label tally counts label
    # OCCURRENCES across all pairs (same as build_dependency_graph.py's CLI).
    n_pairs = sum(len(d) for d in graph.forward_edges.values())

    # Per-edge_type breakdown: forward_edges values are now {dst: {edge_type, ...}},
    # so count occurrences of each EdgeType label across every pair's label set.
    edge_type_counts: Dict[str, int] = {}
    for dsts in graph.forward_edges.values():
        for etypes in dsts.values():
            for etype in etypes:
                edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1

    # Lineage sanity check: "derive" edges show whether any memory in this task
    # actually carries lineage at a glance.
    n_lineage = edge_type_counts.get("derive", 0)

    print(f"=== Stage 1: dependency graph for {task_id} ===")
    print(f"steps loaded: {len(steps)}  memories loaded: {len(memories)}")
    print(f"wrote graph JSON to: {graph_json_path}")
    print("\n--- summary ---")
    print(f"  step nodes:       {n_step}")
    print(f"  memory nodes:     {n_mem}")
    print(f"  user_input nodes: {n_user}")
    print(f"  edge pairs:       {n_pairs}")
    for etype in sorted(edge_type_counts):
        print(f"    {etype}: {edge_type_counts[etype]}")
    print(f"  n_turns:          {graph.n_turns}")
    print(f"  lineage (derive) edges: {n_lineage}")
    print(f"  validation warnings: {len(warnings)}")

    print(f"\n--- validation warnings ({len(warnings)}) ---")
    for w in warnings:
        print(f"  ! {w}")
    if not warnings:
        print("  (none)")

    return graph


# --------------------------------------------------------------------------- #
# Later stages -- stubbed until implemented. Each raises loudly and names the
# file that will own it, so main() stops at the first unimplemented stage
# instead of silently emitting a partial pipeline result.
# --------------------------------------------------------------------------- #

def run_affected_subgraph(
    graph,
    faulty_memory_ids: List[str],
    provenance: Dict[str, Any],
    steps: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, Any]:
    """Compute the affected subgraph for the faulty seeds, serialize, summarize.

    Calls affected_subgraph.compute_affected_subgraph() with the WHOLE
    faulty_memory_ids list, the per-memory ``provenance`` dict from the
    fault-provenance stage, AND ``steps`` (the ordered trace-step dicts, same
    list stage 1/2 already consume) — needed for the stale-seed temporal filter's
    {step_id: position} lookup. Writes the returned payload to
    ``{output_dir}/affected_nodes.json`` by json.dump-ing it directly (one
    serializer, mirroring stage 1's to_dict() pattern — no hand-rolled second
    writer), prints a summary block structured like stage 1's, and returns the
    payload for downstream stages.
    """
    report = compute_affected_subgraph(
        graph, faulty_memory_ids, provenance, steps
    )

    os.makedirs(output_dir, exist_ok=True)
    affected_json_path = os.path.join(output_dir, "affected_nodes.json")
    with open(affected_json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\n=== Stage 3: affected subgraph for {graph.task_id} ===")
    print(f"faulty memory seeds: {report['seed_memory_ids']}")
    print(f"resolved seed ids:   {report['resolved_seed_ids']}")
    print(f"wrote affected nodes JSON to: {affected_json_path}")
    print("\n--- summary ---")
    print(f"  total affected nodes: {report['total_affected']}")
    print(f"  seed reachable from another seed: "
          f"{report['seed_included_in_affected']}")
    print("  counts by type:")
    counts = report["counts_by_type"]
    for t in sorted(counts):
        print(f"    {t:18s} {counts[t]}")
    print(f"  affected edges: {len(report['affected_edges'])}")

    return report


# --------------------------------------------------------------------------- #
# Later stages -- stubbed until implemented. Each raises loudly and names the
# file that will own it, so main() stops at the first unimplemented stage
# instead of silently emitting a partial pipeline result.
# --------------------------------------------------------------------------- #

def run_support_check(
    graph,
    affected_ids: List[str],
    faulty_memory_ids: List[str],
    output_dir: str,
    config: OursConfig,
) -> Dict[str, Any]:
    """Run the independent-support check over stage 2's affected nodes.

    Uses ``config.support_check``'s settings to compute unchanged step verdicts
    plus derived-memory verdicts, serializes the structured report, and prints
    a summary consistent with running support_checker.py standalone.

    If ``config.ablation.skip_support_check`` is True ("Ours w/o independent
    support check", section 15) the independent-support check is NOT run at
    all: every checkable affected step and every active derived-memory candidate
    is marked unsupported directly in the same structured report shape.
    """
    sc = config.support_check

    if config.ablation.skip_support_check:
        # Ablation arm: do NOT consult support_checker.check_all_nodes. Mark
        # every checkable affected step and derived-memory candidate without
        # support unconditionally.
        step_verdicts: Dict[str, bool] = {
            node_id: False
            for node_id in affected_ids
            if graph.node(node_id) is not None
            and graph.node(node_id).semantic_type in _CHECKABLE_SEMANTIC_TYPES
        }
        candidates = find_derived_memory_candidates(
            graph, faulty_memory_ids, affected_ids
        )
        report = {
            "step_verdicts": step_verdicts,
            "derived_memory_verdicts": {
                memory_id: {"supported": False, "support_path": []}
                for memory_id in candidates
            },
        }
        print(
            f"[ablation: skip_support_check] {len(step_verdicts)} steps and "
            f"{len(candidates)} derived memories marked without support"
        )
    else:
        report = build_support_report(
            graph,
            faulty_memory_ids,
            affected_ids,
            use_trust_score=sc.use_trust_score,
            trust_threshold=sc.trust_threshold,
        )

    os.makedirs(output_dir, exist_ok=True)
    support_check_path = os.path.join(output_dir, "support_check.json")
    with open(support_check_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    step_verdicts = report["step_verdicts"]
    derived_verdicts = report["derived_memory_verdicts"]
    total = len(step_verdicts)
    n_supported = sum(1 for v in step_verdicts.values() if v)
    n_unsupported = total - n_supported

    print(f"\n=== Stage 4: independent-support check (task {graph.task_id}) ===")
    print(f"  use_trust_score: {sc.use_trust_score}")
    print(f"  trust_threshold: {sc.trust_threshold}")
    print(f"  nodes checked:            {total}")
    print(f"  with trusted support:     {n_supported}  (keep — clean backing)")
    print(
        f"  without trusted support:  {n_unsupported}  "
        "(act per rollback policy)"
    )
    print(f"  derived candidates:       {len(derived_verdicts)}")
    print(
        "  supported derived:        "
        f"{sum(1 for verdict in derived_verdicts.values() if verdict['supported'])}"
    )
    print(f"  wrote: {support_check_path}")

    return report

def run_fault_provenance(
    graph,
    faulty_memory_ids: List[str],
    steps: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, Any]:
    """For each faulty memory seed, gather its producing step, stale-window
    structural check, and ancestor chain. Returns
    {memory_id: {"producing_step_id": ..., "stale_window": ...,
    "ancestor_chain": [...]}} keyed by faulty memory id — pure structural
    facts, no fault/repair judgment (that is rollback_optimizer.py's job, which
    consumes this output later).

    This stage runs BEFORE affected_subgraph (stage 3), which now needs this
    output to resolve its seeds (the failed-mutation-step / producing-step /
    memory-node priority in affected_subgraph.resolve_seed_id).

    Writes the result to ``{output_dir}/provenance.json`` (one serializer,
    json.dump directly, matching the other stages' pattern), prints a per-memory
    summary, and returns the dict for the affected-subgraph stage.

    A faulty_memory_id that is not a node in ``graph`` at all is skipped with a
    printed warning rather than crashing the whole stage — mirroring
    affected_subgraph.py's "one bad id should not kill the whole report"
    philosophy. fault_provenance.py's individual functions raise KeyError on a
    missing node, so that KeyError is caught here per-id.
    """
    provenance: Dict[str, Any] = {}

    print(f"\n=== Stage 2: fault provenance (task {graph.task_id}) ===")
    for memory_id in faulty_memory_ids:
        try:
            producing_step_id = find_producing_step(graph, memory_id)
            stale_window = find_stale_window(graph, memory_id, steps)
            ancestor_chain = walk_ancestor_chain(graph, memory_id)
        except KeyError:
            print(
                f"  ! skipping {memory_id}: not a node in graph "
                f"{graph.task_id} (one bad id does not abort the stage)"
            )
            continue

        provenance[memory_id] = {
            "producing_step_id": producing_step_id,
            "stale_window": stale_window,
            "ancestor_chain": ancestor_chain,
        }

        # --- per-memory summary ---
        print(f"  memory {memory_id}:")
        if producing_step_id is None:
            print("    producing step: none — root memory")
        else:
            print(f"    producing step: {producing_step_id}")

        if stale_window is None:
            print("    stale window: none found")
        else:
            print(
                f"    stale window: mutation={stale_window['mutation_step_id']} "
                f"took_effect={stale_window['took_effect']} "
                f"signal={stale_window['signal']} "
                f"window={stale_window['window_step_ids']}"
            )

        if not ancestor_chain:
            print("    ancestor chain: none")
        else:
            ancestor_ids = [a["memory_id"] for a in ancestor_chain]
            print(
                f"    ancestor chain: {len(ancestor_ids)} ancestor(s): "
                f"{ancestor_ids}"
            )

    os.makedirs(output_dir, exist_ok=True)
    provenance_path = os.path.join(output_dir, "provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, ensure_ascii=False)
    print(f"  wrote: {provenance_path}")

    return provenance

def run_rollback_optimizer(
    graph,
    task_id: str,
    faulty_memory_ids: List[str],
    affected_nodes: Dict[str, Any],
    support_report: Dict[str, Any],
    provenance: Dict[str, Dict[str, Any]],
    output_dir: str,
    config: OursConfig,
    case_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Combine stage 2/3/4's outputs into a concrete rollback plan.

    Calls rollback_optimizer.build_rollback_plan() with the structural facts
    (affected subgraph), the support verdicts, and the provenance, passing the
    ablation section (``config.ablation`` carries the rollback-policy flags).
    Writes the plan to ``{output_dir}/rollback_plan.json`` (one serializer,
    json.dump directly, matching every other stage's pattern), prints a summary
    block mirroring rollback_optimizer.py's own CLI summary (each action list's
    full contents, not just counts — small enough per task), and returns the
    plan dict.
    """
    plan = build_rollback_plan(
        graph,
        task_id,
        faulty_memory_ids,
        affected_nodes,
        support_report,
        provenance,
        config.ablation,
    )
    if case_id is not None:
        plan["case_id"] = case_id

    os.makedirs(output_dir, exist_ok=True)
    rollback_plan_path = os.path.join(output_dir, "rollback_plan.json")
    with open(rollback_plan_path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, ensure_ascii=False)

    print(f"\n=== Stage 5: rollback plan (task {task_id}) ===")
    print(f"  faulty_memory_ids:     {plan['faulty_memory_ids']}")
    print(f"  delete_memory_ids:     {plan['delete_memory_ids']}")
    print(f"  quarantine_memory_ids: {plan['quarantine_memory_ids']}")
    print(f"  invalidate_claim_ids:  {plan['invalidate_claim_ids']}")
    print(f"  replay_step_ids:       {plan['replay_step_ids']}")
    print(f"  preserve_step_ids:     {plan['preserve_step_ids']}")
    print(f"  redundant_step_ids:    {plan['redundant_step_ids']}")
    print(f"  suspicious_step_ids:   {plan['suspicious_step_ids']}")
    print(f"  reason: {plan['reason']}")
    print(f"  wrote: {rollback_plan_path}")

    return plan

def run_rollback_executor(
    graph,
    plan: Dict[str, Any],
    steps: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    session: List[str],
    output_dir: str,
    *,
    runtime: Any = None,
    backend: Any = None,
    agent_config_path: str = None,
    llm_repair_decision_calls_by_operation: Optional[Dict[str, int]] = None,
    replay_context_observer: Optional[ReplayContextObserver] = None,
) -> Dict[str, Any]:
    """Run setup and canonical backend-driven replay for stage 6.

    ``runtime`` and ``backend`` may be omitted; in that case ``agent_config_path``
    is required and used to auto-construct them from the domain's existing agent
    config. Tool resolution reuses agent/run_agent.py's AgentRunner exactly — the
    runner loads the domain's tool callables from the config's
    ``available_tools_source`` / ``domain``, and we read its ``.tools`` attribute.
    A missing runtime is built via a temp executor-state agent wrapped in
    ``ReplayRuntime.from_agent``; a missing backend defaults to the production
    ``ReplayBackend``.
    """
    if runtime is None or backend is None:
        if agent_config_path is None:
            raise TypeError(
                "run_rollback_executor requires agent_config_path when runtime "
                "and/or backend are not supplied (both are required together to "
                "auto-construct the replay dependencies)"
            )
        runner = AgentRunner(agent_config_path, mode="clean")
        if runtime is None:
            state_agent = _make_executor_state_agent(
                tools=runner.tools,
                config=runner.config,
            )
            runtime = ReplayRuntime.from_agent(state_agent)
        if backend is None:
            backend = ReplayBackend()
    agent = _make_executor_state_agent(
        tools=dict(runtime.tools),
        config=dict(runtime.config),
    )
    print(f"\n=== Stage 6: rollback executor ===")
    return execute_rollback_plan(
        agent, graph, plan, steps, memories, session, output_dir,
        runtime=runtime,
        backend=backend,
        llm_repair_decision_calls_by_operation=(
            llm_repair_decision_calls_by_operation
        ),
        replay_context_observer=replay_context_observer,
    )


_ABLATION_METHOD_FIELDS = (
    "skip_support_check",
    # "skip_derived_cleanup" is intentionally not exposed as a method-level
    # ablation entry. The underlying rollback-optimizer rule remains
    # implemented for internal experiments, but it is no longer part of the
    # public/main ablation surface.
    # "skip_derived_cleanup",
    "skip_rollback_optimizer",
    "skip_selective_replay",
)


def _method_name_from_ablation(config: OursConfig) -> str:
    names = [
        "no_" + field.removeprefix("skip_")
        for field in _ABLATION_METHOD_FIELDS
        if getattr(config.ablation, field, False)
    ]
    return "+".join(names) if names else "ours"


def _plan_for_repaired_result(plan: Dict[str, Any], config: OursConfig) -> Dict[str, Any]:
    """Return an executor plan whose result payload names the active method."""
    result_plan = dict(plan)
    result_plan["method"] = _method_name_from_ablation(config)
    return result_plan


# --------------------------------------------------------------------------- #
# Load + filter
# --------------------------------------------------------------------------- #

def _filter_task(records: List[Dict[str, Any]], task_id: str) -> List[Dict[str, Any]]:
    """Records whose task_id matches (mirrors build_dependency_graph.py's CLI)."""
    return [r for r in records if r.get("task_id") == task_id]


def _load_session(path: str, task_id: str) -> List[str]:
    """Load the session utterance list for ``task_id`` (session[0] is turn 1)."""
    session_recs = _filter_task(_load_jsonl(path), task_id)
    return session_recs[0].get("session", []) if session_recs else []


def _registry_original_task_id(session_path: str, task_id: str) -> Optional[str]:
    data_root = os.path.dirname(os.path.dirname(os.path.abspath(session_path)))
    registry_path = os.path.join(data_root, "registry.jsonl")
    if not os.path.isfile(registry_path):
        return None
    with open(registry_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("task_id") == task_id:
                original_task_id = record.get("original_task_id")
                return original_task_id if isinstance(original_task_id, str) else None
    return None


def _load_agent_domain(config_path: str) -> str:
    """Read only case metadata; runtime/tool construction remains stage 6's job."""
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError("agent config must contain a mapping")
    domain = config.get("domain")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("agent config does not provide a non-empty domain")
    return domain


def _load_and_validate_case_records(
    memories_path: str,
    steps_path: str,
    session_path: str,
    task_id: str,
    faulty_memory_ids: List[str],
) -> tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[Optional[str]]
]:
    """Load one task and return its records plus selected fault types."""
    all_memories = _load_jsonl(memories_path)
    all_steps = _load_jsonl(steps_path)
    all_sessions = _load_jsonl(session_path)
    memories = _filter_task(all_memories, task_id)
    steps = _filter_task(all_steps, task_id)
    session_records = _filter_task(all_sessions, task_id)
    if not session_records:
        original_task_id = _registry_original_task_id(session_path, task_id)
        if original_task_id:
            session_records = _filter_task(all_sessions, original_task_id)

    for label, selected, loaded in (
        ("memory", memories, all_memories),
        ("trace-step", steps, all_steps),
        ("session", session_records, all_sessions),
    ):
        if not selected:
            loaded_ids = sorted({
                str(record.get("task_id")) for record in loaded
                if record.get("task_id") is not None
            })
            raise ValueError(
                f"task_id {task_id!r} does not match any loaded {label} "
                f"records (loaded task_ids: {loaded_ids})"
            )

    memory_by_id = {
        record.get("memory_id"): record
        for record in memories
        if isinstance(record.get("memory_id"), str)
    }
    missing = [memory_id for memory_id in faulty_memory_ids
               if memory_id not in memory_by_id]
    if missing:
        elsewhere = sorted({
            str(record.get("task_id"))
            for record in all_memories
            if record.get("memory_id") in missing
            and record.get("task_id") is not None
        })
        detail = f"; found under task_id(s) {elsewhere}" if elsewhere else ""
        raise ValueError(
            f"faulty_memory_id(s) missing from task {task_id!r}: {missing}{detail}"
        )

    fault_types = [
        record.get("fault_type")
        if isinstance(record.get("fault_type"), str)
        and record.get("fault_type") else None
        for memory_id in faulty_memory_ids
        for record in (memory_by_id[memory_id],)
    ]
    session = session_records[0].get("session", [])
    if not isinstance(session, list):
        raise ValueError(f"session for task {task_id!r} must be a list")
    return steps, memories, session, fault_types


def _resolve_replay_context_path(
    requested_path: str, staging_dir: str, task_id: str,
) -> str:
    """Resolve a replay-context target and prove it remains inside staging."""
    try:
        rendered = requested_path.format(
            output_dir=staging_dir,
            task_id=_sanitize_case_segment(task_id, "task_id"),
        )
    except KeyError as exc:
        raise ValueError(
            f"unsupported replay-context placeholder: {exc.args[0]}"
        ) from exc
    candidate = rendered if os.path.isabs(rendered) else os.path.join(
        staging_dir, rendered
    )
    staging_abs = os.path.abspath(staging_dir)
    candidate_abs = os.path.abspath(candidate)
    if os.path.commonpath([staging_abs, candidate_abs]) != staging_abs:
        raise ValueError("replay_context_output must remain inside the case directory")
    if candidate_abs == staging_abs:
        raise ValueError("replay_context_output must name a file")
    return candidate_abs


def _make_replay_context_observer(
    output_path: str,
    mirror_path: Optional[str] = None,
) -> ReplayContextObserver:
    """Print replay contexts and persist every observed operation as JSON."""
    records: List[Dict[str, Any]] = []
    output_paths = list(dict.fromkeys(
        path for path in (output_path, mirror_path) if path
    ))
    for path in output_paths:
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def write_records() -> None:
        for path in output_paths:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, ensure_ascii=False, indent=2, default=str)

    # Create a valid empty artifact even when a plan has no replay operations.
    write_records()

    def replay_context_observer(
        operation: str, context: ReplayContext
    ) -> None:
        serialized_context = replay_context_to_dict(context)
        # print(
        #     f"[replay-context] operation={operation} "
        #     f"step={context.step_id}"
        # )
        # print(
        #     json.dumps(
        #         serialized_context,
        #         ensure_ascii=False,
        #         indent=2,
        #         default=str,
        #     )
        # )
        records.append({
            "operation": operation,
            "context": serialized_context,
        })
        write_records()

    return replay_context_observer

def _build_affected_nodes_with_source(
    affected_nodes: Dict[str, Any],
    faulty_memory_ids: List[str],
    provenance: Dict[str, Dict[str, Any]],
) -> List[str]:
    '''The full taint set for support_checker.py: every affected node,
    PLUS the faulty memories themselves, PLUS each fault's resolved
    source step (the producing step, or the failed mutation step for
    the stale-and-failed case) — all of which are deliberately EXCLUDED
    from affected_node_ids by affected_subgraph.py's "cause, not
    effect" design, but must still count as tainted for a predecessor-
    validity check (a checkable node citing the faulty memory or its
    source step directly must not be treated as having clean backing).
    '''
    affected_nodes_with_source = set(affected_nodes["affected_node_ids"])
    affected_nodes_with_source.update(faulty_memory_ids)
    for mid in faulty_memory_ids:
        prov = provenance.get(mid, {})
        stale_window = prov.get("stale_window")
        if stale_window is not None and stale_window.get("took_effect") is False:
            affected_nodes_with_source.add(stale_window["mutation_step_id"])
        producing_step_id = prov.get("producing_step_id")
        if producing_step_id is not None:
            affected_nodes_with_source.add(producing_step_id)
    return sorted(affected_nodes_with_source)

# --------------------------------------------------------------------------- #
# CLI / entry point
# --------------------------------------------------------------------------- #
def run(
    args,
    steps,
    memories,
    session,
    output_dir,
    config,
    *,
    runtime=None,
    backend=None,
    agent_config_path=None,
    replay_context_observer: Optional[ReplayContextObserver] = None,
):
    # Stage 1: dependency graph.
    graph = run_dependency_graph(
        args.task_id, steps, memories, session, output_dir
    )

    # Stage 2: structural provenance for each faulty memory — MOVED
    # BEFORE affected_subgraph, since affected_subgraph now needs this to
    # resolve its seeds (failed-mutation-step / producing-step / memory-node
    # priority). No fault judgment here — rollback_optimizer.py interprets these
    # facts.
    provenance = run_fault_provenance(
        graph, args.faulty_memory_ids, steps, output_dir
    )

    # Stage 3: affected subgraph from the oracle faulty seeds, now
    # seeded via provenance and filtered against trace order (steps) for the
    # stale-and-failed case.
    affected_nodes = run_affected_subgraph(
        graph, args.faulty_memory_ids, provenance, steps, output_dir
    )

    # Stage 4: independent-support check over the affected claims
    # (config-driven).
    affected_nodes_with_source = _build_affected_nodes_with_source(
        affected_nodes, args.faulty_memory_ids, provenance
    )
    verdicts = run_support_check(
        graph, affected_nodes_with_source, args.faulty_memory_ids, output_dir, config
    )

    # Stage 5: combine stage 2/3/4's outputs into a concrete rollback plan.
    # Ablation flags live in config.ablation and are set through the same
    # --set mechanism as the rest of the pipeline-level knobs.
    rollback_plan = run_rollback_optimizer(
        graph, args.task_id, args.faulty_memory_ids, affected_nodes,
        verdicts, provenance, output_dir, config,
        case_id=getattr(args, "case_id", None),
    )
    rollback_plan = _plan_for_repaired_result(rollback_plan, config)

    # Stage 6: setup_replay_state -> replay_chain -> ReplayBackend operations.
    replay_result = run_rollback_executor(
        graph, rollback_plan, steps, memories, session, output_dir,
        runtime=runtime,
        backend=backend,
        agent_config_path=agent_config_path,
        llm_repair_decision_calls_by_operation={},
        replay_context_observer=replay_context_observer,
    )

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the 'ours' repair pipeline for one task (stage 1 wired, "
                    "later stages stubbed)."
    )
    parser.add_argument("--memories", required=True, help="memories JSONL path")
    parser.add_argument("--steps", required=True, help="trace steps JSONL path")
    parser.add_argument(
        "--session", required=True, help="tasks_full_session JSONL path"
    )
    parser.add_argument("--task_id", required=True, help="task_id to filter to")
    parser.add_argument(
        "--faulty_memory_ids", required=True, nargs="+",
        help="the oracle faulty memory id(s) injected into this task (1-3 per "
             "task). For the main pipeline these are GIVEN (the injected "
             "ground-truth seeds), not something this stage detects.",
    )
    parser.add_argument(
        "--output_dir",
        default=_DEFAULT_OUTPUT_TEMPLATE,
        help="case output template supporting {method}, {domain}, "
             "{fault_types}, {task_id}, and {faulty_memory_ids}. A literal "
             "path is preserved exactly (default: " + _DEFAULT_OUTPUT_TEMPLATE + ").",
    )
    parser.add_argument(
        "--replay_context_output",
        default="replay_context.json",
        help="filename or case-relative JSON path for observed replay contexts; "
             "{task_id} and {output_dir} are supported and the resolved path "
             "must remain inside the active case transaction.",
    )
    parser.add_argument(
        "--config",
        default="configs/ours_base.yaml",
        help="path to the YAML base config (default: configs/ours_base.yaml).",
    )
    parser.add_argument(
        "--agent_config",
        required=True,
        help="path to the domain agent config such as configs/travel.yaml, the "
             "same file run_agent.py uses -- supplies available_tools_source and "
             "domain so stage 6 can resolve real tool callables.",
    )
    parser.add_argument(
        "--set",
        dest="set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted-path override applied on top of --config, e.g. "
             "'support_check.use_trust_score=false'. Repeatable. This is the "
             "ONLY mechanism for setting support-check / ablation values, so "
             "there is exactly one way to set each value.",
    )
    args = parser.parse_args(argv)

    # Load the base config, then apply any dotted-path --set overrides on top.
    config = load_config(args.config)
    if args.set:
        overrides = {}
        for item in args.set:
            if "=" not in item:
                parser.error(
                    f"--set value {item!r} must be 'section.field=value'"
                )
            key, _, value = item.partition("=")
            overrides[key] = value
        config = apply_overrides(config, overrides)

    # Case identity is intentionally resolved only after all its authoritative
    # metadata has been loaded and validated.
    args.faulty_memory_ids = sorted(set(args.faulty_memory_ids))
    steps, memories, session, fault_types = _load_and_validate_case_records(
        args.memories,
        args.steps,
        args.session,
        args.task_id,
        args.faulty_memory_ids,
    )
    domain = _load_agent_domain(args.agent_config)
    output_dir = resolve_case_output_dir(
        args.output_dir,
        method="ours",
        domain=domain,
        task_id=args.task_id,
        faulty_memory_ids=args.faulty_memory_ids,
        fault_types=fault_types,
    )
    args.case_id = resolve_case_output_dir(
        "{domain}/{fault_types}/{task_id}/{faulty_memory_ids}",
        method="ours",
        domain=domain,
        task_id=args.task_id,
        faulty_memory_ids=args.faulty_memory_ids,
        fault_types=fault_types,
    )

    with CaseOutputTransaction(output_dir) as staging_dir:
        replay_context_output = _resolve_replay_context_path(
            args.replay_context_output, staging_dir, args.task_id
        )
        canonical_replay_context = os.path.join(
            staging_dir, "replay_context.json"
        )
        replay_context_observer = _make_replay_context_observer(
            replay_context_output,
            mirror_path=canonical_replay_context,
        )
        run(
            args, steps, memories, session, staging_dir, config,
            agent_config_path=args.agent_config,
            replay_context_observer=replay_context_observer,
        )
    print(f"  committed case output: {output_dir}")


if __name__ == "__main__":
    main()
