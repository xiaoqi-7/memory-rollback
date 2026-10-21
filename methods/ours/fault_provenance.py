# methods/ours/fault_provenance.py
"""Structural provenance lookups for a (faulty) memory.

Pure structure, no fault judgment, no repair decisions
======================================================
This module answers four structural questions and NOTHING else:

* :func:`find_producing_step` — *which step produced this memory* (one hop
  backward over a ``produce`` edge).
* :func:`find_stale_window` — *what is this memory's mutation context* (was it
  targeted by an update/delete/consolidate step, and did that mutation take
  effect).
* :func:`walk_ancestor_chain` — *what is this memory's lineage* (its
  ``derived_from`` ancestor chain, oldest first).
* :func:`find_tool_action_for_observation` — *which tool_action produced a
  tool_observation a given step cites* (one hop backward over a ``chain`` edge
  from a cited ``tool_observation``). The only lookup keyed on a STEP rather
  than a memory.

It mirrors ``affected_subgraph.py``'s "pure reachability, no
pruning" boundary and ``support_checker.py``'s "verdicts only, no actions"
boundary: ``fault_provenance.py`` never decides whether a memory *is* faulty,
never decides whether something counts as the "stale" fault type, and never
decides what to *do* about any of it.

``rollback_optimizer.py`` (not yet implemented) is the SOLE consumer. It
combines these raw structural facts with (a) the faulty/affected set
(``affected_subgraph.py``) and (b) ``support_checker.py``'s support verdicts to
produce actual repair decisions. The interpretation — "this ancestor is itself
in my faulty set", "this un-applied mutation means the memory is stale", "this
producing step must be replayed" — all lives in that consumer, not here.

How ``find_stale_window`` decides ``took_effect`` (two clean rules)
===================================================================
The mutation steps targeting a memory are found from the GRAPH itself: a
memory's successor edges labelled ``delete`` / ``update`` / ``consolidate``
(the three memory-consumption passes each run memory -> mutating step) name
every step that mutated it — no re-scan of every step's
``invalidated_memory_ids`` is needed. Graph edges carry no ordering, so those
step ids are ordered by their position in the supplied ``steps_in_order``.
Then:

* **More than one mutation step** -> the FIRST one (by trace order) did NOT take
  effect *by construction*: had it applied, no later mutation would have found
  this memory to target. ``took_effect=False``, ``signal="multiple_mutations"``.
* **Exactly one mutation step** -> read the memory's recorded status. ``active``
  means the mutation never changed anything (``took_effect=False``,
  ``signal="status_active"``); any other status, including a missing one,
  counts as the mutation having registered (``took_effect=True``,
  ``signal="status_changed"``).

This supersedes the older dual-signal scheme (per-step-type status expectation
plus a trace-behavior-consistency / memory-read re-surfacing check), which is
gone entirely.

Design boundaries (deliberate, stated so they are not eroded later)
===================================================================
* **No faulty/affected parameter.** None of the three public functions take a
  ``faulty_memory_ids`` or ``affected_node_ids`` argument. They report
  structure; cross-referencing that structure against the faulty/affected set
  is the caller's job. ``walk_ancestor_chain`` hands back each ancestor's raw
  record precisely so the caller can run its OWN trust/status/faulty checks
  without this module making the judgment.
* **``find_stale_window`` takes ``steps_in_order`` explicitly.** The mutation
  steps come from graph edges, but ordering them — and "between the producing
  step and the mutation step" — is a temporal/trace-order concept the graph
  does not store (it keeps only per-step prev/next adjacency and per-step turn
  numbers, no global step ordering), so the trace-order list must be supplied
  by the caller rather than re-derived from the graph.
* **Permissive logging, narrow raising.** Mirroring
  ``build_dependency_graph.py``: unusual-but-legitimate structural situations
  (a root memory with no producing step, more than one producing step, a memory
  never targeted by any mutation, a mutation step missing from the supplied
  trace order) are logged at DEBUG/WARNING and reported in the return value,
  never raised on. A genuine caller bug — a ``memory_id`` that is not a node in
  the graph at all — raises ``KeyError`` so it is loud.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List, Optional

try:  # importable both as ``methods.ours.fault_provenance`` and as a script
    from .graph_schema import DependencyGraph
except ImportError:  # pragma: no cover - direct-script fallback
    from graph_schema import DependencyGraph

logger = logging.getLogger(__name__)

# Edge-type labels this module keys off (the .value strings DependencyGraph
# stores on edges; see graph_schema.EdgeType). Each edge now carries a SET of
# labels, so these are tested with ``in`` rather than ``==``. PRODUCE runs
# producing-step -> memory (the produce pass); DERIVE runs ancestor-memory ->
# derived-memory (the derive pass). The graph also now carries a memory->memory
# "supersede" edge (old memory -> its replacement, from each memory's own
# supersedes field); walk_ancestor_chain could be extended to follow it for a
# future "what replaced this memory" lookup, but this pass does not need it —
# noting only that the edge exists and is available without further graph
# changes.
_PRODUCE = "produce"
_DERIVE = "derive"
_SUPERSEDE = "supersede"
# CHAIN runs step -> step in trace order (the reasoning chain threaded through
# next_step_ids: claim->plan->tool_action->tool_observation->final_answer; see
# graph_schema.EdgePassSpec "chain"). A tool_observation's CHAIN predecessor is
# therefore the tool_action that produced it, which is the direction
# find_tool_action_for_observation walks BACKWARD.
_CHAIN = "chain"

# Semantic type (graph_schema.SemanticType) marking a tool result node. A step's
# used_ids may cite such a node directly; find_tool_action_for_observation keeps
# exactly those cited ids.
_TOOL_OBSERVATION = "tool_observation"

# Memory status vocabulary (instruction_file.md section 6.1:
# "active | quarantined | deleted | superseded"). Read from a real record's
# "status" field rather than guessed. In the single-mutation case,
# find_stale_window treats ``active`` as the ONE status meaning "the mutation
# did not take effect" (the memory still reads as live); any other status —
# including a missing one — counts as the mutation having registered.
_STATUS_ACTIVE = "active"

# Edge labels (graph_schema.EdgeType values) that mark a memory-mutating step.
# The delete/update/consolidate passes each wire memory -> mutating-step, so a
# memory's SUCCESSOR edges carrying any of these name a step that mutated it.
# find_stale_window finds its mutation steps via these edges (produced by those
# passes) rather than re-scanning every step's ``invalidated_memory_ids``.
_MUTATION_EDGE_TYPES = frozenset({"delete", "update", "consolidate"})


def _require_node(graph: DependencyGraph, memory_id: str) -> None:
    """Raise ``KeyError`` if ``memory_id`` is not a node in ``graph``.

    A bad id from the caller should be loud (mirrors support_checker's claim_id
    guard); legitimate-but-unusual structure is handled by the callers' own
    logging, not here.
    """
    if graph.node(memory_id) is None:
        raise KeyError(
            f"memory_id {memory_id!r} is not a node in graph {graph.task_id!r}"
        )


# --------------------------------------------------------------------------- #
# 1. Producing step — one-hop backward over a "produce" edge
# --------------------------------------------------------------------------- #

def find_producing_step(graph: DependencyGraph, memory_id: str) -> Optional[str]:
    """Return the step_id that produces ``memory_id`` (one-hop backward), or None.

    Reads ``graph.predecessor_edges(memory_id)`` and keeps the sources whose
    label set contains ``"produce"`` (the produce pass: a memory_write/update/
    consolidate step listing this memory in its ``generated_memory_ids``).

    * Returns ``None`` if there is NO producing step — a root/seed memory with no
      upstream step is a legitimate graph shape (logged at DEBUG), not an error.
    * Returns the single producing step_id in the normal case.
    * If MORE THAN ONE producing step is found (the produce pass's construction
      should make this impossible, but invariants are not assumed to hold
      blindly), logs a WARNING and returns the first in sorted order —
      deterministic, never an arbitrary dict-iteration pick, and never a raise.

    Raises ``KeyError`` if ``memory_id`` is not a node in ``graph`` at all.
    """
    _require_node(graph, memory_id)
    producers = sorted(
        src
        for src, etypes in graph.predecessor_edges(memory_id).items()
        if _PRODUCE in etypes
    )
    if not producers:
        logger.debug(
            "memory %s has no producing step (root/seed memory)", memory_id
        )
        return None
    if len(producers) > 1:
        logger.warning(
            "memory %s has %d producing steps %s; expected 1 (the produce pass "
            "should emit a single 'produce' edge). Returning the first in sorted "
            "order.",
            memory_id,
            len(producers),
            producers,
        )
    return producers[0]


# --------------------------------------------------------------------------- #
# 2. Stale window — the trace span between production and a later mutation
# --------------------------------------------------------------------------- #

def find_stale_window(
    graph: DependencyGraph,
    memory_id: str,
    steps_in_order: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Report the raw structural situation of a mutation targeting ``memory_id``.

    Structural fact only — does NOT decide whether this means the memory IS
    "stale" in the fault-injection sense. The caller (rollback_optimizer.py)
    interprets a ``took_effect == False`` window as the stale fault type or not.

    The mutation steps are found via GRAPH EDGES, not a trace re-scan: a memory's
    SUCCESSOR edges labelled ``delete`` / ``update`` / ``consolidate`` (emitted by
    the delete/update/consolidate passes, each running memory -> mutating step)
    name every step that mutated this memory. Graph edges carry no ordering, so
    those step ids are then ordered by their position in ``steps_in_order`` (the
    one place trace order lives).

    Two clean rules decide ``took_effect``:

    * **Multiple mutations.** If more than one step mutated this memory, the
      FIRST one (by trace order) did NOT take effect *by construction* — had it
      applied, there would be nothing left for a later mutation to target. So
      ``took_effect=False``, ``signal="multiple_mutations"``, and the returned
      window is built around that FIRST mutation. The later mutation(s) are a
      SEPARATE, subsequent repair situation the caller can rediscover on its own
      (this function neither recurses nor reports more than one window per call).
    * **Single mutation.** Read ``memory_id``'s recorded status (``memory_status``,
      falling back to ``raw["status"]``). ``active`` means the lone mutation never
      changed anything -> ``took_effect=False``, ``signal="status_active"``. Any
      other status — including a missing one, since absence is no positive
      evidence of failure (logged at DEBUG) — means it registered ->
      ``took_effect=True``, ``signal="status_changed"``.

    Returns ``None`` if NO ``delete``/``update``/``consolidate`` edge leaves
    ``memory_id`` (the memory was never targeted by any mutation; logged at
    DEBUG, not an error).

    Otherwise returns::

        {
          "mutation_step_id":   str,   # the chosen mutation step (FIRST in trace
                                       # order when several exist)
          "mutation_step_type": str,   # memory_update / memory_delete / memory_consolidate
          "took_effect":        bool,  # per the two rules above
          "signal":             str,   # "multiple_mutations" | "status_active" |
                                       #   "status_changed" — which rule decided
          "window_step_ids":    List[str],  # steps strictly BETWEEN the producing
                                            # step and the mutation step, in trace
                                            # order, EXCLUSIVE of both endpoints
        }

    The window runs from :func:`find_producing_step` (exclusive) to the mutation
    step (exclusive). If there is no producing step (a root memory), the window
    starts at the BEGINNING of ``steps_in_order`` instead of raising (logged at
    DEBUG). Raises ``KeyError`` if ``memory_id`` is not a node in ``graph``.
    """
    _require_node(graph, memory_id)

    # 1. Mutation steps via graph edges: successors reached by a delete/update/
    #    consolidate label.
    mutation_step_ids = [
        dst
        for dst, etypes in graph.successor_edges(memory_id).items()
        if etypes & _MUTATION_EDGE_TYPES
    ]
    if not mutation_step_ids:
        logger.debug(
            "memory %s has no delete/update/consolidate edge; it was never "
            "targeted by a mutation. No stale window.",
            memory_id,
        )
        return None

    # 2. Order them by trace position (edges carry no ordering; steps_in_order
    #    does). A mutation step missing from steps_in_order means the graph and
    #    the supplied trace disagree — surface it (WARNING) and sort it last
    #    rather than crashing.
    step_index = {
        s.get("step_id"): i for i, s in enumerate(steps_in_order)
    }

    def _trace_pos(step_id: str) -> float:
        pos = step_index.get(step_id)
        if pos is None:
            logger.warning(
                "mutation step %s targeting memory %s is not in steps_in_order; "
                "the graph and the supplied trace order disagree. Sorting it "
                "last.",
                step_id,
                memory_id,
            )
            return float("inf")
        return pos

    mutation_step_ids.sort(key=_trace_pos)
    chosen_step_id = mutation_step_ids[0]
    chosen_node = graph.node(chosen_step_id)
    mutation_step_type = chosen_node.step_type if chosen_node is not None else None

    # 3. took_effect via the two rules.
    if len(mutation_step_ids) > 1:
        # The first mutation could not have applied — a later one still found
        # this memory to target. The later mutation(s) are the caller's separate
        # concern, not reported here.
        took_effect = False
        signal = "multiple_mutations"
    else:
        node = graph.node(memory_id)
        status = node.memory_status if node is not None else None
        if status is None and node is not None:
            # memory_status may be unset on a node hydrated without it; fall back
            # to the raw record's "status" field.
            status = node.raw.get("status")
        if status == _STATUS_ACTIVE:
            took_effect = False
            signal = "status_active"
        else:
            if status is None:
                logger.debug(
                    "memory %s has a single mutation and no recorded status; "
                    "treating absence as 'not active' (took_effect=True), since "
                    "there is no positive evidence the mutation failed.",
                    memory_id,
                )
            took_effect = True
            signal = "status_changed"

    # 4. Locate the chosen mutation step's trace position to bound the window's
    #    RIGHT edge. If it is missing from the trace (the WARNING above already
    #    fired), there is no in-trace span to slice, so the window is empty.
    mutation_idx = step_index.get(chosen_step_id)

    # 5. Locate the producing step's position to bound the window's LEFT edge.
    producing_step_id = find_producing_step(graph, memory_id)
    start_idx = 0  # default: root memory with no producing step -> from the top
    if producing_step_id is None:
        logger.debug(
            "memory %s has no producing step; stale window starts at the "
            "beginning of the trace",
            memory_id,
        )
    else:
        producing_idx = step_index.get(producing_step_id)
        if producing_idx is None:
            # The producing step exists in the graph but not in the supplied
            # trace slice — legitimate if the caller passed a partial trace.
            # Fall back to "from the beginning" rather than raising.
            logger.debug(
                "producing step %s for memory %s not found in steps_in_order; "
                "stale window starts at the beginning of the trace",
                producing_step_id,
                memory_id,
            )
        else:
            start_idx = producing_idx + 1  # EXCLUSIVE of the producing step

    # Steps strictly between the producing step and the chosen mutation step.
    if mutation_idx is None:
        window_step_ids: List[str] = []
    else:
        window_step_ids = [
            s["step_id"] for s in steps_in_order[start_idx:mutation_idx]
        ]

    return {
        "mutation_step_id": chosen_step_id,
        "mutation_step_type": mutation_step_type,
        "took_effect": took_effect,
        "signal": signal,
        "window_step_ids": window_step_ids,
    }


# --------------------------------------------------------------------------- #
# 3. Ancestor chain — backward walk over "derive" edges
# --------------------------------------------------------------------------- #

def walk_ancestor_chain(
    graph: DependencyGraph, memory_id: str
) -> List[Dict[str, Any]]:
    """Return ``memory_id``'s full ancestor chain, OLDEST ancestor first.

    Walks BACKWARD over ``derive`` edges (``graph.predecessor_edges`` filtered
    to ``"derive" in etypes`` — the derive pass runs ancestor -> derived, so a
    derived memory's ``derive`` predecessors are the ancestors it was derived
    from). The walk is breadth-first, which discovers the NEAREST ancestors
    first; the result is then reversed so the OLDEST ancestor comes first,
    letting a caller that processes the list in order repair ancestors before
    descendants. A ``visited`` set (seeded with ``memory_id`` so it can never
    list itself) guards against a cycle, the same way support_checker.py's walk
    does. (The graph also carries a memory->memory ``supersede`` edge now; this
    walk does NOT follow it — it stays a pure ``derive`` lineage walk — but the
    edge is available should a future "what replaced this memory" lookup want
    it.)

    Each entry is::

        {"memory_id": str, "raw": dict}   # raw = graph.node(ancestor).raw

    so the caller can inspect ``trust_score`` / ``status`` or cross-reference the
    ancestor against its own faulty/affected set WITHOUT this function making any
    such judgment. Returns an empty list if ``memory_id`` has no ``derive``
    ancestors at all (logged at DEBUG). Raises ``KeyError`` if ``memory_id`` is
    not a node in ``graph``.
    """
    _require_node(graph, memory_id)

    visited = {memory_id}
    discovery: List[str] = []  # nearest-first (BFS order)
    frontier: List[str] = sorted(
        src
        for src, etypes in graph.predecessor_edges(memory_id).items()
        if _DERIVE in etypes
    )
    if not frontier:
        logger.debug("memory %s has no derive ancestors", memory_id)
        return []

    while frontier:
        ancestor_id = frontier.pop(0)
        if ancestor_id in visited:
            logger.debug(
                "ancestor walk for %s revisited %s; skipping (cycle/diamond)",
                memory_id,
                ancestor_id,
            )
            continue
        visited.add(ancestor_id)
        discovery.append(ancestor_id)
        # Queue this ancestor's OWN derive predecessors (older ancestors),
        # sorted for deterministic order.
        for src in sorted(
            s
            for s, etypes in graph.predecessor_edges(ancestor_id).items()
            if _DERIVE in etypes
        ):
            if src not in visited:
                frontier.append(src)

    # Reverse the nearest-first discovery order so the OLDEST ancestor is first.
    chain: List[Dict[str, Any]] = []
    for ancestor_id in reversed(discovery):
        node = graph.node(ancestor_id)
        chain.append(
            {"memory_id": ancestor_id, "raw": node.raw if node is not None else {}}
        )
    return chain


# --------------------------------------------------------------------------- #
# 4. Tool action for an observation — backward over a chain edge from a cited
#    tool_observation
# --------------------------------------------------------------------------- #

def find_tool_action_for_observation(
    graph: DependencyGraph, step_id: str
) -> Optional[str]:
    """Return the tool_action_id that produced a tool_observation cited by
    ``step_id``'s used_ids, or None.

    Reads ``graph.node(step_id).raw.get("used_ids", [])`` and keeps the cited
    ids whose node has ``semantic_type == "tool_observation"``. For each such
    tool_observation, walks BACKWARD over ``chain`` edges
    (``graph.predecessor_edges(observation_id)`` filtered to ``"chain" in
    etypes``) to find its immediate predecessor. The ``chain`` label runs
    step -> step in trace order (sourced from ``next_step_ids``:
    ``claim->plan->tool_action->tool_observation->final_answer``; confirmed
    against graph_schema.EdgePassSpec "chain"), so a tool_observation's chain
    PREDECESSOR is the tool_action that produced it — the direction this walk
    follows.

    * Returns ``None`` if ``step_id``'s used_ids cites no tool_observation at
      all (logged at DEBUG) — the common case for non-tool_observation sources
      (agent_summary, user_input).
    * Returns ``None`` if a cited tool_observation has no ``chain`` predecessor
      at all (an unusual graph shape — logged at WARNING, since every
      tool_observation should have a producing tool_action by construction).
    * If MORE THAN ONE tool_observation is cited (``step_id`` relied on several
      tool results), returns the FIRST in sorted order by cited-id string,
      logging a WARNING that multiple were found and only one is returned. A
      caller needing ALL of them must iterate ``used_ids`` itself rather than
      rely on this function — it returns a single best-effort answer for the
      common single-observation case only.

    Raises ``KeyError`` if ``step_id`` is not a node in ``graph`` at all (a
    caller bug, should be loud — reuses :func:`_require_node`).
    """
    _require_node(graph, step_id)

    node = graph.node(step_id)
    used_ids = node.raw.get("used_ids", []) if node is not None else []

    # Cited tool_observation steps, deterministic order (sorted by id string).
    cited_observations = sorted(
        cited
        for cited in used_ids
        if (cn := graph.node(cited)) is not None
        and cn.semantic_type == _TOOL_OBSERVATION
    )
    if not cited_observations:
        logger.debug(
            "step %s cites no tool_observation in used_ids; no tool_action to "
            "find (the common case for agent_summary/user_input sources)",
            step_id,
        )
        return None
    if len(cited_observations) > 1:
        logger.warning(
            "step %s cites %d tool_observations %s; this function returns a "
            "single best-effort answer (the first in sorted order). A caller "
            "needing all of them must iterate used_ids itself.",
            step_id,
            len(cited_observations),
            cited_observations,
        )

    observation_id = cited_observations[0]
    chain_predecessors = sorted(
        src
        for src, etypes in graph.predecessor_edges(observation_id).items()
        if _CHAIN in etypes
    )
    if not chain_predecessors:
        logger.warning(
            "cited tool_observation %s (used by step %s) has no chain "
            "predecessor; expected its producing tool_action. Returning None.",
            observation_id,
            step_id,
        )
        return None
    return chain_predecessors[0]


# --------------------------------------------------------------------------- #
# summary block)
# --------------------------------------------------------------------------- #

def _load_graph(path: str) -> DependencyGraph:
    with open(path, "r", encoding="utf-8") as fh:
        return DependencyGraph.from_dict(json.load(fh))


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Structural provenance lookups for one memory: its "
                    "producing step, the trace window to any later "
                    "update/delete/consolidate mutation, and its derived_from "
                    "ancestor chain. Pure structure — no fault judgment, no "
                    "repair decisions."
    )
    parser.add_argument(
        "--graph", required=True,
        help="dependency_graph.json path (DependencyGraph.to_dict output)",
    )
    parser.add_argument(
        "--memory_id", required=True, help="memory_id to look up (e.g. m_004)",
    )
    parser.add_argument(
        "--steps", default=None,
        help="optional trace-steps JSONL (in trace order) for find_stale_window; "
             "filter to a task with --task_id when the file holds many tasks",
    )
    parser.add_argument(
        "--task_id", default=None,
        help="optional task_id to filter --steps to (matches graph.task_id)",
    )
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    print(f"=== fault provenance for {args.memory_id} (task {graph.task_id}) ===")

    # 1. Producing step.
    producing = find_producing_step(graph, args.memory_id)
    if producing is None:
        print("  producing step:   (none — root/seed memory)")
    else:
        print(f"  producing step:   {producing}")
        # 1b. Tool action behind the producing step's cited observation (its
        #     real use case: regenerating the memory from a cited tool result
        #     means replaying that result's tool_action alongside the producing
        #     step). Keyed on the STEP found above, not the input memory_id.
        tool_action = find_tool_action_for_observation(graph, producing)
        if tool_action is None:
            print("  tool_action (for producing step's observation): (none)")
        else:
            print(f"  tool_action (for producing step's observation): {tool_action}")

    # 2. Stale window (only if a steps file was supplied).
    if args.steps:
        steps = _load_jsonl(args.steps)
        if args.task_id is not None:
            steps = [s for s in steps if s.get("task_id") == args.task_id]
        window = find_stale_window(graph, args.memory_id, steps)
        if window is None:
            print("  stale window:     (no mutation step targets this memory)")
        else:
            print("  stale window:")
            print(f"    mutation step:   {window['mutation_step_id']} "
                  f"({window['mutation_step_type']})")
            print(f"    took effect:     {window['took_effect']}")
            print(f"    signal:          {window['signal']}")
            print(f"    window steps:    {window['window_step_ids']} "
                  f"({len(window['window_step_ids'])} step(s) between)")
    else:
        print("  stale window:     (skipped — no --steps provided)")

    # 3. Ancestor chain.
    chain = walk_ancestor_chain(graph, args.memory_id)
    if not chain:
        print("  ancestor chain:   (no derived_from ancestors)")
    else:
        rendered = " -> ".join(
            f"{a['memory_id']}(status={a['raw'].get('status')})" for a in chain
        )
        print(f"  ancestor chain:   {rendered}  [oldest first]")


if __name__ == "__main__":
    _main()
