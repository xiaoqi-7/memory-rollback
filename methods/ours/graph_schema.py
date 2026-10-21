# methods/ours/graph_schema.py
"""Data model + structural contract for the dependency graph.

This module is the single source of truth for *what the dependency graph looks
like*. It owns the two dataclasses that make up the graph — :class:`GraphNode`
and :class:`DependencyGraph` — plus the reference metadata that documents their
shape: the ID naming conventions, the per-kind field layout, the catalogue of
edge passes and the ``step_type`` filters that gate them, the turn-assignment
rule, and the validation invariants.

It contains **no graph-building logic**. Nothing here iterates ``steps`` or
``memories``, assigns turns, adds an edge, or validates a graph; those passes
live in ``build_dependency_graph.py``, which imports the dataclasses from here
and fills them in. Other modules (e.g. ``affected_subgraph.py``) likewise
import :class:`DependencyGraph`/:class:`GraphNode` and the descriptors below
instead of re-deriving "what does this graph look like" from
``build_dependency_graph``'s internals, so the data model and its documentation
can never drift apart.

The descriptor tables (``NODE_KINDS``, ``EDGE_PASSES``, …) retype a few facts
that the construction code expresses as literals (the producing/consuming
``step_type`` sets). The ``__main__`` self-check guards the parts that can be
verified automatically — that each declared node-kind field set really matches
``dataclasses.fields(GraphNode)``.

Design decisions worth calling out (spec sections 6.4, 6.5, 8, 11, 18):

* **Edge types live on the edge — as a SET.** ``forward_edges``/
  ``backward_edges`` map ``node_id -> {neighbour_id: {edge_type, ...}}`` (not
  bare adjacency sets, and not a single label per pair), and each
  :class:`EdgePassSpec` declares the :class:`EdgeType` string it emits. A single
  ``(src, dst)`` pair can hold MORE THAN ONE label at once, so a step->step link
  that is both the reasoning ``chain`` AND a ``cite`` (the later step listed the
  earlier one in ``used_ids``) keeps both labels instead of one overwriting the
  other. ``forward_edges`` and ``backward_edges`` stay SEPARATE structures, but
  a relationship uses the SAME label name in both (direction comes from which
  map you read, not from the label). This is what lets
  ``rollback_optimizer.py``'s "is there an independent *trusted support* path"
  check (section 8.1) and the Derived Memory Cleanup Recall metric (section 11)
  tell a ``support`` edge apart from a ``chain`` or ``derive`` edge via set
  membership instead of merely seeing that two nodes are linked.
  ``successors()``/``predecessors()`` still return a plain ``Set[str]`` of
  neighbour ids (the contract every consumer relies on); the
  ``edge_type()``/``successor_edges()``/``predecessor_edges()`` accessors expose
  the label SETS.

* **Lineage and supersession are first-class edges.** Section 6.1 gives every
  memory ``derived_from`` / ``supersedes`` lineage fields, and section 18 makes
  cleaning up *derived* faulty memories central to this work. The ``derive``
  label declares a ``memory -> memory`` edge (ancestor -> derived) so
  ``affected_subgraph.py`` can walk faulty-memory -> derived-memory chains with
  ``descendants()`` rather than re-parsing ``GraphNode.raw`` per node. The
  ``supersede`` label declares a second ``memory -> memory`` edge (old ->
  replacement) sourced from each memory's OWN ``supersedes`` field, so a
  superseded memory's backward map now records that it was superseded — fixing
  the asymmetry where that fact lived only on the replacement's record. (As with
  every label, the actual edge wiring lives in ``build_dependency_graph.py``;
  this module only declares the spec.)

* **Semantic type is a parallel, derived field — not a new NodeKind.** Section
  6.4 types nodes directly as ``memory`` / ``claim`` / ``tool_action`` rather
  than as a generic ``step`` carrying a sub-type. Rather than *promote*
  step_type into ``NodeKind`` (which would break the ``node_kind`` ∈
  {step, memory, user_input} contract that ``build_dependency_graph`` and every
  consumer relies on), :class:`GraphNode` exposes a derived ``semantic_type``
  that collapses ``node_kind`` + ``step_type`` into the ground-truth vocabulary
  (:class:`SemanticType`), surfaced in ``to_dict()`` so its output is directly
  diffable against ground truth with no translation layer. It is a *computed
  property*, not a stored field, precisely so it can never drift from the
  ``node_kind``/``step_type`` it is derived from (the same anti-drift principle
  the ``dataclasses.fields`` self-check enforces for the stored fields).
"""

from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# --------------------------------------------------------------------------- #
# 1. ID conventions
# --------------------------------------------------------------------------- #

MEMORY_ID_PREFIX = "m_"            # memory records: m_NNN, m_fNNN, ...
STEP_ID_PREFIX = "s_"             # trace steps:    s_NNN
# User-input nodes use the agent's own minted id format verbatim
# (react_agent.run: ``f"user_input_{self.turn_index}"``), where turn_index is
# 1-based — the first turn is "user_input_1", there is no "user_input_0"
# (self.turn_index starts at 0 and is incremented BEFORE minting). This is the
# id a trace step carries in ``used_ids`` to cite a prior user turn, so the
# graph must use the SAME string identity, not a re-derived alias.
USER_INPUT_ID_PREFIX = "user_input_"     # user inputs: user_input_N (1-based)
USER_INPUT_ID_FORMAT = "user_input_{turn}"  # e.g. turn 1 -> "user_input_1"


def is_memory_id(token: str) -> bool:
    """True if ``token`` names a memory record (``m_`` prefix)."""
    return token.startswith(MEMORY_ID_PREFIX)


def is_step_id(token: str) -> bool:
    """True if ``token`` names a trace step (``s_`` prefix).

    The ``s_`` prefix does not collide with the user_input prefix
    (``user_input_``) or the memory prefix (``m_``), so a step id, a
    user_input id, and a memory id remain unambiguously distinguishable by
    prefix alone (see :func:`classify_id`).
    """
    return token.startswith(STEP_ID_PREFIX)


def user_input_node_id(turn: int) -> str:
    """Return the node_id for the user_input node at the given 1-indexed turn.

    Matches the agent's minted format exactly (``user_input_{turn}``, 1-based),
    so this id is byte-for-byte what a step's ``used_ids`` cites for that turn.
    """
    return USER_INPUT_ID_FORMAT.format(turn=turn)


class NodeKind(str, Enum):
    """The three values ``GraphNode.node_kind`` ever takes."""
    STEP = "step"
    MEMORY = "memory"
    USER_INPUT = "user_input"


def classify_id(token: str) -> Optional[NodeKind]:
    """Best-effort id->kind from prefix alone (documentation-grade only)."""
    if is_memory_id(token):
        return NodeKind.MEMORY
    if is_step_id(token):
        return NodeKind.STEP
    if token.startswith(USER_INPUT_ID_PREFIX):
        return NodeKind.USER_INPUT
    return None


class SemanticType(str, Enum):
    """The ground-truth node vocabulary.

    These are the labels the dependency-graph record and the
    ground_truth_affected_nodes.jsonl annotations type nodes with directly —
    ``memory`` / ``claim`` / ``tool_action`` rather than a generic ``step``
    carrying a sub-type. ``GraphNode.semantic_type`` collapses
    ``node_kind`` + ``step_type`` into exactly these values so
    :meth:`DependencyGraph.to_dict` is diffable against ground truth with no
    translation layer. For a ``step`` node the semantic type IS its
    ``step_type``; ``memory`` and ``user_input`` map straight from
    ``node_kind``.
    """
    USER_INPUT = "user_input"
    MEMORY = "memory"
    MEMORY_READ = "memory_read"
    CLAIM = "claim"
    PLAN = "plan"
    TOOL_ACTION = "tool_action"
    TOOL_OBSERVATION = "tool_observation"
    FINAL_ANSWER = "final_answer"
    MEMORY_WRITE = "memory_write"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    MEMORY_CONSOLIDATE = "memory_consolidate"


# Step-level semantic types: the values ``semantic_type`` takes for a step
# node, i.e. the SemanticType members that are populated from ``step_type``
# rather than straight from ``node_kind``. Pulled from SemanticType minus the
# two kind-level labels so the set can't drift from the enum.
_KIND_LEVEL_SEMANTIC_TYPES: FrozenSet[SemanticType] = frozenset(
    {SemanticType.MEMORY, SemanticType.USER_INPUT}
)
STEP_SEMANTIC_TYPES: FrozenSet[str] = frozenset(
    s.value for s in SemanticType if s not in _KIND_LEVEL_SEMANTIC_TYPES
)


def semantic_type_for(node_kind: str, step_type: Optional[str]) -> Optional[str]:
    """Collapse (node_kind, step_type) into the section 6.4 vocabulary.

    Returns the ground-truth semantic-type string for a node, or ``None`` if a
    step node has no ``step_type`` to specialise on (the same fallback the
    ``node_type`` grouping in affected_subgraph.py uses). ``memory`` and
    ``user_input`` map straight from ``node_kind``; a ``step`` adopts its
    ``step_type`` verbatim (memory_read / claim / plan / tool_action /
    tool_observation / final_answer / memory_write / memory_update /
    memory_delete / memory_consolidate).
    """
    if node_kind == NodeKind.STEP.value:
        return step_type
    return node_kind


# --------------------------------------------------------------------------- #
# 2. Data model — the dataclasses that ARE the graph
# --------------------------------------------------------------------------- #

class EdgeType(str, Enum):
    """The edge-type label stored on every edge (spec section 6.4).

    Each value names the relationship a dependency edge encodes, so consumers
    can distinguish *how* two nodes are connected — not just *that* they are.
    ``rollback_optimizer.py``'s independent-trusted-support check (section 8.1)
    keys off ``SUPPORT`` vs ``CHAIN``; the Derived Memory Cleanup Recall metric
    (section 11) keys off ``DERIVE``.

    **Multi-type edges.** A single ``(src, dst)`` pair can carry MORE THAN ONE
    of these labels at once — ``forward_edges``/``backward_edges`` map a
    neighbour to a SET of EdgeType strings, not a single one — so two genuinely
    different relationships between the same pair (e.g. a ``chain`` step->step
    link that is ALSO a ``cite`` because the later step listed the earlier one
    in ``used_ids``) coexist instead of one silently overwriting the other.

    **One shared label name per relationship.** ``forward_edges`` and
    ``backward_edges`` stay SEPARATE structures, but a relationship uses the
    SAME label in both: a ``cite`` edge is "cite" whether you find it in
    ``forward_edges[src]`` or ``backward_edges[dst]`` — there is no separate
    forward-name/backward-name pair. Direction is carried by WHICH map you read,
    not by the label.

    The vocabulary is 10 short verb forms. Each :class:`EdgePassSpec` declares
    exactly one of these as the label its pass emits (see ``EDGE_PASSES``).
    """
    CHAIN = "chain"            # step -> step in the reasoning chain (pass chain)
    CITE = "cite"             # cited step/user_input -> citing step (pass cite)
    SUPPORT = "support"       # memory -> step it justifies (pass support)
    PRODUCE = "produce"       # mutating step -> memory it writes (pass produce)
    DELETE = "delete"         # memory -> memory_delete step (pass delete)
    UPDATE = "update"         # memory -> memory_update step (pass update)
    CONSOLIDATE = "consolidate"  # memory -> memory_consolidate step (pass consolidate)
    SUPERSEDE = "supersede"   # old memory -> superseding memory (pass supersede)
    DERIVE = "derive"         # ancestor memory -> derived memory (pass derive)
    INITIATE = "initiate"     # user_input -> first step of its turn (pass initiate)


@dataclass
class GraphNode:
    node_id: str                  # step_id, memory_id, or user_input id (user_input_N)
    node_kind: str                # "step", "memory", or "user_input"
    step_type: Optional[str] = None      # only set if node_kind == "step"
    turn: Optional[int] = None           # 1-indexed turn (step + user_input nodes)
    memory_status: Optional[str] = None  # only set if node_kind == "memory"
    raw: Dict[str, Any] = field(default_factory=dict)  # original record

    @property
    def semantic_type(self) -> Optional[str]:
        """Ground-truth node label (section 6.4): node_kind+step_type collapsed.

        DERIVED, not stored — see the module docstring's third design note. A
        ``step`` node reports its ``step_type`` (claim / tool_action / …); a
        ``memory`` or ``user_input`` node reports its ``node_kind``. Returns
        ``None`` only for a step that is missing its ``step_type``.
        """
        return semantic_type_for(self.node_kind, self.step_type)


@dataclass
class DependencyGraph:
    task_id: str
    nodes: Dict[str, GraphNode]                  # node_id -> GraphNode
    forward_edges: Dict[str, Dict[str, Set[str]]]  # src -> {dst: {edge_type, ...}}
    backward_edges: Dict[str, Dict[str, Set[str]]]  # dst -> {src: {edge_type, ...}}
    step_turn: Dict[str, int]                     # step_id -> turn number
    n_turns: int

    def successors(
        self, node_id: str, exclude_edge_types: FrozenSet[str] = frozenset()
    ) -> Set[str]:
        """Single-hop forward neighbours (nodes this node points to).

        Returns a bare ``Set[str]`` of neighbour ids — the contract every
        traversal consumer relies on. Multi-type edges are transparent here: a
        pair linked by both ``chain`` and ``cite`` is still one neighbour id.
        Use :meth:`successor_edges` or :meth:`edge_type` when the per-edge
        label set matters.

        ``exclude_edge_types`` is an OPTIONAL set of edge-type labels to ignore
        when deciding whether a neighbour is reachable. It defaults to the empty
        set, so by default EVERY existing caller is unaffected (all edges
        count). When non-empty, a neighbour is included iff AT LEAST ONE
        edge_type connecting to it is NOT in ``exclude_edge_types`` — because
        edges are multi-type label SETS, a pair carrying BOTH ``chain`` and
        ``cite`` still counts as a neighbour via ``cite`` even if ``chain`` is
        excluded; a neighbour is dropped only when EVERY label connecting to it
        is in the exclusion set. (Set difference ``labels - exclude_edge_types``
        is truthy exactly when some surviving label remains.)
        """
        edges = self.forward_edges.get(node_id, {})
        if not exclude_edge_types:
            return set(edges)
        return {
            dst for dst, labels in edges.items() if labels - exclude_edge_types
        }

    def predecessors(self, node_id: str) -> Set[str]:
        """Single-hop backward neighbours (nodes pointing to this node).

        Returns a bare ``Set[str]`` of neighbour ids (see :meth:`successors`).
        """
        return set(self.backward_edges.get(node_id, {}))

    def successor_edges(self, node_id: str) -> Dict[str, Set[str]]:
        """Forward neighbours WITH their edge_type set: ``{dst: {edge_type, ...}}``.

        The typed counterpart of :meth:`successors`. Each neighbour maps to the
        SET of labels on that pair, so a pair carrying both ``chain`` and
        ``cite`` reports ``{"chain", "cite"}``. Lets section 8.1's support check
        and the section 11 cleanup metric tell a ``support`` edge from a
        ``chain`` / ``derive`` edge via set membership. Returns a deep-ish copy
        (fresh outer dict and fresh label sets) so callers cannot mutate the
        graph's internal edge map.
        """
        return {dst: set(labels) for dst, labels in self.forward_edges.get(node_id, {}).items()}

    def predecessor_edges(self, node_id: str) -> Dict[str, Set[str]]:
        """Backward neighbours WITH their edge_type set: ``{src: {edge_type, ...}}``."""
        return {src: set(labels) for src, labels in self.backward_edges.get(node_id, {}).items()}

    def edge_type(self, src: str, dst: str) -> Set[str]:
        """Return the SET of edge_type labels on the src->dst edge.

        Returns a fresh ``set`` of label strings, or an EMPTY set if there is no
        edge between the pair (the "no edge" contract: callers test
        ``"support" in graph.edge_type(a, b)`` or truthiness of the set rather
        than comparing against ``None``). A pair carrying two relationships
        (e.g. ``chain`` + ``cite``) returns both labels.
        """
        return set(self.forward_edges.get(src, {}).get(dst, ()))

    def node(self, node_id: str) -> Optional[GraphNode]:
        """Return the GraphNode for ``node_id`` or None if absent."""
        return self.nodes.get(node_id)

    def user_input_node_id(self, turn: int) -> str:
        """Return the node_id for the user_input node at the given 1-indexed turn."""
        return user_input_node_id(turn)

    def descendants(
        self, *seeds: str, exclude_edge_types: FrozenSet[str] = frozenset()
    ) -> List[str]:
        """Forward-reachable closure from one or more seed node_ids (BFS).

        Walks ``forward_edges`` from every seed and returns every node
        transitively reachable, in BFS discovery order, EXCLUDING the seeds
        themselves. Because the edge set already threads producer steps to the
        memories they create (the produce pass), those memories on to their
        consumers (the support pass), and faulty ancestors on to derived
        memories (the derive pass), a single forward BFS is the full
        multi-generation closure: a derived memory reached mid-walk has its own
        outgoing edges followed without any re-seeding logic here. A pair
        carrying SEVERAL labels (e.g. chain + cite) is still one reachable hop
        to one neighbour, visited once. Successors are sorted before being
        queued so the order is deterministic despite edges living in a dict.
        Unknown seeds contribute no edges (callers validate existence).

        ``exclude_edge_types`` is threaded through EVERY ``successors()`` call
        in the walk (defaulting to the empty set — no exclusion — so existing
        callers are unaffected). When non-empty, a hop along an edge whose
        labels are ALL in the exclusion set is not traversed, so a node only
        reachable via such edges never enters the closure. ``affected_subgraph``
        passes ``{"chain"}`` here so a node merely next-in-sequence after a
        contaminated node (a pure ``chain`` link with no evidentiary label) is
        not swept in.
        """
        visited: Set[str] = set(seeds)
        order: List[str] = []
        queue: deque[str] = deque(seeds)
        while queue:
            node_id = queue.popleft()
            for nxt in sorted(self.successors(node_id, exclude_edge_types)):
                if nxt not in visited:
                    visited.add(nxt)
                    order.append(nxt)
                    queue.append(nxt)
        return order

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain JSON-safe dict (inverse of :meth:`from_dict`).

        Edges are emitted as ``{src: {dst: [edge_type, ...]}}`` with the
        destination keys sorted AND each edge's label set serialized as a SORTED
        LIST, so the output is deterministic and round-trips through ``json``
        (JSON has no set type). A pair carrying two relationships emits both
        labels, e.g. ``"s_2": ["chain", "cite"]``. Every GraphNode is emitted
        with all its stored fields plus the derived ``semantic_type`` (section
        6.4 ground-truth label) so the output is directly diffable against
        ground truth; ``raw`` is passed through as-is (it is already JSON data
        loaded from the trace/memory JSONL).
        """
        return {
            "task_id": self.task_id,
            "n_turns": self.n_turns,
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "node_kind": n.node_kind,
                    "semantic_type": n.semantic_type,
                    "step_type": n.step_type,
                    "turn": n.turn,
                    "memory_status": n.memory_status,
                    "raw": n.raw,
                }
                for nid, n in self.nodes.items()
            },
            "forward_edges": {
                src: {dst: sorted(dsts[dst]) for dst in sorted(dsts)}
                for src, dsts in self.forward_edges.items()
            },
            "backward_edges": {
                dst: {src: sorted(srcs[src]) for src in sorted(srcs)}
                for dst, srcs in self.backward_edges.items()
            },
            "step_turn": dict(self.step_turn),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DependencyGraph":
        """Reconstruct a DependencyGraph from :meth:`to_dict` output.

        Edge maps are rehydrated back into ``{neighbour: {edge_type, ...}}``
        dicts — each serialized label LIST is turned back into a ``set`` — so
        the in-memory contract is preserved for ``successors``/``predecessors``
        (which expose ids) and ``successor_edges``/``edge_type`` (which expose
        the label sets). The derived ``semantic_type`` in the serialized node is
        ignored on load — it is recomputed from ``node_kind``/``step_type`` so
        it can never round-trip a stale value.
        """
        nodes = {
            nid: GraphNode(
                node_id=nd["node_id"],
                node_kind=nd["node_kind"],
                step_type=nd.get("step_type"),
                turn=nd.get("turn"),
                memory_status=nd.get("memory_status"),
                raw=nd.get("raw", {}),
            )
            for nid, nd in data.get("nodes", {}).items()
        }
        forward_edges = {
            src: {dst: set(labels) for dst, labels in dsts.items()}
            for src, dsts in data.get("forward_edges", {}).items()
        }
        backward_edges = {
            dst: {src: set(labels) for src, labels in srcs.items()}
            for dst, srcs in data.get("backward_edges", {}).items()
        }
        return cls(
            task_id=data["task_id"],
            nodes=nodes,
            forward_edges=forward_edges,
            backward_edges=backward_edges,
            step_turn=dict(data.get("step_turn", {})),
            n_turns=data.get("n_turns", 0),
        )

# --------------------------------------------------------------------------- #
# 3. Node kinds — which GraphNode fields are populated per kind
# --------------------------------------------------------------------------- #
# Field names are pulled from ``dataclasses.fields(GraphNode)`` so they can't
# drift if GraphNode changes. node_id/node_kind/raw are set for every kind; the
# kind-specific fields are what distinguish the three rows.

_GRAPH_NODE_FIELDS: Tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(GraphNode)
)

COMMON_NODE_FIELDS: FrozenSet[str] = frozenset({"node_id", "node_kind", "raw"})


@dataclass(frozen=True)
class NodeKindSpec:
    """Describes one ``node_kind`` and the GraphNode fields it populates."""
    node_kind: NodeKind
    id_prefix: str
    kind_specific_fields: FrozenSet[str]
    note: str

    @property
    def populated_fields(self) -> FrozenSet[str]:
        """All GraphNode fields set for this kind (common + kind-specific)."""
        return COMMON_NODE_FIELDS | self.kind_specific_fields


NODE_KINDS: Dict[NodeKind, NodeKindSpec] = {
    NodeKind.STEP: NodeKindSpec(
        node_kind=NodeKind.STEP,
        id_prefix=STEP_ID_PREFIX,
        kind_specific_fields=frozenset({"step_type", "turn"}),
        note="A trace step. step_type + turn are set; memory_status stays None.",
    ),
    NodeKind.MEMORY: NodeKindSpec(
        node_kind=NodeKind.MEMORY,
        id_prefix=MEMORY_ID_PREFIX,
        kind_specific_fields=frozenset({"memory_status"}),
        note="A memory record. memory_status is set; turn is NOT assigned to "
             "memory nodes (turn is a step-level concept).",
    ),
    NodeKind.USER_INPUT: NodeKindSpec(
        node_kind=NodeKind.USER_INPUT,
        id_prefix=USER_INPUT_ID_PREFIX,
        kind_specific_fields=frozenset({"turn"}),
        note="The user's utterance for a turn. turn is set and raw carries "
             "{'turn', 'text'}; step_type and memory_status stay None.",
    ),
}

# --------------------------------------------------------------------------- #
# 4. Edge types — one entry per edge label in the 10-verb vocabulary
# --------------------------------------------------------------------------- #
# step_type literals mirror the producing/consuming sets inside
# build_dependency_graph's passes. memory_read produces no edges; it appears
# only as the turn boundary (see section 5). Each EdgePassSpec declares the ONE
# EdgeType label its edges carry, so the per-edge labels stored in
# forward_edges/backward_edges each trace back to exactly one spec. The three
# memory-consumption labels (delete/update/consolidate) are realised by a single
# construction function in build_dependency_graph.py that branches on
# step.step_type, but they are catalogued here as three distinct specs so the
# "every EdgeType is emitted by exactly one spec" invariant holds.

PRODUCING_STEP_TYPES: FrozenSet[str] = frozenset(
    {"memory_write", "memory_update", "memory_consolidate"}
)
# memory_delete is a CONSUMING step (its invalidated_memory_ids names the
# memory it tears down, so the delete pass wires that memory -> the delete step)
# but NOT a PRODUCING one: a delete's generated_memory_ids is always empty, so it
# never appears in PRODUCING_STEP_TYPES. The consumption passes give a memory
# removed/updated/consolidated by such a step a delete/update/consolidate edge to
# the step that mutated it, so graph walks of "what invalidated this memory"
# (predecessor_edges) no longer silently miss it.
CONSUMING_STEP_TYPES: FrozenSet[str] = frozenset(
    {"memory_update", "memory_consolidate", "memory_delete"}
)
# The consumption step_type -> edge label mapping. build_dependency_graph's
# consumption pass looks the label up here per step rather than hardcoding it,
# so a memory_update step's edge is labelled "update", a memory_delete's
# "delete", a memory_consolidate's "consolidate" — three distinct labels instead
# of one merged one.
CONSUMPTION_STEP_TYPE_TO_EDGE: Dict[str, "EdgeType"] = {
    "memory_delete": EdgeType.DELETE,
    "memory_update": EdgeType.UPDATE,
    "memory_consolidate": EdgeType.CONSOLIDATE,
}


@dataclass(frozen=True)
class EdgePassSpec:
    """Describes one edge label in the graph and the record field that emits it.

    ``src_kind`` names the single NodeKind every source endpoint of this label's
    edges has. The ``cite`` label is the ONE exception: a non-memory ``used_ids``
    citation can name a claim step, a tool_observation step (both ``STEP``), OR a
    user_input node (``USER_INPUT``), so its source endpoint is one of two kinds
    depending on what was cited. ``EdgePassSpec``'s single ``src_kind`` field
    cannot express "either of two kinds", so the cite spec declares
    ``src_kind=STEP`` (the common case) and documents the ``USER_INPUT``
    possibility here and in its description rather than silently forcing a single
    kind that is sometimes wrong. The destination is always the citing step
    (``STEP``).

    ``pass_id`` equals the ``edge_type``'s value (e.g. "chain"), so the catalogue
    is keyed by the verb it emits. Several specs can share a single construction
    function in build_dependency_graph.py (the delete/update/consolidate specs
    are all wired by the one memory-consumption pass, which branches on
    step_type), but each spec still owns exactly one distinct label.
    """
    pass_id: str                       # the edge_type value this spec emits
    src_kind: NodeKind
    dst_kind: NodeKind
    edge_type: EdgeType                 # label every edge from this spec carries
    source_field: str                  # record field that produces the edge
    step_type_filter: FrozenSet[str]   # gating step_types (empty = no filter)
    description: str


EDGE_PASSES: Tuple[EdgePassSpec, ...] = (
    EdgePassSpec(
        pass_id="chain",
        src_kind=NodeKind.STEP,
        dst_kind=NodeKind.STEP,
        edge_type=EdgeType.CHAIN,
        source_field="next_step_ids",
        step_type_filter=frozenset(),
        description="chain: step->step, the intra-turn reasoning chain "
                    "threaded through next_step_ids (the step-chain successor "
                    "links). One reasoning step feeds the next "
                    "(claim->plan->tool_action->observation->final_answer). A "
                    "single step (e.g. a plan) may list more than one "
                    "next_step_id, so it can emit multiple edges. A chain edge "
                    "can COEXIST with a cite edge on the same pair (when the "
                    "later step also lists the earlier one in used_ids).",
    ),
    EdgePassSpec(
        pass_id="produce",
        src_kind=NodeKind.STEP,
        dst_kind=NodeKind.MEMORY,
        edge_type=EdgeType.PRODUCE,
        source_field="generated_memory_ids",
        step_type_filter=PRODUCING_STEP_TYPES,
        description="produce: step->memory, fires when a memory-mutating step "
                    "lists a memory_id in generated_memory_ids (it produced that "
                    "memory).",
    ),
    EdgePassSpec(
        pass_id="support",
        src_kind=NodeKind.MEMORY,
        dst_kind=NodeKind.STEP,
        edge_type=EdgeType.SUPPORT,
        source_field="used_ids",
        step_type_filter=frozenset(),
        description="support: memory->step, fires when a step actually relied "
                    "on a memory per used_ids (any step_type). used_ids is "
                    "mixed-type, so this label covers only the memory-shaped "
                    "(m_*) entries; the non-memory entries carry cite. The "
                    "memory is evidence the step leans on, which is what section "
                    "8.1's independent-support check inspects.",
    ),
    EdgePassSpec(
        pass_id="delete",
        src_kind=NodeKind.MEMORY,
        dst_kind=NodeKind.STEP,
        edge_type=EdgeType.DELETE,
        source_field="invalidated_memory_ids",
        step_type_filter=frozenset({"memory_delete"}),
        description="delete: memory->step, the memory in a memory_delete step's "
                    "invalidated_memory_ids being torn down (the step also "
                    "populates invalidated_memory_ids while producing nothing).",
    ),
    EdgePassSpec(
        pass_id="update",
        src_kind=NodeKind.MEMORY,
        dst_kind=NodeKind.STEP,
        edge_type=EdgeType.UPDATE,
        source_field="invalidated_memory_ids",
        step_type_filter=frozenset({"memory_update"}),
        description="update: memory->step, the OLD memory in a memory_update "
                    "step's invalidated_memory_ids being superseded by the "
                    "update.",
    ),
    EdgePassSpec(
        pass_id="consolidate",
        src_kind=NodeKind.MEMORY,
        dst_kind=NodeKind.STEP,
        edge_type=EdgeType.CONSOLIDATE,
        source_field="invalidated_memory_ids",
        step_type_filter=frozenset({"memory_consolidate"}),
        description="consolidate: memory->step, the OLD memory in a "
                    "memory_consolidate step's invalidated_memory_ids being "
                    "folded into the consolidated memory.",
    ),
    EdgePassSpec(
        pass_id="initiate",
        src_kind=NodeKind.USER_INPUT,
        dst_kind=NodeKind.STEP,
        edge_type=EdgeType.INITIATE,
        source_field="step_turn",
        step_type_filter=frozenset(),
        description="initiate: user_input->first step of each turn, anchoring "
                    "the turn's step chain to its user_input root node. "
                    "One-directional by design — nothing walks backward asking "
                    "'what initiated me' as a distinct operation.",
    ),
    EdgePassSpec(
        pass_id="derive",
        src_kind=NodeKind.MEMORY,
        dst_kind=NodeKind.MEMORY,
        edge_type=EdgeType.DERIVE,
        source_field="derived_from",
        step_type_filter=frozenset(),
        description="derive: memory->memory lineage. For each memory node, "
                    "every id in its derived_from list (section 6.1) gets an "
                    "edge ancestor->derived so a faulty ancestor reaches its "
                    "derived memories via descendants() (section 18 / the "
                    "Derived Memory Cleanup Recall metric, section 11) without "
                    "parsing GraphNode.raw. Actual wiring lives in "
                    "build_dependency_graph.py; this only declares the label.",
    ),
    EdgePassSpec(
        pass_id="cite",
        src_kind=NodeKind.STEP,
        dst_kind=NodeKind.STEP,
        edge_type=EdgeType.CITE,
        source_field="used_ids",
        step_type_filter=frozenset(),
        description="cite: edges from the NON-memory entries of a step's "
                    "used_ids (the entries support skips via is_memory_id). Such "
                    "an entry names another step the citing step drew on — a "
                    "claim step or a tool_observation step — OR a user_input "
                    "node (a prior user turn). For each, an edge runs "
                    "cited-node->citing-step. A cite edge can now COEXIST with a "
                    "chain edge on the same pair with no special-casing (both "
                    "labels sit in the pair's label set). NOTE: src_kind is "
                    "declared STEP (the common case), but a cited user_input "
                    "makes the source a USER_INPUT node instead — cite is the "
                    "one label whose source endpoint is one of two kinds (see "
                    "EdgePassSpec's docstring); the dst is always the citing STEP.",
    ),
    EdgePassSpec(
        pass_id="supersede",
        src_kind=NodeKind.MEMORY,
        dst_kind=NodeKind.MEMORY,
        edge_type=EdgeType.SUPERSEDE,
        source_field="supersedes",
        step_type_filter=frozenset(),
        description="supersede: memory->memory. For each memory node, the id in "
                    "its OWN supersedes field (NOT any step field) names the old "
                    "memory it replaces, so we add an old->this edge. Mirrors "
                    "derive's memory-record-driven structure (different field). "
                    "This fixes the asymmetry where a superseded memory's own "
                    "backward map showed nothing about being superseded — only "
                    "its produce-edge predecessor — because the supersedes link "
                    "lived on the REPLACEMENT's record, not the old memory's.",
    ),
)


# --------------------------------------------------------------------------- #
# 5. Turn assignment rule (summary of build_dependency_graph._assign_turns)
# --------------------------------------------------------------------------- #

TURN_BOUNDARY_STEP_TYPE = "memory_read"  # a new turn begins at each of these

TURN_ASSIGNMENT_RULE = (
    "A new turn begins at each `memory_read` step. Steps before the first "
    "memory_read (and any trailing write/update/consolidate/delete steps after "
    "a final_answer) keep the most recent turn, with a floor of turn 1. Turn is "
    "a step-level concept: it is assigned to step and user_input nodes only, "
    "never to memory nodes. n_turns is the maximum step turn."
)


# --------------------------------------------------------------------------- #
# 6. Validation invariants (mirror of build_dependency_graph.validate_graph)
# --------------------------------------------------------------------------- #

VALIDATION_INVARIANTS: Tuple[str, ...] = (
    "A memory node with no incoming AND no outgoing edges is suspicious "
    "(likely a missing used_ids reference).",
    "A memory-mutating step (memory_write/update/consolidate) should have an "
    "outgoing edge to a memory node (it is expected to produce one).",
    "Each user_input node should have exactly one outgoing initiate edge (to "
    "its turn's first step); zero means no step was found for that turn. It may "
    "carry additional outgoing cite edges when a later step cites the turn, so "
    "only the initiate edge is counted.",
    "graph.n_turns should equal the maximum turn over all step nodes.",
    "No edge endpoint should reference an id that has no node (the "
    "dangling-reference skip in _add_edge should have prevented it).",
    "A derive edge must run memory->memory; its endpoints are both memory nodes "
    "(the ancestor and the memory derived from it). A supersede edge likewise "
    "runs memory->memory (the old memory and its replacement).",
    "Every label in an edge's label set must be one declared by some "
    "EdgePassSpec (no edge carries a label outside the EdgeType enum). A single "
    "(src, dst) pair may carry MORE THAN ONE label (e.g. chain + cite).",
)


# --------------------------------------------------------------------------- #
# Sanity check — guard the schema against drift from the dataclasses.
# --------------------------------------------------------------------------- #

def _self_check() -> None:
    """Assert the declared node-kind field sets match the GraphNode dataclass."""
    node_fields = set(_GRAPH_NODE_FIELDS)

    assert COMMON_NODE_FIELDS <= node_fields, (
        f"common fields not on GraphNode: {COMMON_NODE_FIELDS - node_fields}"
    )
    for kind, spec in NODE_KINDS.items():
        assert spec.node_kind is kind, f"NODE_KINDS key/value mismatch for {kind}"
        unknown = spec.populated_fields - node_fields
        assert not unknown, f"{kind} declares non-existent fields: {unknown}"

    documented = COMMON_NODE_FIELDS.union(
        *(s.kind_specific_fields for s in NODE_KINDS.values())
    )
    assert documented == node_fields, (
        f"fields not documented by any kind: {node_fields - documented}"
    )

    assert {k.value for k in NodeKind} == {"step", "memory", "user_input"}
    assert user_input_node_id(1) == "user_input_1"
    # user_input ids never collide with step ids: a step id starts "s_", a
    # user_input id starts "user_input_", a memory id starts "m_".
    assert not is_step_id(user_input_node_id(1))
    assert not is_memory_id(user_input_node_id(1))
    assert classify_id(user_input_node_id(1)) is NodeKind.USER_INPUT

    # Edge specs — one per label in the 10-verb vocabulary, each carrying a real
    # EdgeType, and every EdgeType is emitted by exactly one spec (so the stored
    # edge labels and the enum can't drift from the catalogue).
    assert tuple(p.pass_id for p in EDGE_PASSES) == (
        "chain", "produce", "support", "delete", "update", "consolidate",
        "initiate", "derive", "cite", "supersede",
    )
    for p in EDGE_PASSES:
        assert isinstance(p.src_kind, NodeKind) and isinstance(p.dst_kind, NodeKind)
        assert isinstance(p.edge_type, EdgeType), (
            f"spec {p.pass_id} edge_type is not an EdgeType: {p.edge_type!r}"
        )
        # pass_id is the verb the spec emits, so it must equal its label value.
        assert p.pass_id == p.edge_type.value, (
            f"spec {p.pass_id} pass_id != edge_type value {p.edge_type.value!r}"
        )
    pass_edge_types = {p.edge_type for p in EDGE_PASSES}
    assert pass_edge_types == set(EdgeType), (
        f"EdgeType members not covered by any spec: {set(EdgeType) - pass_edge_types}"
    )
    assert len(pass_edge_types) == len(EDGE_PASSES), (
        "two specs share an edge_type; each spec must emit a distinct label"
    )
    assert len(EdgeType) == 10, "the edge vocabulary should hold exactly 10 verbs"

    # derive and supersede are the two memory->memory passes; each is unique.
    lineage = [p for p in EDGE_PASSES if p.edge_type is EdgeType.DERIVE]
    assert len(lineage) == 1 and lineage[0].pass_id == "derive"
    assert lineage[0].src_kind is NodeKind.MEMORY
    assert lineage[0].dst_kind is NodeKind.MEMORY
    assert lineage[0].source_field == "derived_from"

    supersede = [p for p in EDGE_PASSES if p.edge_type is EdgeType.SUPERSEDE]
    assert len(supersede) == 1 and supersede[0].pass_id == "supersede"
    assert supersede[0].src_kind is NodeKind.MEMORY
    assert supersede[0].dst_kind is NodeKind.MEMORY
    assert supersede[0].source_field == "supersedes"

    # The consumption step_type -> edge label map covers exactly the three
    # consuming step types and emits exactly the three consumption labels.
    assert set(CONSUMPTION_STEP_TYPE_TO_EDGE) == CONSUMING_STEP_TYPES
    assert {e for e in CONSUMPTION_STEP_TYPE_TO_EDGE.values()} == {
        EdgeType.DELETE, EdgeType.UPDATE, EdgeType.CONSOLIDATE
    }

    # semantic_type is the derived ground-truth label (section 6.4); confirm it
    # collapses node_kind/step_type into the SemanticType vocabulary and stays a
    # property (never a stored field, so it can't drift).
    assert "semantic_type" not in node_fields, (
        "semantic_type must stay a derived property, not a GraphNode field"
    )
    assert isinstance(GraphNode.semantic_type, property)
    semantic_values = {s.value for s in SemanticType}
    assert GraphNode("m_1", "memory").semantic_type == "memory"
    assert GraphNode("user_input_1", "user_input").semantic_type == "user_input"
    for st in STEP_SEMANTIC_TYPES:
        node = GraphNode("s_1", "step", step_type=st)
        assert node.semantic_type == st
        assert node.semantic_type in semantic_values
    assert GraphNode("s_1", "step").semantic_type is None  # no step_type
    # Every step_type a step node carries (the producing/consuming literals and
    # the turn boundary) is a known SemanticType value.
    for st in PRODUCING_STEP_TYPES | CONSUMING_STEP_TYPES | {TURN_BOUNDARY_STEP_TYPE}:
        assert st in semantic_values, f"step_type {st!r} missing from SemanticType"


if __name__ == "__main__":
    _self_check()
    print("graph_schema self-check PASSED")
    print(f"  node kinds:   {[k.value for k in NodeKind]}")
    print(f"  edge passes:  {tuple(p.pass_id for p in EDGE_PASSES)}")
    print(f"  edge types:   {[e.value for e in EdgeType]}")
    print(f"  user_input id format: {user_input_node_id(1)!r} (1-based)")
