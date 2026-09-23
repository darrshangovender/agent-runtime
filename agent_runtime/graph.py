"""Graph builder and build-time validation.

A graph is a set of named async nodes joined by static edges or conditional
routers. Validation runs at :meth:`Graph.build` (or :meth:`Graph.validate`) and
rejects unreachable nodes, missing entry/terminal, dead ends and dangling edges.
Cycles are allowed - agents loop by design - but are detected and flagged
through :class:`CycleWarning` and :attr:`GraphReport.cycles`. Cycle detection
follows static edges and the declared ``targets`` of conditional edges; a
router with no declared targets is not expanded.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field

from .state import AgentState

NodeFn = Callable[[AgentState], Awaitable[AgentState]]
RouterFn = Callable[[AgentState], "str | Awaitable[str]"]


class GraphValidationError(ValueError):
    """Raised at build time when the graph is structurally invalid."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


class RoutingError(RuntimeError):
    """Raised at run time when a router returns something that is not a node."""


class CycleWarning(UserWarning):
    """Emitted when validation finds a cycle. Cycles are allowed but flagged."""


@dataclass(slots=True)
class NodeSpec:
    name: str
    fn: NodeFn
    timeout_s: float | None = None


@dataclass(slots=True)
class ConditionalEdge:
    router: RouterFn
    targets: tuple[str, ...] | None = None  # None means "any node" (validated at run time)


@dataclass(slots=True)
class GraphReport:
    """What validation found. ``cycles`` is informational, ``problems`` is fatal."""

    nodes: list[str] = field(default_factory=list)
    reachable: list[str] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def has_cycles(self) -> bool:
        return bool(self.cycles)


class Graph:
    """Fluent builder for an agent graph. Every mutator returns ``self``."""

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.nodes: dict[str, NodeSpec] = {}
        self.edges: dict[str, str] = {}
        self.conditional: dict[str, ConditionalEdge] = {}
        self.entry: str | None = None
        self.terminals: set[str] = set()
        self._validated: GraphReport | None = None

    # -- building --------------------------------------------------------

    def add_node(self, name: str, fn: NodeFn, *, timeout_s: float | None = None) -> Graph:
        if not name or not isinstance(name, str):
            raise GraphValidationError([f"node name must be a non-empty string, got {name!r}"])
        if name in self.nodes:
            raise GraphValidationError([f"duplicate node {name!r}"])
        if not callable(fn):
            raise GraphValidationError([f"node {name!r}: fn is not callable"])
        if not inspect.iscoroutinefunction(fn):
            raise GraphValidationError(
                [f"node {name!r}: fn must be `async def fn(state) -> state`"]
            )
        if timeout_s is not None and timeout_s <= 0:
            raise GraphValidationError([f"node {name!r}: timeout_s must be positive"])
        self.nodes[name] = NodeSpec(name=name, fn=fn, timeout_s=timeout_s)
        self._validated = None
        return self

    def add_edge(self, src: str, dst: str) -> Graph:
        if src in self.conditional:
            raise GraphValidationError([f"node {src!r} already has a conditional edge"])
        if src in self.edges:
            raise GraphValidationError(
                [f"node {src!r} already has an edge to {self.edges[src]!r}; use a conditional edge"]
            )
        self.edges[src] = dst
        self._validated = None
        return self

    def add_conditional_edge(
        self,
        src: str,
        router: RouterFn,
        targets: list[str] | tuple[str, ...] | None = None,
    ) -> Graph:
        if src in self.edges:
            raise GraphValidationError([f"node {src!r} already has a static edge"])
        if src in self.conditional:
            raise GraphValidationError([f"node {src!r} already has a conditional edge"])
        if not callable(router):
            raise GraphValidationError([f"router for {src!r} is not callable"])
        self.conditional[src] = ConditionalEdge(
            router=router, targets=tuple(targets) if targets is not None else None
        )
        self._validated = None
        return self

    def set_entry(self, name: str) -> Graph:
        self.entry = name
        self._validated = None
        return self

    def set_terminal(self, name: str) -> Graph:
        self.terminals.add(name)
        self._validated = None
        return self

    # -- validation ------------------------------------------------------

    def successors(self, name: str) -> tuple[str, ...] | None:
        """Static successors of a node, or ``None`` if a router may target any node."""
        if name in self.edges:
            return (self.edges[name],)
        if name in self.conditional:
            return self.conditional[name].targets
        return ()

    def validate(self) -> GraphReport:
        """Validate structure. Raises :class:`GraphValidationError` on fatal problems."""
        report = GraphReport(nodes=sorted(self.nodes))
        problems = report.problems

        if not self.nodes:
            problems.append("graph has no nodes")
        if self.entry is None:
            problems.append("no entry node set")
        elif self.entry not in self.nodes:
            problems.append(f"entry node {self.entry!r} is not a node")
        if not self.terminals:
            problems.append("no terminal node set")
        for t in sorted(self.terminals):
            if t not in self.nodes:
                problems.append(f"terminal node {t!r} is not a node")

        for src, dst in self.edges.items():
            if src not in self.nodes:
                problems.append(f"edge source {src!r} is not a node")
            if dst not in self.nodes:
                problems.append(f"edge {src!r} -> {dst!r}: destination is not a node")
        for src, ce in self.conditional.items():
            if src not in self.nodes:
                problems.append(f"conditional edge source {src!r} is not a node")
            for dst in ce.targets or ():
                if dst not in self.nodes:
                    problems.append(f"conditional edge {src!r} -> {dst!r}: target is not a node")

        for name in self.nodes:
            if name in self.terminals:
                continue
            if name not in self.edges and name not in self.conditional:
                problems.append(f"node {name!r} is a dead end: no outgoing edge and not terminal")

        if problems:
            raise GraphValidationError(problems)

        # Reachability from the entry. A router with unknown targets is treated as
        # able to reach every node, because we cannot prove otherwise statically.
        assert self.entry is not None
        reachable: set[str] = set()
        stack = [self.entry]
        while stack:
            n = stack.pop()
            if n in reachable:
                continue
            reachable.add(n)
            succ = self.successors(n)
            nxt = list(self.nodes) if succ is None else list(succ)
            stack.extend(s for s in nxt if s not in reachable)
        report.reachable = sorted(reachable)
        unreachable = sorted(set(self.nodes) - reachable)
        for u in unreachable:
            problems.append(f"node {u!r} is unreachable from entry {self.entry!r}")
        if not any(t in reachable for t in self.terminals):
            problems.append("no terminal node is reachable from the entry")
        if problems:
            raise GraphValidationError(problems)

        report.cycles = self._find_cycles()
        for cyc in report.cycles:
            warnings.warn(
                f"graph {self.name!r} contains a cycle: {' -> '.join(cyc)}",
                CycleWarning,
                stacklevel=2,
            )
        self._validated = report
        return report

    def _find_cycles(self) -> list[list[str]]:
        """Iterative DFS with colouring; returns each back-edge cycle once.

        Only *declared* edges are followed. A router without ``targets`` could
        reach any node, but reporting a cycle through an edge that may not
        exist would be misinformation, so those routers are not expanded here
        (they still count as reaching everything for the reachability check).
        Declare ``targets`` on a conditional edge to have its cycles flagged.
        """
        white, grey, black = 0, 1, 2
        colour = dict.fromkeys(self.nodes, white)
        cycles: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()

        def adj(n: str) -> list[str]:
            succ = self.successors(n)
            return [] if succ is None else list(succ)

        for start in self.nodes:
            if colour[start] != white:
                continue
            path: list[str] = []
            stack: list[tuple[str, Iterator[str]]] = [(start, iter(adj(start)))]
            colour[start] = grey
            path.append(start)
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    if colour[nxt] == white:
                        colour[nxt] = grey
                        path.append(nxt)
                        stack.append((nxt, iter(adj(nxt))))
                        advanced = True
                        break
                    if colour[nxt] == grey:
                        cyc = path[path.index(nxt) :] + [nxt]
                        key = tuple(sorted(set(cyc)))
                        if key not in seen:
                            seen.add(key)
                            cycles.append(cyc)
                if not advanced:
                    colour[node] = black
                    path.pop()
                    stack.pop()
        return cycles

    def build(self) -> Graph:
        """Validate and return self. Use as ``graph = Graph().add_node(...).build()``."""
        self.validate()
        return self

    @property
    def is_validated(self) -> bool:
        return self._validated is not None

    @property
    def report(self) -> GraphReport | None:
        return self._validated

    # -- runtime routing -------------------------------------------------

    async def route(self, state: AgentState) -> str:
        """Compute the next node for ``state.current_node``."""
        node = state.current_node
        if node is None:
            raise RoutingError("state.current_node is None")
        if node in self.edges:
            return self.edges[node]
        ce = self.conditional.get(node)
        if ce is None:
            raise RoutingError(f"node {node!r} has no outgoing edge")
        result = ce.router(state)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, str) or result not in self.nodes:
            raise RoutingError(f"router for {node!r} returned {result!r}, which is not a node")
        if ce.targets is not None and result not in ce.targets:
            raise RoutingError(
                f"router for {node!r} returned {result!r}, not in declared targets {ce.targets}"
            )
        return result


__all__ = [
    "ConditionalEdge",
    "CycleWarning",
    "Graph",
    "GraphReport",
    "GraphValidationError",
    "NodeFn",
    "NodeSpec",
    "RouterFn",
    "RoutingError",
]
