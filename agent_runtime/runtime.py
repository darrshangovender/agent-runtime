"""The executor.

:class:`Runtime` drives an :class:`AgentState` through a :class:`Graph`:

* executes the current node under a per-node timeout,
* records a node-level :class:`Step`,
* checkpoints after every node,
* enforces ``max_loops`` on revisits,
* detects an identical ``(tool, input)`` fired twice in a row and hands control
  to a *resolution hook* with the repeated history,
* and can :meth:`resume` a run from its last checkpoint without re-executing
  nodes that already completed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from .checkpoint import Checkpointer, MemoryCheckpointer
from .graph import Graph, RoutingError
from .state import AgentState, ErrorKind, Output, RunError, RunStatus, Step
from .telemetry import Tracer, current_tracer

ResolutionHook = Callable[[AgentState, list[Step]], Awaitable[AgentState]]

RESOLUTION_MARKER = "__resolution__"

_Verdict = Literal["stop", "redirect", "proceed"]


class RunFailedError(RuntimeError):
    """Raised by :meth:`Runtime.run` when ``raise_on_error=True`` and the run fails."""

    def __init__(self, state: AgentState) -> None:
        self.state = state
        err = state.error
        detail = f"{err.kind} at {err.node!r}: {err.message}" if err else "unknown"
        super().__init__(f"run {state.run_id} failed: {detail}")


class UnknownRunError(KeyError):
    """No checkpoint exists for the requested ``run_id``."""


def find_consecutive_repeat(steps: list[Step]) -> list[Step]:
    """Return the two most recent tool steps if they are the same ``(tool, input)``.

    Only tool steps are considered (node-level steps are ignored) and a
    :data:`RESOLUTION_MARKER` step resets the comparison so a resolved repeat
    is not reported again.
    """
    tool_steps = [s for s in steps if s.kind == "tool" and s.tool is not None]
    if len(tool_steps) < 2:
        return []
    last, prev = tool_steps[-1], tool_steps[-2]
    if last.tool == RESOLUTION_MARKER or prev.tool == RESOLUTION_MARKER:
        return []
    if last.tool == prev.tool and last.input == prev.input:
        return [prev, last]
    return []


class Runtime:
    """Executes graphs. Construct once, run many.

    ``default_timeout_s`` applies to nodes that do not set their own timeout.
    ``resolution`` is called when an identical tool call repeats; if omitted the
    run halts with a ``repeated_tool_call`` error instead of looping forever.
    """

    def __init__(
        self,
        *,
        checkpointer: Checkpointer | None = None,
        tracer: Tracer | None = None,
        default_timeout_s: float | None = 30.0,
        resolution: ResolutionHook | None = None,
        raise_on_error: bool = False,
        graph: Graph | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.tracer = tracer or Tracer()
        self.default_timeout_s = default_timeout_s
        self.resolution = resolution
        self.raise_on_error = raise_on_error
        self.graph = graph

    # -- public API ------------------------------------------------------

    async def run(
        self,
        graph: Graph,
        state: AgentState,
        checkpointer: Checkpointer | None = None,
    ) -> AgentState:
        """Execute ``state`` through ``graph`` until a terminal node or a limit."""
        if not graph.is_validated:
            graph.validate()
        self.graph = graph
        cp = self._checkpointer(checkpointer)
        if state.current_node is None:
            state.current_node = graph.entry
        elif state.current_node not in graph.nodes:
            raise ValueError(
                f"state.current_node {state.current_node!r} is not a node of graph "
                f"{graph.name!r}; pass a fresh AgentState or use resume()"
            )
        state.status = "running"
        state.error = None
        await cp.save(state)
        return await self._drive(graph, state, cp)

    async def resume(
        self,
        run_id: str,
        checkpointer: Checkpointer | None = None,
        graph: Graph | None = None,
    ) -> AgentState:
        """Continue a run from its last checkpoint.

        Nodes recorded as completed in the checkpointed scratchpad are not
        executed again; execution restarts at ``state.current_node``, which is
        the node that was running (or about to run) when the checkpoint was
        written. A completed run is returned untouched.
        """
        cp = self._checkpointer(checkpointer)
        g = graph or self.graph
        if g is None:
            raise ValueError("resume needs a graph: pass graph= or construct Runtime(graph=...)")
        if not g.is_validated:
            g.validate()
        self.graph = g
        state = await cp.load(run_id)
        if state is None:
            raise UnknownRunError(run_id)
        if state.status == "completed":
            return state
        if state.current_node is None or state.current_node not in g.nodes:
            state.current_node = g.entry
        # A routing failure happens *after* current_node completed (its "ok"
        # step is already in the scratchpad), so re-executing it would run the
        # node twice. Pick up at the routing step instead.
        route_first = state.error is not None and state.error.kind == "routing"
        state.error = None
        state.status = "running"
        await cp.save(state)
        return await self._drive(g, state, cp, route_first=route_first)

    # -- internals -------------------------------------------------------

    def _checkpointer(self, override: Checkpointer | None) -> Checkpointer:
        if override is not None:
            return override
        if self.checkpointer is None:
            self.checkpointer = MemoryCheckpointer()
        return self.checkpointer

    async def _fail(
        self,
        state: AgentState,
        cp: Checkpointer,
        *,
        kind: ErrorKind,
        message: str,
        node: str | None,
        exception_type: str | None = None,
        status: RunStatus = "failed",
    ) -> AgentState:
        state.error = RunError(kind=kind, message=message, node=node, exception_type=exception_type)
        state.status = status
        await cp.save(state)
        self.tracer.event("run.failed", run_id=state.run_id, kind=kind, node=node, message=message)
        if self.raise_on_error:
            raise RunFailedError(state)
        return state

    async def _drive(
        self,
        graph: Graph,
        state: AgentState,
        cp: Checkpointer,
        *,
        route_first: bool = False,
    ) -> AgentState:
        token = current_tracer.set(self.tracer)
        try:
            with self.tracer.span(f"run:{graph.name}", "run", run_id=state.run_id) as run_span:
                result = await self._loop(graph, state, cp, route_first=route_first)
                run_span.set(status=result.status, steps=len(result.scratchpad))
                return result
        finally:
            current_tracer.reset(token)

    async def _loop(
        self,
        graph: Graph,
        state: AgentState,
        cp: Checkpointer,
        *,
        route_first: bool = False,
    ) -> AgentState:
        visits: dict[str, int] = state.visit_counts()
        execute = not route_first

        while True:
            node = state.current_node
            assert node is not None

            if execute:
                state, verdict = await self._run_node(graph, state, cp, node, visits)
                if verdict == "stop":
                    return state
                if verdict == "redirect":
                    continue
            execute = True

            # Terminal reached: done.
            if node in graph.terminals:
                if state.final is None:
                    state.final = Output(content="", data={"note": "terminal reached without final"})
                state.status = "completed"
                await cp.save(state)
                return state

            # Route to the next node and (d) checkpoint.
            try:
                nxt = await graph.route(state)
            except RoutingError as exc:
                return await self._fail(
                    state, cp, kind="routing", node=node, message=str(exc),
                    exception_type=type(exc).__name__,
                )
            state.current_node = nxt
            await cp.save(state)

    async def _run_node(
        self,
        graph: Graph,
        state: AgentState,
        cp: Checkpointer,
        node: str,
        visits: dict[str, int],
    ) -> tuple[AgentState, _Verdict]:
        """Execute ``node`` once: loop limit, timeout, step record, repeat detection.

        Returns the (possibly replaced) state and what the loop should do next:
        ``"stop"`` (run finished or failed), ``"redirect"`` (a resolution hook
        moved ``current_node``; start there) or ``"proceed"`` (check terminal /
        route as normal).
        """
        spec = graph.nodes[node]

        # (a) loop limit: loop_count is the most re-entries any single node has
        # had. Entering a node for the (max_loops + 2)th time trips the limit.
        visits[node] = visits.get(node, 0) + 1
        state.loop_count = max(state.loop_count, visits[node] - 1)
        if state.loop_count > state.max_loops:
            failed = await self._fail(
                state,
                cp,
                kind="loop_limit",
                message=(
                    f"loop_count {state.loop_count} exceeded max_loops "
                    f"{state.max_loops} re-entering {node!r}"
                ),
                node=node,
                status="halted",
            )
            return failed, "stop"

        # (c) per-node timeout. ``asyncio.timeout`` lets us tell *our* budget
        # expiring apart from a TimeoutError the node raised itself (for example
        # a ToolTimeoutError from a tool with its own, shorter budget).
        timeout = spec.timeout_s if spec.timeout_s is not None else self.default_timeout_s
        t0 = time.perf_counter()
        with self.tracer.span(f"node:{node}", "node", node=node, timeout_s=timeout) as span:
            budget = asyncio.timeout(timeout)
            try:
                async with budget:
                    new_state = await spec.fn(state)
            except Exception as exc:  # noqa: BLE001 - node errors become RunError
                if isinstance(exc, TimeoutError) and budget.expired():
                    span.set(status="timeout")
                    state.record(
                        Step(node=node, kind="node", status="timeout", duration_ms=_ms(t0),
                             error=f"node exceeded {timeout}s")
                    )
                    failed = await self._fail(
                        state, cp, kind="timeout", node=node,
                        message=f"node {node!r} exceeded {timeout}s",
                        exception_type="TimeoutError",
                    )
                    return failed, "stop"
                span.set(status="error", error=f"{type(exc).__name__}: {exc}")
                state.record(
                    Step(node=node, kind="node", status="error", duration_ms=_ms(t0),
                         error=f"{type(exc).__name__}: {exc}")
                )
                failed = await self._fail(
                    state, cp, kind="node_error", node=node, message=str(exc) or repr(exc),
                    exception_type=type(exc).__name__,
                )
                return failed, "stop"

        if not isinstance(new_state, AgentState):
            state.record(
                Step(node=node, kind="node", status="error", duration_ms=_ms(t0),
                     error=f"node returned {type(new_state).__name__}, expected AgentState")
            )
            failed = await self._fail(
                state, cp, kind="node_error", node=node,
                message=f"node {node!r} returned {type(new_state).__name__}, not AgentState",
                exception_type="TypeError",
            )
            return failed, "stop"
        state = new_state
        state.record(Step(node=node, kind="node", status="ok", duration_ms=_ms(t0)))

        # (b) identical consecutive tool call -> resolution hook.
        repeats = find_consecutive_repeat(state.scratchpad)
        if not repeats:
            return state, "proceed"
        self.tracer.event(
            "tool.repeat", run_id=state.run_id, node=node,
            tool=repeats[-1].tool, input=repeats[-1].input,
        )
        if self.resolution is None:
            failed = await self._fail(
                state, cp, kind="repeated_tool_call", node=node, status="halted",
                message=(
                    f"tool {repeats[-1].tool!r} called twice consecutively with "
                    f"identical input and no resolution hook is configured"
                ),
            )
            return failed, "stop"
        with self.tracer.span("resolution", "node", node=node, tool=repeats[-1].tool):
            t1 = time.perf_counter()
            try:
                state = await self.resolution(state, list(repeats))
            except Exception as exc:  # noqa: BLE001 - hook errors become RunError
                failed = await self._fail(
                    state, cp, kind="node_error", node=node,
                    message=f"resolution hook failed: {exc}",
                    exception_type=type(exc).__name__,
                )
                return failed, "stop"
            state.record(
                Step(
                    node=node, kind="tool", tool=RESOLUTION_MARKER,
                    input={"tool": repeats[-1].tool, "repeats": len(repeats)},
                    output={"final": state.final is not None,
                            "current_node": state.current_node},
                    duration_ms=_ms(t1),
                )
            )
        if state.final is not None:
            state.status = "completed"
            await cp.save(state)
            return state, "stop"
        if state.current_node != node and state.current_node in graph.nodes:
            # The hook redirected the run; checkpoint and continue there.
            await cp.save(state)
            return state, "redirect"
        return state, "proceed"


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


__all__ = [
    "RESOLUTION_MARKER",
    "ResolutionHook",
    "RunFailedError",
    "Runtime",
    "UnknownRunError",
    "find_consecutive_repeat",
]
