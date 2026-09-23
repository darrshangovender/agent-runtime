"""Typed, strictly validated, JSON-serialisable agent state.

Everything the runtime needs to resume a run lives in :class:`AgentState`. Node
functions receive it, mutate it, and return it; the checkpointer persists it as
one JSON document after every node.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StepStatus = Literal["ok", "error", "timeout"]
StepKind = Literal["node", "tool", "model", "note"]
RunStatus = Literal["pending", "running", "completed", "failed", "halted"]
ErrorKind = Literal[
    "node_error",
    "timeout",
    "loop_limit",
    "repeated_tool_call",
    "routing",
    "tool_error",
]


def _now() -> datetime:
    return datetime.now(UTC)


class _Strict(BaseModel):
    """Shared config: unknown fields are rejected and types are not coerced."""

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)


class Step(_Strict):
    """One unit of work recorded in the scratchpad.

    ``kind`` says who wrote it: ``"node"`` steps are written by the runtime once
    per node execution (these drive resume and loop accounting), ``"tool"`` steps
    by :meth:`agent_runtime.tools.Tool.run_step`, and ``"model"``/``"note"`` steps
    by node code itself. ``tier`` records which
    :class:`~agent_runtime.models.FallbackChain` tier answered a model call.
    """

    node: str
    kind: StepKind = "note"
    tool: str | None = None
    input: Any = None
    output: Any = None
    status: StepStatus = "ok"
    duration_ms: float = 0.0
    model: str | None = None
    tier: int | None = None
    cost_usd: float = 0.0
    error: str | None = None
    started_at: datetime = Field(default_factory=_now)


class Output(_Strict):
    """The agent's final answer plus any structured payload."""

    content: str
    data: dict[str, Any] = Field(default_factory=dict)


class RunError(_Strict):
    """Why a run stopped before reaching a terminal node."""

    kind: ErrorKind
    message: str
    node: str | None = None
    exception_type: str | None = None
    occurred_at: datetime = Field(default_factory=_now)


class AgentState(_Strict):
    """The whole state of one agent run.

    * ``scratchpad`` is the append-only history of :class:`Step` records.
    * ``current_node`` is the node the runtime will execute next (or is executing).
    * ``loop_count`` is the largest number of *re-entries* into any single node
      (a node that has run four times has looped three times); when it exceeds
      ``max_loops`` the runtime halts the run with a ``loop_limit`` error.
    """

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tenant_id: str
    input: str
    scratchpad: list[Step] = Field(default_factory=list)
    current_node: str | None = None
    loop_count: int = 0
    max_loops: int = 5
    final: Output | None = None
    error: RunError | None = None
    status: RunStatus = "pending"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    # -- helpers ---------------------------------------------------------

    def record(self, step: Step) -> Step:
        """Append a step to the scratchpad and return it."""
        self.scratchpad.append(step)
        self.updated_at = _now()
        return step

    def tool_steps(self) -> list[Step]:
        """Only the steps that represent tool calls."""
        return [s for s in self.scratchpad if s.kind == "tool" and s.tool is not None]

    def node_steps(self, status: StepStatus | None = None) -> list[Step]:
        """Only the runtime-written node-level steps, optionally filtered by status."""
        return [
            s
            for s in self.scratchpad
            if s.kind == "node" and (status is None or s.status == status)
        ]

    def completed_nodes(self) -> list[str]:
        """Names of nodes that finished successfully, in execution order."""
        return [s.node for s in self.node_steps("ok")]

    def visit_counts(self) -> dict[str, int]:
        """How many times each node has completed successfully."""
        counts: dict[str, int] = {}
        for name in self.completed_nodes():
            counts[name] = counts.get(name, 0) + 1
        return counts

    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.scratchpad), 6)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str | bytes) -> AgentState:
        return cls.model_validate_json(raw)


__all__ = [
    "AgentState",
    "ErrorKind",
    "Output",
    "RunError",
    "RunStatus",
    "Step",
    "StepKind",
    "StepStatus",
]
