"""Shared fixtures: tiny graphs and node factories."""

from __future__ import annotations

import warnings
from itertools import pairwise

import pytest

from agent_runtime import AgentState, CycleWarning, Graph, Output, Step


@pytest.fixture(autouse=True)
def _quiet_cycle_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CycleWarning)
        yield


def make_state(**kw) -> AgentState:
    base = {"tenant_id": "acme", "input": "hello"}
    base.update(kw)
    return AgentState(**base)


async def passthrough(state: AgentState) -> AgentState:
    return state


async def finish(state: AgentState) -> AgentState:
    state.final = Output(content="done")
    return state


def touching(name: str):
    """Node that records a marker step so tests can see it ran."""

    async def node(state: AgentState) -> AgentState:
        state.record(Step(node=name, output={"touched": name}))
        return state

    node.__name__ = name
    return node


def linear_graph(*names: str) -> Graph:
    """a -> b -> ... -> last (terminal). Every node is a touching node."""
    g = Graph("linear")
    for n in names:
        g.add_node(n, touching(n))
    for a, b in pairwise(names):
        g.add_edge(a, b)
    g.set_entry(names[0]).set_terminal(names[-1])
    return g.build()


@pytest.fixture
def state() -> AgentState:
    return make_state()
