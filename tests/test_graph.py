import warnings

import pytest

from agent_runtime import CycleWarning, Graph, GraphValidationError, RoutingError
from tests.conftest import finish, linear_graph, make_state, passthrough


def _two_node() -> Graph:
    return Graph().add_node("a", passthrough).add_node("b", finish)


def test_build_requires_entry():
    with pytest.raises(GraphValidationError, match="no entry"):
        _two_node().add_edge("a", "b").set_terminal("b").build()


def test_build_requires_terminal():
    with pytest.raises(GraphValidationError, match="no terminal"):
        _two_node().add_edge("a", "b").set_entry("a").build()


def test_build_rejects_unreachable_node():
    g = _two_node().add_node("orphan", finish).add_edge("a", "b")
    g.set_entry("a").set_terminal("b").set_terminal("orphan")
    with pytest.raises(GraphValidationError, match="unreachable"):
        g.build()


def test_build_rejects_dead_end():
    g = _two_node().set_entry("a").set_terminal("b")  # a has no outgoing edge
    with pytest.raises(GraphValidationError, match="dead end"):
        g.build()


def test_build_rejects_dangling_edge():
    with pytest.raises(GraphValidationError, match="not a node"):
        _two_node().add_edge("a", "nowhere").set_entry("a").set_terminal("b").build()


def test_build_rejects_unknown_conditional_target():
    g = _two_node().add_conditional_edge("a", lambda s: "b", targets=["b", "ghost"])
    with pytest.raises(GraphValidationError, match="ghost"):
        g.set_entry("a").set_terminal("b").build()


def test_sync_node_function_rejected():
    def not_async(state):
        return state

    with pytest.raises(GraphValidationError, match="async def"):
        Graph().add_node("a", not_async)


def test_duplicate_node_and_double_edge_rejected():
    g = Graph().add_node("a", passthrough)
    with pytest.raises(GraphValidationError, match="duplicate"):
        g.add_node("a", passthrough)
    g.add_node("b", finish).add_edge("a", "b")
    with pytest.raises(GraphValidationError):
        g.add_edge("a", "b")
    with pytest.raises(GraphValidationError):
        g.add_conditional_edge("a", lambda s: "b")


def test_cycles_are_allowed_but_flagged():
    g = (
        Graph("loopy")
        .add_node("a", passthrough)
        .add_node("b", passthrough)
        .add_node("c", finish)
        .add_edge("a", "b")
        .add_conditional_edge("b", lambda s: "c", targets=["a", "c"])
        .set_entry("a")
        .set_terminal("c")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = g.validate()
    assert report.has_cycles
    assert report.cycles == [["a", "b", "a"]]
    assert any(issubclass(w.category, CycleWarning) for w in caught)


def test_router_without_targets_does_not_fabricate_cycles():
    """An untyped router may go anywhere, but the report must not invent a cycle
    through an edge that does not exist (e.g. "plan -> gather -> plan")."""
    g = (
        Graph("untyped")
        .add_node("plan", passthrough)
        .add_node("gather", passthrough)
        .add_node("answer", finish)
        .add_edge("plan", "gather")
        .add_conditional_edge("gather", lambda s: "answer")
        .set_entry("plan")
        .set_terminal("answer")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = g.validate()
    assert report.cycles == []
    assert not any(issubclass(w.category, CycleWarning) for w in caught)
    assert report.reachable == ["answer", "gather", "plan"]  # still reachable via router
    # declaring targets makes the real self-loop visible
    g.conditional["gather"].targets = ("answer", "gather")
    assert g.validate().cycles == [["gather", "gather"]]


def test_acyclic_graph_reports_no_cycles():
    g = linear_graph("a", "b", "c")
    assert g.report is not None and not g.report.has_cycles
    assert g.report.reachable == ["a", "b", "c"]


async def test_route_static_and_conditional():
    g = (
        Graph()
        .add_node("a", passthrough)
        .add_node("b", passthrough)
        .add_node("c", finish)
        .add_edge("a", "b")
        .add_conditional_edge("b", lambda s: "c" if s.loop_count == 0 else "a")
        .set_entry("a")
        .set_terminal("c")
        .build()
    )
    s = make_state(current_node="a")
    assert await g.route(s) == "b"
    s.current_node = "b"
    assert await g.route(s) == "c"
    s.loop_count = 1
    assert await g.route(s) == "a"


async def test_route_async_router_and_bad_return():
    async def router(state):
        return "zzz"

    g = (
        Graph()
        .add_node("a", passthrough)
        .add_node("b", finish)
        .add_conditional_edge("a", router)
        .set_entry("a")
        .set_terminal("b")
        .build()
    )
    with pytest.raises(RoutingError, match="not a node"):
        await g.route(make_state(current_node="a"))


async def test_route_enforces_declared_targets():
    g = (
        Graph()
        .add_node("a", passthrough)
        .add_node("b", finish)
        .add_node("c", finish)
        .add_conditional_edge("a", lambda s: "c", targets=["b"])
        .set_entry("a")
        .set_terminal("b")
    )
    # "c" is unreachable statically, so build fails - validate the router path directly
    with pytest.raises(GraphValidationError):
        g.build()
    g.add_edge("b", "c")
    g.terminals = {"c"}
    g.build()
    with pytest.raises(RoutingError, match="declared targets"):
        await g.route(make_state(current_node="a"))
