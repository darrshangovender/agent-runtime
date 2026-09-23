import asyncio

import pytest

from agent_runtime import (
    RESOLUTION_MARKER,
    AgentState,
    Graph,
    MemoryCheckpointer,
    Output,
    RunFailedError,
    Runtime,
    Step,
    UnknownRunError,
    find_consecutive_repeat,
    tool,
)
from agent_runtime.tools import ToolRegistry
from tests.conftest import finish, linear_graph, make_state, passthrough, touching

reg = ToolRegistry()


@tool(name="echo", registry=reg)
async def echo(q: str) -> dict:
    return {"q": q}


def looping_graph(router):
    return (
        Graph("loop")
        .add_node("work", touching("work"))
        .add_node("check", passthrough)
        .add_node("end", finish)
        .add_edge("work", "check")
        .add_conditional_edge("check", router, targets=["work", "end"])
        .set_entry("work")
        .set_terminal("end")
        .build()
    )


async def test_linear_run_completes_and_records_node_steps():
    g = linear_graph("a", "b", "c")
    out = await Runtime().run(g, make_state())
    assert out.status == "completed"
    assert out.completed_nodes() == ["a", "b", "c"]
    assert out.current_node == "c"
    assert out.final is not None  # runtime fills a placeholder when nodes set none


async def test_checkpoint_written_after_every_node():
    g = linear_graph("a", "b", "c")
    cp = MemoryCheckpointer()
    out = await Runtime(checkpointer=cp).run(g, make_state())
    hist = await cp.history(out.run_id)
    # initial + one per node transition + final
    assert [h.node for h in hist] == ["a", "b", "c", "c"]
    assert hist[-1].status == "completed"


async def test_loop_limit_halts_run():
    g = looping_graph(lambda s: "work")  # never finishes
    out = await Runtime().run(g, make_state(max_loops=3))
    assert out.status == "halted"
    assert out.error is not None and out.error.kind == "loop_limit"
    assert out.loop_count == 4  # 3 allowed revisits, 4th trips the limit
    assert out.completed_nodes().count("work") == 4


async def test_loop_under_limit_completes():
    counter = {"n": 0}

    def router(s):
        counter["n"] += 1
        return "end" if counter["n"] >= 3 else "work"

    out = await Runtime().run(looping_graph(router), make_state(max_loops=5))
    assert out.status == "completed"
    assert out.loop_count == 2


async def test_identical_tool_call_without_hook_halts():
    async def call_tool(state):
        await echo.run_step(state, q="same")
        return state

    g = (
        Graph()
        .add_node("t", call_tool)
        .add_node("end", finish)
        .add_conditional_edge("t", lambda s: "t", targets=["t", "end"])
        .set_entry("t")
        .set_terminal("end")
        .build()
    )
    out = await Runtime().run(g, make_state())
    assert out.status == "halted"
    assert out.error is not None and out.error.kind == "repeated_tool_call"
    assert len([s for s in out.tool_steps() if s.tool == "echo"]) == 2


async def test_identical_tool_call_invokes_resolution_hook_with_history():
    seen = {}

    async def hook(state: AgentState, history: list[Step]) -> AgentState:
        seen["history"] = history
        state.final = Output(content="resolved")
        return state

    async def call_tool(state):
        await echo.run_step(state, q="same")
        return state

    g = (
        Graph()
        .add_node("t", call_tool)
        .add_node("end", finish)
        .add_conditional_edge("t", lambda s: "t", targets=["t", "end"])
        .set_entry("t")
        .set_terminal("end")
        .build()
    )
    out = await Runtime(resolution=hook).run(g, make_state())
    assert out.status == "completed"
    assert out.final and out.final.content == "resolved"
    assert [s.input for s in seen["history"]] == [{"q": "same"}, {"q": "same"}]
    assert out.scratchpad[-1].tool == RESOLUTION_MARKER


async def test_resolution_hook_can_redirect_instead_of_finalising():
    async def hook(state, history):
        state.current_node = "end"
        return state

    async def call_tool(state):
        await echo.run_step(state, q="x")
        return state

    g = (
        Graph()
        .add_node("t", call_tool)
        .add_node("end", finish)
        .add_conditional_edge("t", lambda s: "t", targets=["t", "end"])
        .set_entry("t")
        .set_terminal("end")
        .build()
    )
    out = await Runtime(resolution=hook).run(g, make_state())
    assert out.status == "completed" and out.final.content == "done"


async def test_different_inputs_do_not_trigger_detection():
    n = {"i": 0}

    async def call_tool(state):
        n["i"] += 1
        await echo.run_step(state, q=f"q{n['i']}")
        return state

    g = (
        Graph()
        .add_node("t", call_tool)
        .add_node("end", finish)
        .add_conditional_edge("t", lambda s: "end" if n["i"] >= 3 else "t")
        .set_entry("t")
        .set_terminal("end")
        .build()
    )
    out = await Runtime().run(g, make_state())
    assert out.status == "completed"


def test_find_consecutive_repeat_ignores_node_steps_and_marker():
    a = Step(node="n", kind="tool", tool="t", input=1)
    b = Step(node="n", kind="node")  # node-level step in between
    c = Step(node="n", kind="tool", tool="t", input=1)
    assert find_consecutive_repeat([a, b, c]) == [a, c]
    marker = Step(node="n", kind="tool", tool=RESOLUTION_MARKER)
    assert find_consecutive_repeat([a, marker, c]) == []
    assert find_consecutive_repeat([a]) == []
    assert find_consecutive_repeat([a, Step(node="n", kind="tool", tool="t", input=2)]) == []


async def test_node_timeout_fails_run_with_timeout_error():
    async def slow(state):
        await asyncio.sleep(5)
        return state

    g = (
        Graph()
        .add_node("slow", slow, timeout_s=0.05)
        .add_node("end", finish)
        .add_edge("slow", "end")
        .set_entry("slow")
        .set_terminal("end")
        .build()
    )
    out = await Runtime().run(g, make_state())
    assert out.status == "failed"
    assert out.error.kind == "timeout" and out.error.node == "slow"
    assert out.scratchpad[-1].status == "timeout"


async def test_default_timeout_applies_when_node_has_none():
    async def slow(state):
        await asyncio.sleep(5)
        return state

    g = Graph().add_node("s", slow).add_node("e", finish).add_edge("s", "e")
    g.set_entry("s").set_terminal("e").build()
    out = await Runtime(default_timeout_s=0.05).run(g, make_state())
    assert out.error.kind == "timeout"


async def test_tool_timeout_inside_node_is_a_node_error_not_a_node_timeout():
    """A ToolTimeoutError raised *inside* a node must not be mistaken for the node's
    own budget expiring: the node ran for 0.02s, not the 30s the message would claim."""

    @tool(name="slow_tool", registry=reg, timeout_s=0.02)
    async def slow_tool() -> int:
        await asyncio.sleep(2)
        return 1

    async def n(state):
        await slow_tool.run_step(state)
        return state

    g = Graph().add_node("n", n).add_node("e", finish).add_edge("n", "e")
    g.set_entry("n").set_terminal("e").build()
    out = await Runtime(default_timeout_s=30).run(g, make_state())
    assert out.status == "failed"
    assert out.error.kind == "node_error"
    assert out.error.exception_type == "ToolTimeoutError"
    assert "exceeded 30" not in out.error.message
    assert out.node_steps()[-1].status == "error"


async def test_node_budget_still_wins_when_node_raises_nothing():
    """The plain node-timeout path is unchanged by the ToolTimeoutError distinction."""

    async def slow(state):
        await asyncio.sleep(5)
        return state

    g = Graph().add_node("s", slow, timeout_s=0.02).add_node("e", finish).add_edge("s", "e")
    g.set_entry("s").set_terminal("e").build()
    out = await Runtime().run(g, make_state())
    assert out.error.kind == "timeout" and out.error.exception_type == "TimeoutError"
    assert out.node_steps()[-1].status == "timeout"


async def test_node_exception_is_recorded_and_checkpointed():
    async def boom(state):
        raise ValueError("kaboom")

    g = Graph().add_node("b", boom).add_node("e", finish).add_edge("b", "e")
    g.set_entry("b").set_terminal("e").build()
    cp = MemoryCheckpointer()
    out = await Runtime(checkpointer=cp).run(g, make_state())
    assert out.status == "failed"
    assert out.error.kind == "node_error" and out.error.exception_type == "ValueError"
    saved = await cp.load(out.run_id)
    assert saved.error.message == "kaboom"


async def test_raise_on_error_raises_run_failed():
    async def boom(state):
        raise ValueError("x")

    g = Graph().add_node("b", boom).add_node("e", finish).add_edge("b", "e")
    g.set_entry("b").set_terminal("e").build()
    with pytest.raises(RunFailedError) as ei:
        await Runtime(raise_on_error=True).run(g, make_state())
    assert ei.value.state.error.kind == "node_error"


async def test_node_returning_wrong_type_fails():
    async def bad(state):
        return {"not": "state"}

    g = Graph().add_node("b", bad).add_node("e", finish).add_edge("b", "e")
    g.set_entry("b").set_terminal("e").build()
    out = await Runtime().run(g, make_state())
    assert out.error.exception_type == "TypeError"


async def test_routing_error_fails_run():
    g = (
        Graph()
        .add_node("a", passthrough)
        .add_node("e", finish)
        .add_conditional_edge("a", lambda s: "nope")
        .set_entry("a")
        .set_terminal("e")
        .build()
    )
    out = await Runtime().run(g, make_state())
    assert out.error.kind == "routing"


async def test_resume_does_not_reexecute_completed_nodes():
    crash = {"on": True}
    executions: list[str] = []

    async def c(state):
        executions.append("c")
        if crash["on"]:
            crash["on"] = False
            raise RuntimeError("crash")
        state.record(Step(node="c", output="ok"))
        return state

    g = (
        Graph()
        .add_node("a", touching("a"))
        .add_node("b", touching("b"))
        .add_node("c", c)
        .add_node("d", finish)
        .add_edge("a", "b")
        .add_edge("b", "c")
        .add_edge("c", "d")
        .set_entry("a")
        .set_terminal("d")
        .build()
    )
    cp = MemoryCheckpointer()
    first = await Runtime(checkpointer=cp).run(g, make_state())
    assert first.status == "failed" and first.current_node == "c"

    resumed = await Runtime(checkpointer=cp).resume(first.run_id, graph=g)
    assert resumed.status == "completed"
    assert resumed.run_id == first.run_id
    ok = resumed.completed_nodes()
    assert ok.count("a") == 1 and ok.count("b") == 1 and ok.count("c") == 1
    assert executions == ["c", "c"]
    assert resumed.loop_count == 0  # retrying a failed node is not a loop
    assert resumed.error is None


async def test_resume_after_routing_failure_does_not_reexecute_completed_node():
    """A routing error happens after the node finished, so resume must route from it,
    not run it a second time (its side effects would otherwise happen twice)."""
    executions: list[str] = []
    broken = {"router": True}

    async def a(state):
        executions.append("a")
        state.record(Step(node="a", output="side effect"))
        return state

    def router(s):
        return "nowhere" if broken["router"] else "e"

    g = (
        Graph()
        .add_node("a", a)
        .add_node("e", finish)
        .add_conditional_edge("a", router)
        .set_entry("a")
        .set_terminal("e")
        .build()
    )
    cp = MemoryCheckpointer()
    first = await Runtime(checkpointer=cp).run(g, make_state())
    assert first.status == "failed" and first.error.kind == "routing"
    assert first.completed_nodes() == ["a"]

    broken["router"] = False  # "fix the router" and resume
    resumed = await Runtime(checkpointer=cp, graph=g).resume(first.run_id)
    assert resumed.status == "completed"
    assert resumed.completed_nodes() == ["a", "e"]
    assert executions == ["a"]
    assert resumed.error is None


async def test_resume_after_node_error_reruns_only_that_node_from_sqlite(tmp_path):
    """End-to-end crash/resume against the durable checkpointer: the crashed node runs
    again, the finished ones do not, and history keeps one row per seq."""
    from agent_runtime import SQLiteCheckpointer

    crash = {"on": True}
    runs: list[str] = []

    async def b(state):
        runs.append("b")
        if crash["on"]:
            crash["on"] = False
            raise ConnectionError("upstream down")
        return state

    g = (
        Graph()
        .add_node("a", touching("a"))
        .add_node("b", b)
        .add_node("c", finish)
        .add_edge("a", "b")
        .add_edge("b", "c")
        .set_entry("a")
        .set_terminal("c")
        .build()
    )
    db = tmp_path / "resume.db"
    with SQLiteCheckpointer(db) as cp:
        failed = await Runtime(checkpointer=cp).run(g, make_state())
        assert failed.status == "failed" and failed.current_node == "b"
    with SQLiteCheckpointer(db) as cp2:
        resumed = await Runtime(checkpointer=cp2, graph=g).resume(failed.run_id)
        seqs = [h.seq for h in await cp2.history(failed.run_id)]
    assert resumed.status == "completed"
    assert resumed.completed_nodes() == ["a", "b", "c"]
    assert runs == ["b", "b"]
    assert seqs == sorted(set(seqs))


async def test_run_rejects_state_whose_current_node_is_not_in_graph():
    g = linear_graph("a", "b")
    with pytest.raises(ValueError, match="not a node"):
        await Runtime().run(g, make_state(current_node="zzz"))


async def test_resume_completed_run_is_noop():
    g = linear_graph("a", "b")
    cp = MemoryCheckpointer()
    done = await Runtime(checkpointer=cp).run(g, make_state())
    again = await Runtime(checkpointer=cp, graph=g).resume(done.run_id)
    assert again == done


async def test_resume_unknown_run_raises():
    with pytest.raises(UnknownRunError):
        await Runtime(graph=linear_graph("a")).resume("nope", MemoryCheckpointer())


async def test_resume_requires_graph():
    cp = MemoryCheckpointer()
    with pytest.raises(ValueError, match="graph"):
        await Runtime().resume("x", cp)


async def test_tracer_records_run_and_node_spans():
    g = linear_graph("a", "b")
    rt = Runtime()
    await rt.run(g, make_state())
    kinds = [s.kind for s in rt.tracer.spans]
    assert kinds.count("node") == 2 and kinds.count("run") == 1
    assert rt.tracer.spans[-1].kind == "run"  # run span closes last
