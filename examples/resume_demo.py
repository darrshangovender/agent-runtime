"""Crash mid-run, then resume from SQLite without re-executing finished nodes.

Node 3 (``enrich``) raises the first time it runs. The runtime records the
failure, checkpoints, and returns a failed state. A *fresh* Runtime then
resumes the same ``run_id`` from the SQLite file, re-runs only ``enrich`` and
``finish``, and completes. Nodes 1 and 2 keep a single ``ok`` step each.

Run:  python examples/resume_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_runtime import (
    AgentState,
    Graph,
    Output,
    Runtime,
    SQLiteCheckpointer,
    Step,
    tool,
)

CRASH_NEXT = {"enrich": True}  # flipped off after the first crash


@tool(description="Deterministic lookup.", timeout_s=2.0)
async def lookup(key: str) -> dict:
    return {"key": key, "value": f"value-for-{key}"}


async def ingest(state: AgentState) -> AgentState:
    await lookup.run_step(state, key="ingest")
    return state


async def analyse(state: AgentState) -> AgentState:
    await lookup.run_step(state, key="analyse")
    state.record(Step(node="analyse", output={"analysis": "3 findings"}))
    return state


async def enrich(state: AgentState) -> AgentState:
    if CRASH_NEXT["enrich"]:
        CRASH_NEXT["enrich"] = False
        raise ConnectionError("simulated crash: upstream enrichment API unreachable")
    await lookup.run_step(state, key="enrich")
    return state


async def finish(state: AgentState) -> AgentState:
    state.final = Output(content="pipeline complete", data={"nodes": state.completed_nodes()})
    return state


def build_graph() -> Graph:
    return (
        Graph("resume-demo")
        .add_node("ingest", ingest)
        .add_node("analyse", analyse)
        .add_node("enrich", enrich)
        .add_node("finish", finish)
        .add_edge("ingest", "analyse")
        .add_edge("analyse", "enrich")
        .add_edge("enrich", "finish")
        .set_entry("ingest")
        .set_terminal("finish")
        .build()
    )


def _db_path() -> Path:
    env = os.environ.get("AGENT_RUNTIME_DB")
    if env:
        Path(env).parent.mkdir(parents=True, exist_ok=True)
        return Path(env)
    return Path(tempfile.gettempdir()) / "agent_runtime_resume_demo.db"


async def main() -> AgentState:
    db = _db_path()
    graph = build_graph()

    # --- first attempt: crashes at node 3 ---------------------------------
    with SQLiteCheckpointer(db) as cp:
        runtime = Runtime(checkpointer=cp)
        state = AgentState(tenant_id="acme", input="resume me")
        crashed = await runtime.run(graph, state)
        run_id = crashed.run_id
        print(f"db           : {db}")
        print(f"run_id       : {run_id}")
        print(f"first status : {crashed.status}")
        print(f"error        : {crashed.error.kind if crashed.error else None} "
              f"-> {crashed.error.message if crashed.error else ''}")
        print(f"completed    : {crashed.completed_nodes()}")
        print(f"checkpoints  : {len(await cp.history(run_id))}")

    # --- second process: brand-new runtime, resumes from disk -------------
    with SQLiteCheckpointer(db) as cp2:
        runtime2 = Runtime(checkpointer=cp2, graph=graph)
        resumed = await runtime2.resume(run_id)
        ok_nodes = resumed.completed_nodes()
        print(f"resumed      : {resumed.status}")
        print(f"completed    : {ok_nodes}")
        per_node = {n: ok_nodes.count(n) for n in dict.fromkeys(ok_nodes)}
        print(f"ok per node  : {per_node}")
        print(f"final        : {resumed.final.content if resumed.final else None}")
        print(f"checkpoints  : {len(await cp2.history(run_id))}")
        assert ok_nodes.count("ingest") == 1 and ok_nodes.count("analyse") == 1
        assert resumed.status == "completed"
        await cp2.delete(run_id)
    return resumed


if __name__ == "__main__":
    asyncio.run(main())
