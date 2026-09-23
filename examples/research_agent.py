"""A 4-node research agent: plan -> search (tool) -> synthesise -> verify.

Fully offline. The verifier rejects the first draft and routes back to
``search``; the search node then fires the *same* tool call with the *same*
input, which the runtime detects as a stuck loop and hands to the resolution
hook. The hook finalises the run from the history instead of looping again.

Run:  python examples/research_agent.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from agent_runtime import (
    AgentState,
    DeterministicFallback,
    FallbackChain,
    Graph,
    MemoryCheckpointer,
    MockModel,
    ModelError,
    Output,
    Runtime,
    Step,
    Tracer,
    enforce_schema,
    sanitize_input,
    tool,
)

# --- tools ------------------------------------------------------------------


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str


class SearchResult(BaseModel):
    tenant_id: str
    query: str
    hits: list[SearchHit]


@tool(description="Mock web search that always returns the same two hits.", timeout_s=5.0)
async def web_search(query: str, tenant_id: str, limit: int = 2) -> SearchResult:
    hits = [
        SearchHit(
            title=f"Result {i + 1} for {query}",
            url=f"https://example.invalid/{i + 1}",
            snippet=f"Snippet {i + 1} about {query}.",
        )
        for i in range(limit)
    ]
    return SearchResult(tenant_id=tenant_id, query=query, hits=hits)


# --- models -----------------------------------------------------------------


class Plan(BaseModel):
    query: str = Field(description="the single search query to run")
    rationale: str


# Planner: first answer is malformed JSON, the second one validates (self-correction).
planner = MockModel(
    [
        '{"query": "durable agent runtimes", "rationale": 12}',
        '{"query": "durable agent runtimes", "rationale": "one focused query is enough"}',
    ],
    name="planner-mock",
)

# Synthesiser: primary tier raises, secondary answers, deterministic tail never raises.
synth_chain = FallbackChain(
    [
        MockModel([ModelError("primary provider 503")], name="primary-mock"),
        MockModel(["Durable runtimes checkpoint state after every node so a crash can resume."],
                  name="secondary-mock", cost_per_call_usd=0.0004),
        DeterministicFallback([(r"runtime", "A runtime executes agent graphs.")]),
    ],
    timeout_s=2.0,
)

verifier = MockModel(["REJECT: need one more source"], name="verifier-mock")


# --- nodes ------------------------------------------------------------------


async def plan(state: AgentState) -> AgentState:
    prompt = f"Plan a single search for: {sanitize_input(state.input)}"
    result = await enforce_schema(planner, prompt, Plan, max_retries=2)
    if isinstance(result, Plan):
        state.record(Step(node="plan", output=result.model_dump(), model=planner.name))
    else:
        state.record(Step(node="plan", output=result.model_dump(), status="error",
                          model=planner.name, error="degraded plan"))
        result = Plan(query=state.input, rationale="fallback: use the raw input")
    return state


async def search(state: AgentState) -> AgentState:
    plan_step = next(s for s in state.scratchpad if s.node == "plan" and s.output)
    query = plan_step.output["query"]
    await web_search.run_step(state, query=query, tenant_id=state.tenant_id)
    return state


async def synthesise(state: AgentState) -> AgentState:
    hits = [s.output for s in state.tool_steps() if s.tool == "web_search"][-1]["hits"]
    prompt = "Summarise:\n" + "\n".join(h["snippet"] for h in hits)
    resp = await synth_chain.complete(prompt)
    state.record(
        Step(node="synthesise", output=resp.text, model=resp.model, tier=resp.tier,
             cost_usd=resp.cost_usd)
    )
    return state


async def verify(state: AgentState) -> AgentState:
    draft = [s for s in state.scratchpad if s.node == "synthesise" and s.output][-1].output
    resp = await verifier.complete(f"Verify this draft: {draft}")
    state.record(Step(node="verify", output=resp.text, model=resp.model))
    if not resp.text.startswith("REJECT"):
        state.final = Output(content=draft, data={"verified": True})
    return state


def route_after_verify(state: AgentState) -> str:
    return "done" if state.final is not None else "search"


async def done(state: AgentState) -> AgentState:
    return state


# --- resolution hook: what to do when the agent is stuck ------------------


async def resolve_stuck_loop(state: AgentState, history: list[Step]) -> AgentState:
    drafts = [s.output for s in state.scratchpad if s.node == "synthesise" and s.output]
    state.final = Output(
        content=drafts[-1] if drafts else "No draft produced.",
        data={
            "resolved_by": "loop-break",
            "repeated_tool": history[-1].tool,
            "repeated_input": history[-1].input,
            "repeat_count": len(history),
        },
    )
    return state


def build_graph() -> Graph:
    return (
        Graph("research")
        .add_node("plan", plan, timeout_s=5.0)
        .add_node("search", search, timeout_s=5.0)
        .add_node("synthesise", synthesise, timeout_s=5.0)
        .add_node("verify", verify, timeout_s=5.0)
        .add_node("done", done)
        .add_edge("plan", "search")
        .add_edge("search", "synthesise")
        .add_edge("synthesise", "verify")
        .add_conditional_edge("verify", route_after_verify, targets=["done", "search"])
        .set_entry("plan")
        .set_terminal("done")
        .build()
    )


async def main() -> AgentState:
    graph = build_graph()
    tracer = Tracer(sink=sys.stderr if "--trace" in sys.argv else None)
    runtime = Runtime(checkpointer=MemoryCheckpointer(), tracer=tracer,
                      resolution=resolve_stuck_loop)
    state = AgentState(tenant_id="acme", input="How do durable agent runtimes survive crashes?")

    final = await runtime.run(graph, state)

    print(f"run_id       : {final.run_id}")
    print(f"status       : {final.status}")
    print(f"loop_count   : {final.loop_count}")
    print(f"nodes run    : {final.completed_nodes()}")
    print(f"planner calls: {planner.call_count} (1 self-correction)")
    synth = next(s for s in final.scratchpad if s.node == "synthesise" and s.tier is not None)
    print(f"synth tier   : {synth.tier} ({synth.model})")
    print(f"tool calls   : {[(s.tool, s.input) for s in final.tool_steps()]}")
    print(f"final        : {final.final.content if final.final else None}")
    print(f"resolution   : {json.dumps(final.final.data if final.final else {}, indent=2)}")
    print(f"trace        : {tracer.summary()}")
    return final


if __name__ == "__main__":
    asyncio.run(main())
