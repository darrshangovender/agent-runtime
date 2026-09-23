import json

import pytest
from pydantic import ValidationError

from agent_runtime import AgentState, Output, RunError, Step
from tests.conftest import make_state


def test_state_defaults():
    s = make_state()
    assert len(s.run_id) == 32
    assert s.loop_count == 0 and s.max_loops == 5
    assert s.status == "pending" and s.final is None and s.error is None


def test_state_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        AgentState(tenant_id="t", input="x", bogus=1)


def test_state_is_strict_about_types():
    with pytest.raises(ValidationError):
        AgentState(tenant_id="t", input="x", loop_count="3")


def test_step_status_literal():
    with pytest.raises(ValidationError):
        Step(node="n", status="weird")


def test_json_round_trip_preserves_everything():
    s = make_state()
    s.record(Step(node="a", tool="search", input={"q": "x"}, output=[1, 2], duration_ms=1.5,
                  model="m", tier=1, cost_usd=0.001))
    s.final = Output(content="ok", data={"k": "v"})
    s.error = RunError(kind="timeout", message="slow", node="a")
    raw = s.to_json()
    json.loads(raw)  # valid JSON
    back = AgentState.from_json(raw)
    assert back == s
    assert back.scratchpad[0].started_at == s.scratchpad[0].started_at


def test_helpers_filter_steps():
    s = make_state()
    s.record(Step(node="a", kind="node"))
    s.record(Step(node="a", kind="tool", tool="t", input=1))
    s.record(Step(node="a", output="a note"))  # default kind="note", ignored by both filters
    s.record(Step(node="b", kind="node", status="error"))
    assert [x.tool for x in s.tool_steps()] == ["t"]
    assert s.completed_nodes() == ["a"]
    assert len(s.node_steps()) == 2
    assert s.visit_counts() == {"a": 1}


def test_total_cost_sums_steps():
    s = make_state()
    s.record(Step(node="a", cost_usd=0.001))
    s.record(Step(node="b", cost_usd=0.002))
    assert s.total_cost_usd() == pytest.approx(0.003)
