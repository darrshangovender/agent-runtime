import asyncio

import pytest
from pydantic import BaseModel

from agent_runtime import (
    Tool,
    ToolInputError,
    ToolNotAllowedError,
    ToolNotFoundError,
    ToolOutputError,
    ToolRegistry,
    ToolTimeoutError,
    tool,
)
from tests.conftest import make_state


class Q(BaseModel):
    text: str
    limit: int = 3


class R(BaseModel):
    hits: list[str]


@pytest.fixture
def reg():
    return ToolRegistry()


def test_decorator_builds_input_model_from_signature(reg):
    @tool(registry=reg, description="add")
    def add(a: int, b: int = 1) -> int:
        return a + b

    assert isinstance(add, Tool)
    schema = add.input_schema()
    assert set(schema["properties"]) == {"a", "b"}
    assert schema["required"] == ["a"]
    assert add.output_model is None
    assert "add" in reg and reg.get("add") is add


async def test_single_model_parameter_and_output_validation(reg):
    @tool(registry=reg)
    async def search(q: Q) -> R:
        return R(hits=[q.text] * q.limit)

    assert search.input_model is Q and search.output_model is R
    out = await search.invoke({"text": "x", "limit": 2}, tenant_id="t")
    assert out == R(hits=["x", "x"])
    out2 = await search.invoke(Q(text="y", limit=1), tenant_id="t")
    assert out2.hits == ["y"]


async def test_invalid_input_raises(reg):
    @tool(registry=reg)
    async def f(n: int) -> int:
        return n

    with pytest.raises(ToolInputError):
        await f.invoke({"n": "not-int"}, tenant_id="t")
    with pytest.raises(ToolInputError):
        await f.invoke({"n": 1, "extra": 2}, tenant_id="t")


async def test_output_validated_against_model(reg):
    @tool(registry=reg)
    async def bad() -> R:
        return {"hits": "not-a-list"}

    with pytest.raises(ToolOutputError):
        await bad.invoke({}, tenant_id="t")


async def test_timeout_enforced(reg):
    @tool(registry=reg, timeout_s=0.05)
    async def slow() -> int:
        await asyncio.sleep(2)
        return 1

    with pytest.raises(ToolTimeoutError):
        await slow.invoke({}, tenant_id="t")


async def test_tenant_allow_list(reg):
    @tool(registry=reg, tenants=["acme"])
    async def secret() -> str:
        return "s"

    assert secret.allowed_for("acme") and not secret.allowed_for("evil")
    assert await secret.invoke({}, tenant_id="acme") == "s"
    with pytest.raises(ToolNotAllowedError):
        await secret.invoke({}, tenant_id="evil")
    with pytest.raises(ToolNotAllowedError):
        reg.get("secret", tenant_id="evil")
    reg.allow("secret", "evil")
    assert await secret.invoke({}, tenant_id="evil") == "s"
    assert [t.name for t in reg.for_tenant("nobody")] == []


def test_registry_lookup_and_duplicates(reg):
    t = Tool(name="x", fn=lambda: 1)
    reg.register(t)
    with pytest.raises(ValueError):
        reg.register(t)
    reg.register(t, replace=True)
    with pytest.raises(ToolNotFoundError):
        reg.get("missing")
    assert reg.names() == ["x"] and len(reg) == 1
    assert reg.schemas()[0]["name"] == "x"


def test_tool_validation_errors():
    with pytest.raises(ValueError):
        Tool(name="", fn=lambda: 1)
    with pytest.raises(ValueError):
        Tool(name="t", fn=lambda: 1, timeout_s=0)
    with pytest.raises(ValueError):
        Tool(name="t", fn="not callable")  # type: ignore[arg-type]


async def test_run_step_records_success(reg):
    @tool(registry=reg)
    async def echo(q: str) -> dict:
        return {"q": q}

    s = make_state(current_node="n")
    out = await echo.run_step(s, q="hi")
    assert out == {"q": "hi"}
    step = s.scratchpad[-1]
    assert step.tool == "echo" and step.node == "n" and step.status == "ok"
    assert step.input == {"q": "hi"} and step.output == {"q": "hi"}
    assert step.duration_ms >= 0


async def test_run_step_records_failure_then_reraises(reg):
    @tool(registry=reg, tenants=["other"])
    async def locked() -> int:
        return 1

    s = make_state(current_node="n")
    with pytest.raises(ToolNotAllowedError):
        await locked.run_step(s)
    assert s.scratchpad[-1].status == "error"
    assert "ToolNotAllowedError" in s.scratchpad[-1].error


async def test_run_step_records_timeout_status(reg):
    @tool(registry=reg, timeout_s=0.02)
    async def slow() -> int:
        await asyncio.sleep(1)
        return 1

    s = make_state(current_node="n")
    with pytest.raises(ToolTimeoutError):
        await slow.run_step(s)
    assert s.scratchpad[-1].status == "timeout"


async def test_nested_model_parameters_arrive_as_models_not_dicts(reg):
    """A tool with several parameters, one of them a BaseModel, must receive the
    model instance (as its signature says), not a dumped dict."""

    @tool(registry=reg)
    async def search(q: Q, limit: int = 1) -> list[str]:
        assert isinstance(q, Q), type(q)
        return [q.text] * limit

    assert await search.invoke({"q": {"text": "x"}, "limit": 2}, tenant_id="t") == ["x", "x"]
    assert await search.invoke({"q": Q(text="y")}, tenant_id="t") == ["y"]


async def test_sync_function_tools_work(reg):
    @tool(registry=reg)
    def upper(text: str) -> str:
        return text.upper()

    assert await upper.invoke({"text": "a"}, tenant_id="t") == "A"
