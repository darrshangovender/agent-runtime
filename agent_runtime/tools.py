"""Tool registry with typed schemas, timeouts and per-tenant allow-lists.

A :class:`Tool` wraps a sync or async function with a Pydantic input model
(derived from the signature, or the single ``BaseModel`` parameter), an optional
output model, a timeout, and an optional set of tenants allowed to call it.
:meth:`Tool.run_step` executes the tool against an :class:`AgentState`, records
a :class:`Step` in the scratchpad (success or failure) and returns the output.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from .state import AgentState, Step
from .telemetry import get_tracer


class ToolError(RuntimeError):
    """Base class for tool failures."""


class ToolNotFoundError(ToolError, KeyError):
    pass


class ToolNotAllowedError(ToolError, PermissionError):
    pass


class ToolTimeoutError(ToolError, TimeoutError):
    pass


class ToolInputError(ToolError, ValueError):
    pass


class ToolOutputError(ToolError, ValueError):
    pass


def _is_model(tp: Any) -> bool:
    return inspect.isclass(tp) and issubclass(tp, BaseModel)


def _build_input_model(fn: Callable[..., Any], name: str) -> tuple[type[BaseModel], bool]:
    """Return ``(model, single)``: ``single`` is True when fn takes one BaseModel arg."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001 - unresolvable forward refs: fall back to raw annotations
        hints = {}
    params = [p for p in sig.parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    if len(params) == 1 and _is_model(hints.get(params[0].name, params[0].annotation)):
        return hints.get(params[0].name, params[0].annotation), True
    fields: dict[str, Any] = {}
    for p in params:
        ann = hints.get(p.name, p.annotation)
        if ann is inspect.Parameter.empty:
            ann = Any
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[p.name] = (ann, default)
    model = create_model(  # type: ignore[call-overload]
        f"{_camel(name)}Input",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    return model, False


def _camel(s: str) -> str:
    return "".join(part.capitalize() for part in s.replace("-", "_").split("_"))


@dataclass(slots=True)
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str = ""
    input_model: type[BaseModel] | None = None
    output_model: type[BaseModel] | None = None
    timeout_s: float | None = 30.0
    tenants: set[str] | None = None  # None = every tenant may call
    _single_model: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must be non-empty")
        if not callable(self.fn):
            # ValueError is the documented contract for every Tool construction error.
            raise ValueError(f"tool {self.name!r}: fn is not callable")  # noqa: TRY004
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(f"tool {self.name!r}: timeout_s must be positive")
        if self.input_model is None:
            self.input_model, self._single_model = _build_input_model(self.fn, self.name)
        else:
            self._single_model = True
        if self.output_model is None:
            try:
                ret = get_type_hints(self.fn).get("return")
            except Exception:  # noqa: BLE001 - unresolvable forward refs: no output model
                ret = None
            if _is_model(ret):
                self.output_model = ret
        if not self.description:
            self.description = (inspect.getdoc(self.fn) or "").strip().split("\n")[0]

    # -- access control --------------------------------------------------

    def allowed_for(self, tenant_id: str) -> bool:
        return self.tenants is None or tenant_id in self.tenants

    def allow(self, *tenant_ids: str) -> Tool:
        """Restrict to (or extend) an allow-list. Once set, only listed tenants may call."""
        if self.tenants is None:
            self.tenants = set()
        self.tenants.update(tenant_ids)
        return self

    # -- schema helpers ----------------------------------------------------

    def input_schema(self) -> dict[str, Any]:
        assert self.input_model is not None
        return self.input_model.model_json_schema()

    def output_schema(self) -> dict[str, Any] | None:
        return self.output_model.model_json_schema() if self.output_model else None

    def parse_input(self, data: Any) -> BaseModel:
        assert self.input_model is not None
        try:
            if isinstance(data, self.input_model):
                return data
            return self.input_model.model_validate(data)
        except ValidationError as exc:
            raise ToolInputError(f"tool {self.name!r}: invalid input: {exc}") from exc

    # -- execution ---------------------------------------------------------

    async def invoke(self, data: Any, *, tenant_id: str) -> Any:
        """Validate input, enforce tenant allow-list and timeout, validate output."""
        if not self.allowed_for(tenant_id):
            raise ToolNotAllowedError(f"tool {self.name!r} is not allowed for tenant {tenant_id!r}")
        parsed = self.parse_input(data)
        if self._single_model:
            call = self.fn(parsed)
        else:
            # Pass field values as-is: ``model_dump()`` would turn a nested
            # BaseModel argument into a dict, which is not what fn's signature says.
            call = self.fn(**{name: getattr(parsed, name) for name in type(parsed).model_fields})
        try:
            if inspect.isawaitable(call):
                result = (
                    await asyncio.wait_for(call, self.timeout_s)
                    if self.timeout_s is not None
                    else await call
                )
            else:
                result = call
        except TimeoutError as exc:
            raise ToolTimeoutError(f"tool {self.name!r} exceeded {self.timeout_s}s") from exc
        if self.output_model is not None and not isinstance(result, self.output_model):
            try:
                result = self.output_model.model_validate(result)
            except ValidationError as exc:
                raise ToolOutputError(f"tool {self.name!r}: invalid output: {exc}") from exc
        return result

    async def run_step(self, state: AgentState, /, **kwargs: Any) -> Any:
        """Invoke against ``state`` and record a :class:`Step` (success or failure)."""
        node = state.current_node or "?"
        data: Any = kwargs.pop("_input", None) if "_input" in kwargs else kwargs
        payload = _jsonable(data)
        t0 = time.perf_counter()
        tracer = get_tracer()
        cm = tracer.span(f"tool:{self.name}", "tool", tool=self.name) if tracer else _nullcm()
        with cm:
            try:
                result = await self.invoke(data, tenant_id=state.tenant_id)
            except Exception as exc:
                status = "timeout" if isinstance(exc, ToolTimeoutError) else "error"
                state.record(
                    Step(
                        node=node,
                        kind="tool",
                        tool=self.name,
                        input=payload,
                        output=None,
                        status=status,
                        duration_ms=_ms(t0),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                raise
        state.record(
            Step(
                node=node,
                kind="tool",
                tool=self.name,
                input=payload,
                output=_jsonable(result),
                status="ok",
                duration_ms=_ms(t0),
            )
        )
        return result


class _nullcm:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class ToolRegistry:
    """Named collection of tools with tenant-aware lookup."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        if tool.name in self._tools and not replace:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str, *, tenant_id: str | None = None) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        if tenant_id is not None and not tool.allowed_for(tenant_id):
            raise ToolNotAllowedError(f"tool {name!r} is not allowed for tenant {tenant_id!r}")
        return tool

    def allow(self, name: str, *tenant_ids: str) -> Tool:
        return self.get(name).allow(*tenant_ids)

    def for_tenant(self, tenant_id: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.allowed_for(tenant_id)]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        tools: Iterable[Tool] = (
            self._tools.values() if tenant_id is None else self.for_tenant(tenant_id)
        )
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema(),
                "output_schema": t.output_schema(),
            }
            for t in tools
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def clear(self) -> None:
        self._tools.clear()


default_registry = ToolRegistry()


def tool(
    name: str | None = None,
    *,
    description: str = "",
    timeout_s: float | None = 30.0,
    tenants: Iterable[str] | None = None,
    output_model: type[BaseModel] | None = None,
    registry: ToolRegistry | None = default_registry,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorate a function into a :class:`Tool` and (by default) register it."""

    def decorate(fn: Callable[..., Any]) -> Tool:
        t = Tool(
            name=name or fn.__name__,
            fn=fn,
            description=description,
            output_model=output_model,
            timeout_s=timeout_s,
            tenants=set(tenants) if tenants is not None else None,
        )
        if registry is not None:
            registry.register(t)
        return t

    return decorate


__all__ = [
    "Tool",
    "ToolError",
    "ToolInputError",
    "ToolNotAllowedError",
    "ToolNotFoundError",
    "ToolOutputError",
    "ToolRegistry",
    "ToolTimeoutError",
    "default_registry",
    "tool",
]
