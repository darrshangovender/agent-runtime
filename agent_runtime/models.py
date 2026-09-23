"""Model clients and the fallback chain.

:class:`ModelClient` is the protocol every client implements. :class:`FallbackChain`
tries tiers in order; an exception or timeout at tier N moves to tier N+1. The
last tier is expected to be a :class:`DeterministicFallback`, which is rule-based
and never raises, so a chain that ends with it always returns a response.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .telemetry import get_tracer


class ModelError(RuntimeError):
    """A client failed to produce a response."""


class ModelTimeoutError(ModelError, TimeoutError):
    """A client did not answer within its budget."""


class AllTiersFailedError(ModelError):
    """Every tier in a :class:`FallbackChain` raised (only possible without a deterministic tail)."""

    def __init__(self, attempts: list[TierAttempt]) -> None:
        self.attempts = attempts
        detail = "; ".join(f"tier {a.tier} ({a.model}): {a.error}" for a in attempts)
        super().__init__(f"all {len(attempts)} tiers failed: {detail}")


class TierAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: int
    model: str
    ok: bool
    error: str | None = None
    duration_ms: float = 0.0


class ModelResponse(BaseModel):
    """What every client returns. ``tier`` is filled in by :class:`FallbackChain`."""

    model_config = ConfigDict(extra="forbid")

    text: str
    model: str
    tier: int | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: list[TierAttempt] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ModelClient(Protocol):
    """Anything with a ``name`` and an async ``complete``."""

    name: str

    async def complete(self, prompt: str, *, system: str | None = None) -> ModelResponse: ...


ScriptItem = str | BaseException | Callable[[str], str]


class MockModel:
    """Scripted client for tests and examples.

    ``script`` items are consumed in order: a ``str`` is returned, an exception
    instance (or class) is raised, and any other callable is invoked with the prompt. When the
    script is exhausted the last item is repeated (or ``ModelError`` raised if
    ``repeat_last=False``). ``latency_s`` sleeps before answering so timeouts can
    be exercised.
    """

    def __init__(
        self,
        script: Sequence[ScriptItem] | None = None,
        *,
        name: str = "mock",
        latency_s: float = 0.0,
        cost_per_call_usd: float = 0.0,
        repeat_last: bool = True,
    ) -> None:
        self.name = name
        self.script: list[ScriptItem] = list(script or [])
        self.latency_s = latency_s
        self.cost_per_call_usd = cost_per_call_usd
        self.repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []
        self._cursor = 0

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def complete(self, prompt: str, *, system: str | None = None) -> ModelResponse:
        self.calls.append({"prompt": prompt, "system": system})
        if self.latency_s > 0:
            await asyncio.sleep(self.latency_s)
        if not self.script:
            raise ModelError(f"{self.name}: empty script")
        if self._cursor < len(self.script):
            item = self.script[self._cursor]
            self._cursor += 1
        elif self.repeat_last:
            item = self.script[-1]
        else:
            raise ModelError(f"{self.name}: script exhausted after {len(self.script)} calls")
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item(f"{self.name}: scripted failure")
        text = item(prompt) if callable(item) else item
        return ModelResponse(
            text=text,
            model=self.name,
            cost_usd=self.cost_per_call_usd,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
        )


Rule = tuple[str, str]  # (regex pattern, response)


class DeterministicFallback:
    """Rule-based client that never raises.

    Rules are ``(regex, response)`` pairs tried in order against the prompt; the
    first match wins, otherwise ``default`` is returned. Every code path is
    guarded so that a bad pattern or a non-string prompt still yields a response.
    """

    DEFAULT = (
        "Unable to complete this request with a language model right now. "
        "This is a deterministic fallback response; please retry later."
    )

    def __init__(
        self,
        rules: Sequence[Rule] | None = None,
        *,
        default: str = DEFAULT,
        name: str = "deterministic",
    ) -> None:
        self.name = name
        self.default = default
        self._rules: list[tuple[re.Pattern[str] | None, str]] = []
        for pattern, response in rules or []:
            try:
                compiled: re.Pattern[str] | None = re.compile(pattern, re.IGNORECASE | re.DOTALL)
            except re.error:
                compiled = None
            self._rules.append((compiled, response))

    async def complete(self, prompt: str, *, system: str | None = None) -> ModelResponse:
        text = self.default
        try:
            haystack = prompt if isinstance(prompt, str) else str(prompt)
            for compiled, response in self._rules:
                if compiled is not None and compiled.search(haystack):
                    text = response
                    break
        except Exception:  # noqa: BLE001 - contract: this tier never raises
            text = self.default
        try:
            return ModelResponse(text=text, model=self.name, cost_usd=0.0)
        except Exception:  # noqa: BLE001 - contract: this tier never raises
            return ModelResponse(text=str(self.default), model=self.name, cost_usd=0.0)


class FallbackChain:
    """Try tiers in order; exceptions and timeouts fall through to the next tier.

    ``timeout_s`` may be a single float applied to every tier or a list with one
    entry per tier (``None`` disables the timeout for that tier). The response's
    ``tier`` is the zero-based index of the tier that answered and ``attempts``
    lists every tier tried.
    """

    def __init__(
        self,
        tiers: Sequence[ModelClient],
        *,
        timeout_s: float | Sequence[float | None] | None = 30.0,
        name: str | None = None,
    ) -> None:
        if not tiers:
            raise ValueError("FallbackChain needs at least one tier")
        self.tiers: list[ModelClient] = list(tiers)
        if isinstance(timeout_s, Sequence) and not isinstance(timeout_s, (str, bytes)):
            if len(timeout_s) != len(self.tiers):
                raise ValueError("timeout_s list must have one entry per tier")
            self.timeouts: list[float | None] = list(timeout_s)
        else:
            self.timeouts = [timeout_s] * len(self.tiers)
        self.name = name or "chain(" + ">".join(t.name for t in self.tiers) + ")"

    @property
    def has_deterministic_tail(self) -> bool:
        return isinstance(self.tiers[-1], DeterministicFallback)

    async def complete(self, prompt: str, *, system: str | None = None) -> ModelResponse:
        attempts: list[TierAttempt] = []
        last_exc: Exception | None = None
        tracer = get_tracer()
        for idx, (client, timeout) in enumerate(zip(self.tiers, self.timeouts, strict=True)):
            t0 = time.perf_counter()
            try:
                if tracer is not None:
                    with tracer.span(f"model:{client.name}", "model", tier=idx):
                        resp = await self._call(client, prompt, system, timeout)
                else:
                    resp = await self._call(client, prompt, system, timeout)
            except Exception as exc:  # noqa: BLE001 - any tier failure falls through
                last_exc = exc
                ms = round((time.perf_counter() - t0) * 1000, 3)
                attempts.append(
                    TierAttempt(
                        tier=idx,
                        model=client.name,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                        duration_ms=ms,
                    )
                )
                continue
            ms = round((time.perf_counter() - t0) * 1000, 3)
            attempts.append(TierAttempt(tier=idx, model=client.name, ok=True, duration_ms=ms))
            resp.tier = idx
            resp.attempts = attempts
            return resp
        # Chain the last tier's exception so the original traceback is not lost.
        raise AllTiersFailedError(attempts) from last_exc

    @staticmethod
    async def _call(
        client: ModelClient, prompt: str, system: str | None, timeout: float | None
    ) -> ModelResponse:
        coro = client.complete(prompt, system=system)
        if timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout)
        except TimeoutError as exc:
            raise ModelTimeoutError(f"{client.name} exceeded {timeout}s") from exc


__all__ = [
    "AllTiersFailedError",
    "DeterministicFallback",
    "FallbackChain",
    "MockModel",
    "ModelClient",
    "ModelError",
    "ModelResponse",
    "ModelTimeoutError",
    "TierAttempt",
]
