"""Schema enforcement with self-correction routing.

:func:`enforce_schema` asks a model for JSON, validates it against a Pydantic
model, and on failure re-prompts with the validation error appended so the
model can correct itself. After ``max_retries`` further attempts it returns a
:class:`DegradedResult` instead of raising, so the caller can decide what to do
with a partially usable answer.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import ModelClient, ModelResponse

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class SchemaAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int
    raw: str
    error: str | None = None
    model: str | None = None
    tier: int | None = None
    cost_usd: float = 0.0


class DegradedResult(BaseModel):
    """Returned when the model never produced valid output.

    ``partial`` holds whatever JSON object was last parseable (even if it failed
    validation) so downstream code can salvage fields; ``errors`` lists every
    attempt's failure.
    """

    model_config = ConfigDict(extra="forbid")

    schema_name: str
    raw: str
    partial: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    attempts: list[SchemaAttempt] = Field(default_factory=list)
    total_cost_usd: float = 0.0

    @property
    def degraded(self) -> bool:
        return True

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


class JSONExtractionError(ValueError):
    """No JSON object could be found in the model output."""


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of free text.

    Accepts bare JSON, JSON in a ``` fence, or JSON surrounded by prose (finds the
    outermost ``{...}`` or ``[...]`` block).
    """
    if not isinstance(text, str) or not text.strip():
        raise JSONExtractionError("empty response")
    candidates: list[str] = [text.strip()]
    candidates.extend(m.strip() for m in _FENCE_RE.findall(text))
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    last_error: Exception | None = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError as exc:
            last_error = exc
    raise JSONExtractionError(f"no valid JSON in response: {last_error}")


def _schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object and nothing else. It must validate "
        f"against this JSON schema:\n{json.dumps(schema.model_json_schema(), indent=2)}"
    )


def _correction_prompt(base: str, raw: str, error: str) -> str:
    return (
        f"{base}\n\n"
        "Your previous response was invalid.\n"
        f"Previous response:\n{raw}\n\n"
        f"Validation error:\n{error}\n\n"
        "Fix the problem and respond again with only the corrected JSON object."
    )


def _summarise_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts) or str(exc)


async def enforce_schema(
    model: ModelClient,
    prompt: str,
    schema: type[T],
    *,
    max_retries: int = 2,
    system: str | None = None,
    include_schema_in_prompt: bool = True,
) -> T | DegradedResult:
    """Call ``model`` until its output validates as ``schema``.

    The first call uses ``prompt`` (plus the JSON schema unless disabled). Each
    retry appends the previous raw output and the validation error. Returns an
    instance of ``schema`` on success, otherwise :class:`DegradedResult` after
    ``1 + max_retries`` total calls. Exceptions raised by the model itself
    propagate; wrap the client in a :class:`~agent_runtime.models.FallbackChain`
    to absorb those.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    base = f"{prompt}\n\n{_schema_instructions(schema)}" if include_schema_in_prompt else prompt
    current = base
    attempts: list[SchemaAttempt] = []
    errors: list[str] = []
    last_partial: dict[str, Any] | None = None
    last_raw = ""
    total_cost = 0.0

    for attempt in range(1, max_retries + 2):
        resp: ModelResponse = await model.complete(current, system=system)
        raw = resp.text
        last_raw = raw
        total_cost += resp.cost_usd
        try:
            data = extract_json(raw)
            if isinstance(data, dict):
                last_partial = data
            return schema.model_validate(data)
        except ValidationError as exc:
            error = _summarise_validation_error(exc)
        except (JSONExtractionError, ValueError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        attempts.append(
            SchemaAttempt(
                attempt=attempt,
                raw=raw,
                error=error,
                model=resp.model,
                tier=resp.tier,
                cost_usd=resp.cost_usd,
            )
        )
        errors.append(error)
        current = _correction_prompt(base, raw, error)

    return DegradedResult(
        schema_name=schema.__name__,
        raw=last_raw,
        partial=last_partial,
        errors=errors,
        attempts=attempts,
        total_cost_usd=round(total_cost, 6),
    )


__all__ = [
    "DegradedResult",
    "JSONExtractionError",
    "SchemaAttempt",
    "enforce_schema",
    "extract_json",
]
