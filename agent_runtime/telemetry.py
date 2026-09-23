"""Structured tracing.

:class:`Tracer` records a span per run, node, tool and model call and writes
each finished span as one JSON line to an optional sink. :class:`OpenTelemetryTracer`
mirrors those spans into OpenTelemetry when the ``opentelemetry`` package is
installed and silently behaves like the plain tracer when it is not.

The active tracer is exposed through a :class:`contextvars.ContextVar` so tools
and model clients can attach spans without threading a tracer argument through
every call.
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

SpanKind = str  # "run" | "node" | "tool" | "model" | custom


@dataclass(slots=True)
class Span:
    name: str
    kind: SpanKind
    span_id: str
    parent_id: str | None
    trace_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return round((end - self.started_at) * 1000, 3)

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "started_at": datetime.fromtimestamp(self.started_at, UTC).isoformat(),
            "ended_at": (
                datetime.fromtimestamp(self.ended_at, UTC).isoformat()
                if self.ended_at is not None
                else None
            ),
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return repr(obj)


class Tracer:
    """Records spans in memory and optionally streams them as JSON lines.

    ``sink`` may be a file path (appended to), an open text stream, or ``None``
    to keep spans in memory only.
    """

    def __init__(self, sink: str | Path | TextIO | None = None, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.spans: list[Span] = []
        self._stack: list[Span] = []
        self._path: Path | None = None
        self._stream: TextIO | None = None
        if isinstance(sink, (str, Path)):
            self._path = Path(sink)
            self._path.parent.mkdir(parents=True, exist_ok=True)
        elif sink is not None:
            self._stream = sink

    # -- span lifecycle --------------------------------------------------

    @property
    def current(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    def start_span(self, name: str, kind: SpanKind = "custom", **attributes: Any) -> Span:
        parent = self.current
        span = Span(
            name=name,
            kind=kind,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            trace_id=self.trace_id,
            attributes=dict(attributes),
        )
        self._stack.append(span)
        return span

    def end_span(self, span: Span, error: BaseException | None = None) -> Span:
        span.ended_at = time.time()
        if error is not None:
            span.status = "error"
            span.error = f"{type(error).__name__}: {error}"
        if self._stack and self._stack[-1] is span:
            self._stack.pop()
        elif span in self._stack:
            self._stack.remove(span)
        self.spans.append(span)
        self._emit(span.to_dict())
        return span

    @contextlib.contextmanager
    def span(self, name: str, kind: SpanKind = "custom", **attributes: Any) -> Iterator[Span]:
        span = self.start_span(name, kind, **attributes)
        try:
            yield span
        except BaseException as exc:
            self.end_span(span, error=exc)
            raise
        else:
            self.end_span(span)

    def event(self, name: str, **attributes: Any) -> None:
        """Emit a point-in-time record (no duration)."""
        parent = self.current
        self._emit(
            {
                "event": name,
                "trace_id": self.trace_id,
                "span_id": parent.span_id if parent else None,
                "at": datetime.now(UTC).isoformat(),
                "attributes": attributes,
            }
        )

    # -- output ----------------------------------------------------------

    def _emit(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=_json_default, ensure_ascii=False)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        elif self._stream is not None:
            self._stream.write(line + "\n")
            self._stream.flush()

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        total_ms = 0.0
        errors = 0
        for s in self.spans:
            by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
            if s.kind == "node":
                total_ms += s.duration_ms
            if s.status == "error":
                errors += 1
        return {
            "trace_id": self.trace_id,
            "spans": len(self.spans),
            "by_kind": by_kind,
            "node_time_ms": round(total_ms, 3),
            "errors": errors,
        }


class OpenTelemetryTracer(Tracer):
    """A :class:`Tracer` that also exports to OpenTelemetry if it is importable.

    The import happens lazily in ``__init__``. When ``opentelemetry`` is missing
    (or ``enabled=False``) the tracer degrades to the plain JSON-lines tracer and
    :attr:`otel_enabled` is ``False``; nothing raises.
    """

    def __init__(
        self,
        sink: str | Path | TextIO | None = None,
        trace_id: str | None = None,
        *,
        service_name: str = "agent-runtime",
        enabled: bool = True,
    ) -> None:
        super().__init__(sink=sink, trace_id=trace_id)
        self._otel: Any = None
        self._otel_spans: dict[str, Any] = {}
        self.otel_enabled = False
        if not enabled:
            return
        try:
            from opentelemetry import trace as otel_trace  # lazy by design
        except Exception:  # noqa: BLE001 - ImportError or broken install: degrade to no-op
            return
        self._otel = otel_trace.get_tracer(service_name)
        self.otel_enabled = True

    def start_span(self, name: str, kind: SpanKind = "custom", **attributes: Any) -> Span:
        span = super().start_span(name, kind, **attributes)
        if self.otel_enabled and self._otel is not None:
            try:
                ospan = self._otel.start_span(name)
                ospan.set_attribute("agent_runtime.kind", kind)
                for k, v in attributes.items():
                    ospan.set_attribute(f"agent_runtime.{k}", _otel_value(v))
                self._otel_spans[span.span_id] = ospan
            except Exception:  # noqa: BLE001, S110 - exporter faults must not break the run
                pass
        return span

    def end_span(self, span: Span, error: BaseException | None = None) -> Span:
        result = super().end_span(span, error=error)
        ospan = self._otel_spans.pop(span.span_id, None)
        if ospan is not None:
            try:
                for k, v in span.attributes.items():
                    ospan.set_attribute(f"agent_runtime.{k}", _otel_value(v))
                if error is not None:
                    ospan.record_exception(error)
                ospan.end()
            except Exception:  # noqa: BLE001, S110 - exporter faults must not break the run
                pass
        return result


def _otel_value(v: Any) -> Any:
    if isinstance(v, (str, bool, int, float)):
        return v
    return json.dumps(v, default=_json_default)


# The tracer in effect for the current task, set by the runtime for the
# duration of a run so tools and models can attach spans.
current_tracer: ContextVar[Tracer | None] = ContextVar("agent_runtime_tracer", default=None)


def get_tracer() -> Tracer | None:
    return current_tracer.get()


__all__ = ["OpenTelemetryTracer", "Span", "SpanKind", "Tracer", "current_tracer", "get_tracer"]
