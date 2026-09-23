import io
import json
import sys

import pytest

from agent_runtime import OpenTelemetryTracer, Tracer, current_tracer, get_tracer


def test_spans_nest_and_write_json_lines():
    buf = io.StringIO()
    t = Tracer(sink=buf)
    with t.span("run", "run", run_id="r1") as outer:
        with t.span("node:a", "node") as inner:
            inner.set(cost=0.1)
        t.event("tool.repeat", tool="x")
    lines = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert len(lines) == 3
    node, event, run = lines
    assert node["kind"] == "node" and node["parent_id"] == outer.span_id
    assert node["attributes"] == {"cost": 0.1}
    assert event["event"] == "tool.repeat" and event["span_id"] == outer.span_id
    assert run["kind"] == "run" and run["parent_id"] is None
    assert run["duration_ms"] >= node["duration_ms"]


def test_span_records_error_and_reraises():
    t = Tracer()
    with pytest.raises(ValueError), t.span("boom"):
        raise ValueError("bad")
    assert t.spans[0].status == "error" and "ValueError: bad" in t.spans[0].error
    assert t.summary()["errors"] == 1


def test_file_sink_appends(tmp_path):
    path = tmp_path / "trace" / "spans.jsonl"
    t = Tracer(sink=path)
    with t.span("a"):
        pass
    with t.span("b"):
        pass
    assert len(path.read_text().splitlines()) == 2


def test_summary_counts_by_kind():
    t = Tracer()
    with t.span("r", "run"):
        with t.span("n", "node"):
            pass
        with t.span("tool", "tool"):
            pass
    s = t.summary()
    assert s["by_kind"] == {"node": 1, "tool": 1, "run": 1} and s["spans"] == 3


def test_current_tracer_contextvar():
    assert get_tracer() is None
    t = Tracer()
    token = current_tracer.set(t)
    try:
        assert get_tracer() is t
    finally:
        current_tracer.reset(token)
    assert get_tracer() is None


def test_otel_tracer_degrades_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry", None)  # force ImportError
    buf = io.StringIO()
    t = OpenTelemetryTracer(sink=buf)
    assert t.otel_enabled is False
    with t.span("x", "node"):
        pass
    assert len(t.spans) == 1 and buf.getvalue().count("\n") == 1


def test_otel_tracer_disabled_flag():
    t = OpenTelemetryTracer(enabled=False)
    assert t.otel_enabled is False
    with t.span("x"):
        pass
    assert t.spans[0].name == "x"


def test_otel_tracer_uses_package_when_present(monkeypatch):
    calls = []

    class FakeSpan:
        def set_attribute(self, k, v):
            calls.append(("attr", k, v))

        def record_exception(self, e):
            calls.append(("exc", type(e).__name__))

        def end(self):
            calls.append(("end",))

    class FakeTracer:
        def start_span(self, name):
            calls.append(("start", name))
            return FakeSpan()

    class FakeTraceModule:
        @staticmethod
        def get_tracer(name):
            return FakeTracer()

    import types

    pkg = types.ModuleType("opentelemetry")
    pkg.trace = FakeTraceModule()
    monkeypatch.setitem(sys.modules, "opentelemetry", pkg)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", pkg.trace)

    t = OpenTelemetryTracer()
    assert t.otel_enabled
    with pytest.raises(RuntimeError), t.span("node:a", "node", node="a"):
        raise RuntimeError("x")
    assert ("start", "node:a") in calls
    assert ("exc", "RuntimeError") in calls and ("end",) in calls
