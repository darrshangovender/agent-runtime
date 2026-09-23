"""Run both examples end to end (offline) and assert on their observable outcomes."""

import importlib
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def examples_path(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    for name in ("research_agent", "resume_demo"):
        sys.modules.pop(name, None)
    yield
    for name in ("research_agent", "resume_demo"):
        sys.modules.pop(name, None)


async def test_research_agent_breaks_the_loop(examples_path, capsys):
    mod = importlib.import_module("research_agent")
    final = await mod.main()
    out = capsys.readouterr().out
    assert final.status == "completed"
    assert final.final.data["resolved_by"] == "loop-break"
    assert final.final.data["repeated_tool"] == "web_search"
    # search ran twice, both with the same input -> detection fired
    searches = [s for s in final.tool_steps() if s.tool == "web_search"]
    assert len(searches) == 2 and searches[0].input == searches[1].input
    # planner self-corrected once, synthesiser answered from tier 1
    assert mod.planner.call_count == 2
    synth = next(s for s in final.scratchpad if s.node == "synthesise" and s.tier is not None)
    assert synth.tier == 1 and synth.model == "secondary-mock"
    assert "loop-break" in out


async def test_resume_demo_completes_after_crash(examples_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_RUNTIME_DB", str(tmp_path / "demo.db"))
    mod = importlib.import_module("resume_demo")
    mod.CRASH_NEXT["enrich"] = True
    resumed = await mod.main()
    out = capsys.readouterr().out
    assert resumed.status == "completed"
    assert resumed.completed_nodes() == ["ingest", "analyse", "enrich", "finish"]
    assert "first status : failed" in out and "resumed      : completed" in out
