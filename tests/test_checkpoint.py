import sqlite3

import pytest

from agent_runtime import Checkpoint, MemoryCheckpointer, SQLiteCheckpointer, Step
from tests.conftest import make_state


@pytest.fixture(params=["memory", "sqlite"])
def cp(request, tmp_path):
    if request.param == "memory":
        yield MemoryCheckpointer()
    else:
        c = SQLiteCheckpointer(tmp_path / "cp.db")
        yield c
        c.close()


async def test_round_trip(cp):
    s = make_state()
    s.record(Step(node="a", tool="t", input={"x": 1}, output=[1, 2], cost_usd=0.5))
    s.current_node = "b"
    await cp.save(s)
    back = await cp.load(s.run_id)
    assert back == s


async def test_load_unknown_returns_none(cp):
    assert await cp.load("missing") is None


async def test_history_ordered_by_seq(cp):
    s = make_state()
    await cp.save(s)
    s.record(Step(node="a"))
    await cp.save(s)
    s.record(Step(node="b"))
    await cp.save(s)
    hist = await cp.history(s.run_id)
    assert [h.seq for h in hist] == [0, 1, 2]
    assert (await cp.load(s.run_id)).scratchpad[-1].node == "b"


async def test_save_is_idempotent_on_replay(cp):
    s = make_state()
    s.record(Step(node="a"))
    await cp.save(s)
    await cp.save(s)  # replaying the same step must not duplicate
    assert len(await cp.history(s.run_id)) == 1


async def test_delete_and_list_runs(cp):
    a, b = make_state(), make_state()
    await cp.save(a)
    await cp.save(b)
    assert set(await cp.list_runs()) == {a.run_id, b.run_id}
    assert await cp.delete(a.run_id) == 1
    assert await cp.load(a.run_id) is None
    assert await cp.list_runs() == [b.run_id]


async def test_sqlite_uses_wal_and_schema(tmp_path):
    path = tmp_path / "wal.db"
    c = SQLiteCheckpointer(path)
    await c.save(make_state())
    c.close()
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    cols = [r[1] for r in conn.execute("PRAGMA table_info(checkpoints)")]
    assert cols == ["run_id", "seq", "node", "status", "state_json", "created_at"]
    conn.close()


async def test_sqlite_survives_reopen(tmp_path):
    path = tmp_path / "durable.db"
    s = make_state()
    s.record(Step(node="a"))
    with SQLiteCheckpointer(path) as c:
        await c.save(s)
    with SQLiteCheckpointer(path) as c2:
        assert await c2.load(s.run_id) == s


def test_checkpoint_from_state_sets_seq_and_node():
    s = make_state(current_node="x")
    s.record(Step(node="x"))
    cp = Checkpoint.from_state(s)
    assert cp.seq == 1 and cp.node == "x" and cp.status == "pending"
    assert cp.state == s
