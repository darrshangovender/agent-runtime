"""Checkpoint persistence.

The runtime calls :meth:`Checkpointer.save` after every node. A checkpoint's
sequence number is the length of the scratchpad at save time, so replaying the
same state twice (for example after a resume) writes the same row and is
idempotent rather than duplicating history.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from .state import AgentState


class Checkpoint(BaseModel):
    """One persisted snapshot of a run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    seq: int
    node: str | None
    status: str
    state_json: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def state(self) -> AgentState:
        return AgentState.from_json(self.state_json)

    @classmethod
    def from_state(cls, state: AgentState) -> Checkpoint:
        return cls(
            run_id=state.run_id,
            seq=len(state.scratchpad),
            node=state.current_node,
            status=state.status,
            state_json=state.to_json(),
        )


class Checkpointer(ABC):
    """Persist and reload :class:`AgentState` snapshots keyed by ``run_id``."""

    @abstractmethod
    async def save(self, state: AgentState) -> Checkpoint:
        """Persist the current state. Must be idempotent for the same ``(run_id, seq)``."""

    @abstractmethod
    async def load(self, run_id: str) -> AgentState | None:
        """Return the latest state for a run, or ``None`` if unknown."""

    @abstractmethod
    async def history(self, run_id: str) -> list[Checkpoint]:
        """All checkpoints for a run, ordered by ``seq``."""

    @abstractmethod
    async def delete(self, run_id: str) -> int:
        """Remove every checkpoint for a run. Returns rows removed."""

    @abstractmethod
    async def list_runs(self) -> list[str]:
        """Known run ids."""


class MemoryCheckpointer(Checkpointer):
    """In-process checkpointer for tests and ephemeral runs."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[int, Checkpoint]] = {}
        self._lock = asyncio.Lock()

    async def save(self, state: AgentState) -> Checkpoint:
        cp = Checkpoint.from_state(state)
        async with self._lock:
            self._runs.setdefault(cp.run_id, {})[cp.seq] = cp
        return cp

    async def load(self, run_id: str) -> AgentState | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            return run[max(run)].state

    async def history(self, run_id: str) -> list[Checkpoint]:
        async with self._lock:
            run = self._runs.get(run_id, {})
            return [run[k] for k in sorted(run)]

    async def delete(self, run_id: str) -> int:
        async with self._lock:
            return len(self._runs.pop(run_id, {}))

    async def list_runs(self) -> list[str]:
        async with self._lock:
            return sorted(self._runs)


SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id     TEXT    NOT NULL,
    seq        INTEGER NOT NULL,
    node       TEXT,
    status     TEXT    NOT NULL,
    state_json TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    PRIMARY KEY (run_id, seq)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints (run_id, seq DESC);
"""


class SQLiteCheckpointer(Checkpointer):
    """Durable checkpointer on a single SQLite file in WAL mode.

    One row per checkpoint (``run_id``, ``seq``). Writes use ``INSERT OR REPLACE``
    so replaying an already-persisted step is a no-op rather than a duplicate.
    A single connection is shared and guarded by a lock; blocking calls run in
    a worker thread so the event loop is never stalled.
    """

    def __init__(self, path: str | Path = "checkpoints.db") -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(SCHEMA)

    # -- sync internals (run in a thread) --------------------------------

    def _save_sync(self, cp: Checkpoint) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO checkpoints "
                "(run_id, seq, node, status, state_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (cp.run_id, cp.seq, cp.node, cp.status, cp.state_json, cp.created_at.isoformat()),
            )

    def _load_sync(self, run_id: str) -> AgentState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM checkpoints WHERE run_id = ? ORDER BY seq DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return AgentState.from_json(row[0]) if row else None

    def _history_sync(self, run_id: str) -> list[Checkpoint]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, seq, node, status, state_json, created_at "
                "FROM checkpoints WHERE run_id = ? ORDER BY seq ASC",
                (run_id,),
            ).fetchall()
        return [
            Checkpoint(
                run_id=r[0],
                seq=r[1],
                node=r[2],
                status=r[3],
                state_json=r[4],
                created_at=datetime.fromisoformat(r[5]),
            )
            for r in rows
        ]

    def _delete_sync(self, run_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
            return cur.rowcount

    def _list_sync(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT run_id FROM checkpoints ORDER BY run_id"
            ).fetchall()
        return [r[0] for r in rows]

    # -- async API -------------------------------------------------------

    async def save(self, state: AgentState) -> Checkpoint:
        cp = Checkpoint.from_state(state)
        await asyncio.to_thread(self._save_sync, cp)
        return cp

    async def load(self, run_id: str) -> AgentState | None:
        return await asyncio.to_thread(self._load_sync, run_id)

    async def history(self, run_id: str) -> list[Checkpoint]:
        return await asyncio.to_thread(self._history_sync, run_id)

    async def delete(self, run_id: str) -> int:
        return await asyncio.to_thread(self._delete_sync, run_id)

    async def list_runs(self) -> list[str]:
        return await asyncio.to_thread(self._list_sync)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["SCHEMA", "Checkpoint", "Checkpointer", "MemoryCheckpointer", "SQLiteCheckpointer"]
