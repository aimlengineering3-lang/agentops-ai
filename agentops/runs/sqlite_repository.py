import logging
import sqlite3
import threading

from agentops.contracts import AgentName, EventType, RunState, RunStatus
from agentops.contracts.ids import utc_now

logger = logging.getLogger(__name__)

_ACTIVE = (RunStatus.PENDING.value, RunStatus.RUNNING.value)


class SQLiteRunRepository:
    """RunRepository backed by one SQLite file (stdlib only, zero cost, no server).

    Each run is one row holding the full RunState as JSON: RunState is already the single
    source of truth, so there is nothing to normalise. save() is called on every trace
    event, which is fine at this scale (tens of events per run).

    Limits, stated honestly: SQLite is a single-writer file. On hosts with an ephemeral
    disk (most free tiers) it survives a process restart but not a redeploy; for that, swap
    in a Postgres repository behind the same three-method interface.
    """

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        # check_same_thread=False: runs save from background worker threads; the lock
        # serialises every use of the connection.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS runs ("
                "run_id TEXT PRIMARY KEY, status TEXT NOT NULL, state_json TEXT NOT NULL)"
            )
            self._conn.commit()

    def save(self, state: RunState) -> None:
        payload = state.model_dump_json()
        with self._lock:
            # Upsert keeps the row's rowid, so list() order stays "order first saved".
            self._conn.execute(
                "INSERT INTO runs (run_id, status, state_json) VALUES (?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "status = excluded.status, state_json = excluded.state_json",
                (state.run_id, state.status.value, payload),
            )
            self._conn.commit()

    def get(self, run_id: str) -> RunState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return RunState.model_validate_json(row[0]) if row else None

    def list(self, limit: int = 25) -> list[RunState]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT state_json FROM runs ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [RunState.model_validate_json(r[0]) for r in rows]

    def interrupt_orphans(self) -> int:
        """Mark runs still PENDING/RUNNING as INTERRUPTED. Call once at startup.

        A run that was in flight when the process died has no worker any more; leaving it
        RUNNING would make the UI poll it forever. Returns how many runs were closed.
        """
        with self._lock:
            rows = self._conn.execute(
                f"SELECT state_json FROM runs WHERE status IN ({','.join('?' * len(_ACTIVE))})",
                _ACTIVE,
            ).fetchall()
        for (payload,) in rows:
            state = RunState.model_validate_json(payload)
            reason = "Run interrupted: the server restarted before it finished"
            state.status = RunStatus.INTERRUPTED
            state.finished_at = utc_now()
            state.limitations.append(reason)
            state.add_event(AgentName.ORCHESTRATOR, EventType.ERROR, reason, success=False)
            self.save(state)
        if rows:
            logger.warning("marked %d orphaned run(s) as interrupted", len(rows))
        return len(rows)
