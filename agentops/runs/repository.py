import threading
from typing import Protocol

from agentops.contracts import RunState


class RunRepository(Protocol):
    """Persists RunState so a run outlives the request that started it, and (on a
    real backend) survives a host restart. RunManager code never depends on which
    backend it's talking to -- InMemoryRunRepository here, Postgres later."""

    def save(self, state: RunState) -> None: ...
    def get(self, run_id: str) -> RunState | None: ...
    def list(self, limit: int = 25) -> list[RunState]: ...


class InMemoryRunRepository:
    """Thread-safe in-memory RunRepository. Used when DATABASE_URL is unset, or as
    the Postgres repository's fallback if the DB is unreachable (master doc, S11).

    save()/get()/list() all deal in deep copies, never the caller's live object --
    a run mutated by a background thread must never be read mid-mutation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, RunState] = {}
        self._order: list[str] = []  # insertion order, oldest first

    def save(self, state: RunState) -> None:
        with self._lock:
            if state.run_id not in self._runs:
                self._order.append(state.run_id)
            self._runs[state.run_id] = state.model_copy(deep=True)

    def get(self, run_id: str) -> RunState | None:
        with self._lock:
            state = self._runs.get(run_id)
            return state.model_copy(deep=True) if state else None

    def list(self, limit: int = 25) -> list[RunState]:
        with self._lock:
            ids = self._order[-limit:][::-1]  # newest first
            return [self._runs[i].model_copy(deep=True) for i in ids]
