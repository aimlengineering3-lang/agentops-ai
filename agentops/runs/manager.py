import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from agentops.contracts import AgentName, BudgetLimits, EventType, RunState, RunStatus
from agentops.contracts.ids import utc_now
from agentops.orchestrator import EventBus, Orchestrator

from .repository import RunRepository

logger = logging.getLogger(__name__)

OrchestratorFactory = Callable[[EventBus], Orchestrator]


class RunManager:
    """Starts runs in the background and makes them findable while they're in flight.

    Framework-agnostic on purpose (no FastAPI import here) -- the API layer is a
    thin shell around this, per the master doc's "core has no FastAPI/Streamlit
    imports". Each run gets its own EventBus and Orchestrator (via
    `orchestrator_factory`), so concurrent runs never cross-subscribe into each
    other's repository saves.
    """

    def __init__(
        self,
        orchestrator_factory: OrchestratorFactory,
        repository: RunRepository,
        *,
        max_workers: int = 4,
    ) -> None:
        self._orchestrator_factory = orchestrator_factory
        self._repository = repository
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}

    def start_run(self, objective: str, limits: BudgetLimits | None = None) -> RunState:
        state = RunState(objective=objective, limits=limits or BudgetLimits())
        self._repository.save(state)

        bus = EventBus()
        bus.subscribe(lambda _event: self._repository.save(state))
        orchestrator = self._orchestrator_factory(bus)

        future = self._executor.submit(self._run_and_save, orchestrator, state)
        with self._lock:
            self._futures[state.run_id] = future

        # A snapshot, not the live object: `state` now belongs to the background
        # thread, and reading it concurrently here would race the orchestrator's
        # in-place mutations (e.g. a list changing size mid-iteration).
        return state.model_copy(deep=True)

    def get_run(self, run_id: str) -> RunState | None:
        return self._repository.get(run_id)

    def list_runs(self, limit: int = 25) -> list[RunState]:
        return self._repository.list(limit)

    def join(self, run_id: str, timeout: float | None = None) -> RunState:
        """Block until the run finishes (or `timeout` elapses); return its final state.

        Mainly for tests and scripts -- the HTTP API polls get_run()/list_runs() instead.
        """
        with self._lock:
            future = self._futures.get(run_id)
        if future is not None:
            future.result(timeout=timeout)
        state = self._repository.get(run_id)
        if state is None:
            raise KeyError(f"no run with id {run_id}")
        return state

    def _run_and_save(self, orchestrator: Orchestrator, state: RunState) -> None:
        try:
            orchestrator.run(state)
        except Exception as exc:
            # The orchestrator already turns its own known failure modes into a
            # graceful COMPLETED_WITH_LIMITATIONS status; reaching here means
            # something genuinely unexpected happened. Still never let it vanish
            # silently in a background thread -- record it and finish the run.
            logger.exception("run %s failed outside the orchestrator's own handling", state.run_id)
            state.status = RunStatus.FAILED
            state.finished_at = utc_now()
            reason = f"Run crashed unexpectedly: {type(exc).__name__}"
            state.limitations.append(reason)
            state.add_event(AgentName.ORCHESTRATOR, EventType.ERROR, reason, success=False)
        finally:
            self._repository.save(state)
