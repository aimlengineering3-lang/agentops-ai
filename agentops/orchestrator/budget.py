import time
from collections.abc import Callable

from agentops.contracts import AgentName, EventType, RunState
from agentops.errors import BudgetExceededError
from agentops.llm import Usage

from .event_bus import EventBus


class BudgetGuard:
    """Enforces hard run budgets. Owned by the orchestrator, never by an LLM.

    Calls are counted BEFORE they are made: a failed call still burns provider quota,
    so counting only successes would let a failing provider be hammered forever.
    """

    def __init__(
        self,
        state: RunState,
        bus: EventBus,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._state = state
        self._bus = bus
        self._clock = clock
        self._started = clock()  # monotonic: immune to system clock changes
        self._finalizing = False

    def check(self) -> None:
        """Raise BudgetExceededError if any hard budget is used up."""
        usage = self._state.usage
        usage.elapsed_s = self._clock() - self._started
        hit = usage.exhausted(self._state.limits, finalizing=self._finalizing)
        if hit:
            names = ", ".join(hit)
            self._bus.emit(
                self._state,
                AgentName.ORCHESTRATOR,
                EventType.BUDGET,
                f"Budget exhausted: {names}",
                success=False,
                data={"exhausted": hit},
            )
            raise BudgetExceededError(f"budget exhausted: {names}")

    def enter_finalization(self) -> None:
        """Release the wall-clock reserve: from here only the full limit applies."""
        self._finalizing = True

    def before_llm_call(self) -> None:
        self.check()
        self._state.usage.llm_calls += 1

    def before_tool_call(self) -> None:
        self.check()
        self._state.usage.tool_calls += 1

    def record_tokens(self, usage: Usage, *, attempts: int = 1) -> None:
        """Add a call's token cost. `attempts` > 1 means schema-repair requests were made;
        llm_calls stays a count of logical calls, llm_repairs counts the extra requests."""
        self._state.usage.tokens_in += usage.tokens_in
        self._state.usage.tokens_out += usage.tokens_out
        self._state.usage.llm_repairs += max(attempts - 1, 0)
