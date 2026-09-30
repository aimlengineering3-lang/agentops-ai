import pytest

from agentops.contracts import BudgetLimits, EventType, RunState
from agentops.errors import BudgetExceededError
from agentops.llm import Usage
from agentops.orchestrator import BudgetGuard, EventBus


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _guard(**limits):
    state = RunState(
        objective="Compare three approaches for an enterprise AI platform",
        limits=BudgetLimits(**limits),
    )
    clock = FakeClock()
    return state, clock, BudgetGuard(state, EventBus(), clock)


def test_llm_calls_are_counted_before_the_call():
    state, _, guard = _guard(max_llm_calls=2)
    guard.before_llm_call()
    guard.before_llm_call()
    assert state.usage.llm_calls == 2
    with pytest.raises(BudgetExceededError, match="llm_calls"):
        guard.before_llm_call()
    assert state.usage.llm_calls == 2  # the refused call is not counted


def test_tool_budget_and_budget_event():
    state, _, guard = _guard(max_tool_calls=1)
    guard.before_tool_call()
    with pytest.raises(BudgetExceededError):
        guard.before_tool_call()
    assert state.events[-1].event_type == EventType.BUDGET
    assert state.events[-1].success is False


def test_wall_clock_limit_uses_injected_clock():
    state, clock, guard = _guard(max_wall_clock_s=60)
    guard.check()
    clock.now += 61
    with pytest.raises(BudgetExceededError, match="wall_clock"):
        guard.check()
    assert state.usage.elapsed_s == 61


def test_tokens_are_accumulated():
    state, _, guard = _guard()
    guard.record_tokens(Usage(tokens_in=10, tokens_out=5))
    guard.record_tokens(Usage(tokens_in=1, tokens_out=2))
    assert (state.usage.tokens_in, state.usage.tokens_out) == (11, 7)


def test_critic_cycles_do_not_hard_stop_the_run():
    state, _, guard = _guard(max_critic_cycles=1)
    state.usage.critic_cycles = 1
    guard.check()  # must not raise: finalizer still has to run
    assert state.usage.can_run_critic_cycle(state.limits) is False
