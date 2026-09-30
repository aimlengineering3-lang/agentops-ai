import pytest
from pydantic import BaseModel

from agentops.agents.llm_call import call_llm_structured
from agentops.contracts import AgentName, BudgetLimits, EventType, RunState
from agentops.errors import BudgetExceededError, MalformedOutputError
from agentops.llm import Message
from agentops.orchestrator import BudgetGuard, EventBus, RunContext
from agentops.testing.fakes import FakeLLM


class Toy(BaseModel):
    name: str


MESSAGES = [Message(role="user", content="give me a toy")]
OBJECTIVE = "Compare three approaches for an enterprise AI platform"


def _context(state: RunState) -> RunContext:
    bus = EventBus()
    return RunContext(bus=bus, guard=BudgetGuard(state, bus))


def test_records_budget_and_trace_on_success():
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    llm = FakeLLM(['{"name": "robot"}'])

    toy = call_llm_structured(state, ctx, AgentName.PLANNER, llm, MESSAGES, Toy)

    assert toy == Toy(name="robot")
    assert state.usage.llm_calls == 1
    assert state.usage.tokens_in > 0
    event = state.events[-1]
    assert event.event_type == EventType.LLM_CALL
    assert event.success is True
    assert event.agent == AgentName.PLANNER


def test_emits_failure_event_and_reraises_when_repair_fails():
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    llm = FakeLLM(["not json", "still not json"])

    with pytest.raises(MalformedOutputError):
        call_llm_structured(state, ctx, AgentName.PLANNER, llm, MESSAGES, Toy)

    assert state.usage.llm_calls == 1  # counted once, before the call, not per repair
    event = state.events[-1]
    assert event.event_type == EventType.LLM_CALL
    assert event.success is False


def test_budget_is_checked_before_the_call():
    state = RunState(objective=OBJECTIVE, limits=BudgetLimits(max_llm_calls=1))
    state.usage.llm_calls = 1  # simulate the budget already being used up
    ctx = _context(state)
    llm = FakeLLM(['{"name": "robot"}'])

    with pytest.raises(BudgetExceededError):
        call_llm_structured(state, ctx, AgentName.PLANNER, llm, MESSAGES, Toy)

    assert len(llm.calls) == 0  # the guard stopped it before the LLM was ever touched
