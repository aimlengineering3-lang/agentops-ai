from agentops.agents.planner import LLMPlanner
from agentops.contracts import EventType, RunState
from agentops.orchestrator import BudgetGuard, EventBus, RunContext
from agentops.testing.fakes import FakeLLM

OBJECTIVE = "Compare three approaches for building an enterprise AI platform"

VALID_PLAN_JSON = """
{
  "subtasks": [
    {
      "id": "s1",
      "description": "Research the three approaches",
      "acceptance_criteria": ["a"],
      "depends_on": []
    },
    {
      "id": "s2",
      "description": "Compare the trade-offs",
      "acceptance_criteria": ["b"],
      "depends_on": ["s1"]
    }
  ]
}
"""


def _context(state: RunState) -> RunContext:
    bus = EventBus()
    return RunContext(bus=bus, guard=BudgetGuard(state, bus))


def test_planner_returns_parsed_plan():
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    planner = LLMPlanner(FakeLLM([VALID_PLAN_JSON]))

    plan = planner.plan(state, ctx)

    assert [s.id for s in plan.subtasks] == ["s1", "s2"]
    assert plan.subtasks[1].depends_on == ["s1"]
    assert state.usage.llm_calls == 1
    assert any(e.event_type == EventType.LLM_CALL for e in state.events)


def test_planner_repairs_once_on_bad_json():
    llm = FakeLLM(["not json", VALID_PLAN_JSON])
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    planner = LLMPlanner(llm)

    plan = planner.plan(state, ctx)

    assert len(plan.subtasks) == 2
    assert len(llm.calls) == 2


def test_planner_prompt_includes_the_objective():
    llm = FakeLLM([VALID_PLAN_JSON])
    state = RunState(objective=OBJECTIVE)
    ctx = _context(state)
    planner = LLMPlanner(llm)

    planner.plan(state, ctx)

    user_message = llm.calls[0][-1]
    assert OBJECTIVE in user_message.content
