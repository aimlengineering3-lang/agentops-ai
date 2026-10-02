import pytest

from agentops.contracts import (
    BudgetLimits,
    Critique,
    CritiqueAction,
    CritiqueIssue,
    CritiqueVerdict,
    EventType,
    IssueType,
    Plan,
    RunState,
    RunStatus,
    Severity,
    Subtask,
    SubtaskStatus,
)
from agentops.errors import MalformedOutputError, RateLimitError
from agentops.orchestrator import Orchestrator


def _plan() -> Plan:
    return Plan(
        subtasks=[
            Subtask(id="s1", description="Research the options", acceptance_criteria=["a"]),
            Subtask(
                id="s2",
                description="Compare the trade-offs",
                acceptance_criteria=["b"],
                depends_on=["s1"],
            ),
        ]
    )


class FakePlanner:
    def plan(self, state, ctx):
        return _plan()


class RecordingAgent:
    """Used as both Researcher and Analyst: records which subtasks it was called for."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def research(self, state, ctx, subtask):
        self.calls.append(subtask.id)

    analyze = research


class ScriptedCritic:
    def __init__(self, *verdicts: Critique) -> None:
        self._script = list(verdicts)
        self.iterations: list[int] = []

    def critique(self, state, ctx, *, iteration):
        self.iterations.append(iteration)
        if not self._script:
            raise AssertionError("critic called more often than the test expected")
        return self._script.pop(0)


class CountingFinalizer:
    def __init__(self) -> None:
        self.calls = 0

    def finalize(self, state, ctx):
        self.calls += 1


def _pass() -> Critique:
    return Critique(iteration=1, verdict=CritiqueVerdict.PASS)


def _fail(*issues: tuple[str, CritiqueAction]) -> Critique:
    return Critique(
        iteration=1,
        verdict=CritiqueVerdict.FAIL,
        issues=[
            CritiqueIssue(
                type=IssueType.INSUFFICIENT_EVIDENCE,
                subtask_id=sid,
                severity=Severity.MEDIUM,
                action=action,
                detail=f"problem in {sid}",
            )
            for sid, action in issues
        ],
    )


def _run(critic: ScriptedCritic, **limits):
    researcher, analyst, finalizer = RecordingAgent(), RecordingAgent(), CountingFinalizer()
    state = RunState(
        objective="Compare three approaches for an enterprise AI platform",
        limits=BudgetLimits(**limits),
    )
    orchestrator = Orchestrator(
        planner=FakePlanner(),
        researcher=researcher,
        analyst=analyst,
        critic=critic,
        finalizer=finalizer,
    )
    orchestrator.run(state)
    return state, researcher, analyst, finalizer


def test_happy_path_completes_cleanly():
    critic = ScriptedCritic(_pass())
    state, researcher, analyst, finalizer = _run(critic)
    assert researcher.calls == ["s1", "s2"]  # dependency order
    assert analyst.calls == ["s1", "s2"]
    assert finalizer.calls == 1
    assert state.status == RunStatus.COMPLETED
    assert state.finished_at is not None
    types = [e.event_type for e in state.events]
    assert types[0] == EventType.RUN_STARTED
    assert types[-1] == EventType.RUN_FINISHED
    assert EventType.PLAN in types and EventType.CRITIQUE in types


def test_research_more_triggers_targeted_retry_then_passes():
    critic = ScriptedCritic(_fail(("s2", CritiqueAction.RESEARCH_MORE)), _pass())
    state, researcher, analyst, _ = _run(critic)
    assert researcher.calls == ["s1", "s2", "s2"]  # only s2 was redone
    assert analyst.calls == ["s1", "s2", "s2"]
    assert critic.iterations == [1, 2]
    assert state.usage.retries_by_subtask == {"s2": 1}
    assert state.status == RunStatus.COMPLETED
    assert EventType.RETRY in [e.event_type for e in state.events]


def test_reanalyze_does_not_repeat_research():
    critic = ScriptedCritic(_fail(("s1", CritiqueAction.REANALYZE)), _pass())
    _, researcher, analyst, _ = _run(critic)
    assert researcher.calls == ["s1", "s2"]
    assert analyst.calls == ["s1", "s2", "s1"]


def test_multiple_issues_on_one_subtask_cause_one_retry():
    issues = (("s2", CritiqueAction.REANALYZE), ("s2", CritiqueAction.RESEARCH_MORE))
    critic = ScriptedCritic(_fail(*issues), _pass())
    state, researcher, _, _ = _run(critic)
    assert researcher.calls == ["s1", "s2", "s2"]  # research wins over reanalyze
    assert state.usage.retries_by_subtask == {"s2": 1}


def test_critic_cycle_limit_ends_gracefully_with_limitations():
    fail = ("s2", CritiqueAction.RESEARCH_MORE)
    critic = ScriptedCritic(_fail(fail), _fail(fail))  # a third call would raise
    state, _, _, finalizer = _run(critic, max_critic_cycles=2)
    assert critic.iterations == [1, 2]
    assert state.usage.critic_cycles == 2
    assert finalizer.calls == 1  # finalizer still runs
    assert state.status == RunStatus.COMPLETED_WITH_LIMITATIONS
    assert state.limitations


def test_retry_budget_per_subtask_is_respected():
    critic = ScriptedCritic(_fail(("s2", CritiqueAction.RESEARCH_MORE)))
    state, researcher, _, _ = _run(critic, max_retries_per_subtask=0)
    assert researcher.calls == ["s1", "s2"]  # no retry happened
    assert critic.iterations == [1]  # and no pointless second critique
    assert state.status == RunStatus.COMPLETED_WITH_LIMITATIONS


def test_plan_over_subtask_limit_is_rejected():
    with pytest.raises(MalformedOutputError):
        _run(ScriptedCritic(_pass()), max_subtasks=1)


class _JumpingClock:
    """Clock stub: 0.0 on the first call (run start), a fixed later time after that.

    Lets a test force BudgetGuard.check() to see the wall-clock budget as exhausted,
    without waiting in real time or wiring up fake tool/LLM call counters.
    """

    def __init__(self, jump_to: float) -> None:
        self._jump_to = jump_to
        self._calls = 0

    def __call__(self) -> float:
        self._calls += 1
        return 0.0 if self._calls == 1 else self._jump_to


def test_budget_exhaustion_mid_run_ends_gracefully_not_crash():
    researcher, analyst, finalizer = RecordingAgent(), RecordingAgent(), CountingFinalizer()
    state = RunState(
        objective="Compare three approaches for an enterprise AI platform",
        limits=BudgetLimits(max_wall_clock_s=10),
    )
    orchestrator = Orchestrator(
        planner=FakePlanner(),
        researcher=researcher,
        analyst=analyst,
        critic=ScriptedCritic(_pass()),
        finalizer=finalizer,
        clock=_JumpingClock(jump_to=999),
    )

    orchestrator.run(state)  # must NOT raise BudgetExceededError

    assert state.status == RunStatus.COMPLETED_WITH_LIMITATIONS
    assert state.finished_at is not None
    assert state.limitations
    assert researcher.calls == []  # budget was already gone before s1 could start
    assert finalizer.calls == 1  # finalizer still gets a chance to run
    assert state.events[-1].event_type == EventType.RUN_FINISHED


class RateLimitedResearcher(RecordingAgent):
    """Researcher whose provider is rate-limited for the given subtask ids."""

    def __init__(self, failing: set[str]) -> None:
        super().__init__()
        self._failing = failing

    def research(self, state, ctx, subtask):
        self.calls.append(subtask.id)
        if subtask.id in self._failing:
            raise RateLimitError("groq rate limit (HTTP 429)", provider="groq")


def _orchestrator(*, planner=None, researcher=None, critic=None):
    analyst, finalizer = RecordingAgent(), CountingFinalizer()
    orchestrator = Orchestrator(
        planner=planner or FakePlanner(),
        researcher=researcher or RecordingAgent(),
        analyst=analyst,
        critic=critic or ScriptedCritic(_pass()),
        finalizer=finalizer,
    )
    return orchestrator, analyst, finalizer


def test_provider_error_in_one_subtask_degrades_instead_of_failing_run():
    state = RunState(objective="Compare three approaches for an enterprise AI platform")
    orchestrator, analyst, finalizer = _orchestrator(researcher=RateLimitedResearcher({"s1"}))

    orchestrator.run(state)

    assert state.status == RunStatus.COMPLETED_WITH_LIMITATIONS
    assert state.plan.get("s1").status == SubtaskStatus.FAILED
    assert state.plan.get("s2").status == SubtaskStatus.DONE  # the run kept going
    assert analyst.calls == ["s2"]  # no analysis of a subtask whose research failed
    assert finalizer.calls == 1
    assert any("s1" in item and "RateLimitError" in item for item in state.limitations)
    errors = [e for e in state.events if e.event_type == EventType.ERROR]
    assert len(errors) == 1 and errors[0].subtask_id == "s1"
    assert state.events[-1].event_type == EventType.RUN_FINISHED


def test_persistent_provider_failure_stays_bounded_across_critic_retries():
    critic = ScriptedCritic(
        _fail(("s1", CritiqueAction.RESEARCH_MORE)),
        _fail(("s1", CritiqueAction.RESEARCH_MORE)),
    )
    state = RunState(
        objective="Compare three approaches for an enterprise AI platform",
        limits=BudgetLimits(max_critic_cycles=2),
    )
    researcher = RateLimitedResearcher({"s1"})
    orchestrator, _, finalizer = _orchestrator(researcher=researcher, critic=critic)

    orchestrator.run(state)  # must terminate, not loop or raise

    assert state.status == RunStatus.COMPLETED_WITH_LIMITATIONS
    assert finalizer.calls == 1
    assert researcher.calls.count("s1") <= 2


class RateLimitedPlanner:
    def plan(self, state, ctx):
        raise RateLimitError("groq rate limit (HTTP 429)", provider="groq")


def test_planner_provider_failure_fails_run_with_reason_and_closed_trace():
    state = RunState(objective="Compare three approaches for an enterprise AI platform")
    orchestrator, _, finalizer = _orchestrator(planner=RateLimitedPlanner())

    orchestrator.run(state)  # must not raise

    assert state.status == RunStatus.FAILED
    assert state.finished_at is not None
    assert finalizer.calls == 0
    assert any("Planning failed" in item for item in state.limitations)
    types = [e.event_type for e in state.events]
    assert types[0] == EventType.RUN_STARTED
    assert EventType.ERROR in types
    assert types[-1] == EventType.RUN_FINISHED


def test_finalizer_runs_with_released_reserve_after_work_phase_budget_is_spent():
    class _Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = _Clock()

    class _SlowResearcher(RecordingAgent):
        def research(self, state, ctx, subtask):
            super().research(state, ctx, subtask)
            clock.now = 85.0  # past the work limit (80) but inside the full limit (100)

    class _GuardCheckingFinalizer:
        calls = 0

        def finalize(self, state, ctx):
            self.calls += 1
            ctx.guard.check()  # would raise if the reserve were not released

    finalizer = _GuardCheckingFinalizer()
    state = RunState(
        objective="Compare three approaches for an enterprise AI platform",
        limits=BudgetLimits(max_wall_clock_s=100, finalize_reserve_s=20),
    )
    orchestrator = Orchestrator(
        planner=FakePlanner(),
        researcher=_SlowResearcher(),
        analyst=RecordingAgent(),
        critic=ScriptedCritic(_pass()),
        finalizer=finalizer,
        clock=clock,
    )

    orchestrator.run(state)

    assert finalizer.calls == 1
    assert not any("Finalizer could not complete" in item for item in state.limitations)
