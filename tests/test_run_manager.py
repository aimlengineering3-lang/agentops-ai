import time

import pytest

from agentops.contracts import Critique, CritiqueVerdict, Plan, RunStatus, Subtask
from agentops.orchestrator import EventBus, Orchestrator
from agentops.runs import InMemoryRunRepository, RunManager

OBJECTIVE = "Compare three approaches for an enterprise AI platform"


class _FakePlanner:
    def plan(self, state, ctx):
        return Plan(subtasks=[Subtask(id="s1", description="Do it", acceptance_criteria=["a"])])


class _FakeAgent:
    def research(self, state, ctx, subtask):
        time.sleep(0.05)  # long enough for the main thread's snapshot to catch RUNNING

    analyze = research


class _FakeCritic:
    def critique(self, state, ctx, *, iteration):
        return Critique(iteration=iteration, verdict=CritiqueVerdict.PASS)


class _FakeFinalizer:
    def finalize(self, state, ctx):
        pass


def _orchestrator_factory(bus: EventBus) -> Orchestrator:
    agent = _FakeAgent()
    return Orchestrator(
        planner=_FakePlanner(),
        researcher=agent,
        analyst=agent,
        critic=_FakeCritic(),
        finalizer=_FakeFinalizer(),
        bus=bus,
    )


def _manager() -> RunManager:
    return RunManager(_orchestrator_factory, InMemoryRunRepository())


def test_start_run_returns_immediately_in_pending_or_running_state():
    manager = _manager()

    state = manager.start_run(OBJECTIVE)

    assert state.status in (RunStatus.PENDING, RunStatus.RUNNING)
    manager.join(state.run_id, timeout=2.0)  # let the background thread finish first


def test_run_completes_in_the_background_and_is_retrievable():
    manager = _manager()
    started = manager.start_run(OBJECTIVE)

    finished = manager.join(started.run_id, timeout=2.0)

    assert finished.status == RunStatus.COMPLETED
    assert manager.get_run(started.run_id).status == RunStatus.COMPLETED


def test_events_are_persisted_incrementally():
    manager = _manager()
    started = manager.start_run(OBJECTIVE)
    manager.join(started.run_id, timeout=2.0)

    fetched = manager.get_run(started.run_id)
    assert len(fetched.events) > 0  # the per-event subscriber wrote state through


def test_get_run_returns_none_for_unknown_id():
    manager = _manager()
    assert manager.get_run("nope") is None


def test_list_runs_includes_started_runs():
    manager = _manager()
    a = manager.start_run(OBJECTIVE)
    manager.join(a.run_id, timeout=2.0)
    b = manager.start_run(OBJECTIVE)
    manager.join(b.run_id, timeout=2.0)

    run_ids = {r.run_id for r in manager.list_runs(limit=10)}

    assert {a.run_id, b.run_id} <= run_ids


def test_join_raises_for_unknown_run_id():
    manager = _manager()
    with pytest.raises(KeyError):
        manager.join("nope", timeout=1.0)
