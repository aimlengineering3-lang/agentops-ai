import pytest
from fastapi.testclient import TestClient

from agentops.config import Settings, get_settings
from agentops.contracts import BudgetLimits, Critique, CritiqueVerdict, Plan, Subtask
from agentops.orchestrator import EventBus, Orchestrator
from agentops.runs import InMemoryRunRepository, RunManager
from api.dependencies import get_run_manager
from api.main import app, clamp_limits

OBJECTIVE = "Compare three approaches for an enterprise AI platform"


class _Planner:
    def plan(self, state, ctx):
        return Plan(subtasks=[Subtask(id="s1", description="Do it", acceptance_criteria=["a"])])


class _Noop:
    def research(self, state, ctx, subtask):
        pass

    analyze = research


class _Critic:
    def critique(self, state, ctx, *, iteration):
        return Critique(iteration=iteration, verdict=CritiqueVerdict.PASS)


class _Finalizer:
    def finalize(self, state, ctx):
        pass


def _factory(bus: EventBus) -> Orchestrator:
    return Orchestrator(
        planner=_Planner(),
        researcher=_Noop(),
        analyst=_Noop(),
        critic=_Critic(),
        finalizer=_Finalizer(),
        bus=bus,
    )


@pytest.fixture(autouse=True)
def _reset_overrides():
    yield
    app.dependency_overrides.clear()


def _client(daily_cap: int = 30) -> TestClient:
    manager = RunManager(_factory, InMemoryRunRepository())
    app.dependency_overrides[get_run_manager] = lambda: manager
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, daily_run_cap=daily_cap
    )
    return TestClient(app)


def test_daily_run_cap_returns_429_once_reached():
    client = _client(daily_cap=2)

    assert client.post("/runs", json={"objective": OBJECTIVE}).status_code == 201
    assert client.post("/runs", json={"objective": OBJECTIVE}).status_code == 201
    blocked = client.post("/runs", json={"objective": OBJECTIVE})

    assert blocked.status_code == 429
    assert "limit" in blocked.json()["detail"].lower()


def test_too_short_objective_is_a_422_not_a_crash():
    client = _client()

    assert client.post("/runs", json={"objective": "hi"}).status_code == 422


def test_clamp_limits_caps_oversized_requests_and_keeps_small_ones():
    clamped = clamp_limits(BudgetLimits(max_llm_calls=5000, max_wall_clock_s=99999, max_subtasks=3))

    assert clamped.max_llm_calls == 60
    assert clamped.max_wall_clock_s == 600
    assert clamped.max_subtasks == 3
    assert clamp_limits(None) is None
