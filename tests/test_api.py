import time

import pytest
from fastapi.testclient import TestClient

from agentops.contracts import Critique, CritiqueVerdict, Plan, Subtask
from agentops.orchestrator import EventBus, Orchestrator
from agentops.runs import InMemoryRunRepository, RunManager
from api.dependencies import get_run_manager
from api.main import app

OBJECTIVE = "Compare three approaches for an enterprise AI platform"


class _FakePlanner:
    def plan(self, state, ctx):
        return Plan(subtasks=[Subtask(id="s1", description="Do it", acceptance_criteria=["a"])])


class _FakeAgent:
    def research(self, state, ctx, subtask):
        time.sleep(0.05)  # long enough for the response to reflect PENDING/RUNNING

    analyze = research


class _FakeCritic:
    def critique(self, state, ctx, *, iteration):
        return Critique(iteration=iteration, verdict=CritiqueVerdict.PASS)


class _FakeFinalizer:
    def finalize(self, state, ctx):
        pass


def _fake_orchestrator_factory(bus: EventBus) -> Orchestrator:
    agent = _FakeAgent()
    return Orchestrator(
        planner=_FakePlanner(),
        researcher=agent,
        analyst=agent,
        critic=_FakeCritic(),
        finalizer=_FakeFinalizer(),
        bus=bus,
    )


@pytest.fixture(autouse=True)
def _reset_overrides():
    yield
    app.dependency_overrides.clear()


def _client() -> tuple[TestClient, RunManager]:
    manager = RunManager(_fake_orchestrator_factory, InMemoryRunRepository())
    app.dependency_overrides[get_run_manager] = lambda: manager
    return TestClient(app), manager


def test_health():
    client, _ = _client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_run_returns_201_with_the_objective():
    client, _ = _client()
    response = client.post("/runs", json={"objective": OBJECTIVE})
    assert response.status_code == 201
    body = response.json()
    assert body["objective"] == OBJECTIVE
    assert body["status"] in ("pending", "running")


def test_get_run_after_it_completes():
    client, manager = _client()
    created = client.post("/runs", json={"objective": OBJECTIVE}).json()

    manager.join(created["run_id"], timeout=2.0)
    response = client.get(f"/runs/{created['run_id']}")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_get_unknown_run_returns_404():
    client, _ = _client()
    response = client.get("/runs/does-not-exist")
    assert response.status_code == 404


def test_list_runs_includes_created_run():
    client, manager = _client()
    created = client.post("/runs", json={"objective": OBJECTIVE}).json()
    manager.join(created["run_id"], timeout=2.0)

    response = client.get("/runs")

    ids = [r["run_id"] for r in response.json()]
    assert created["run_id"] in ids


def test_get_events_after_filters_by_seq():
    client, manager = _client()
    created = client.post("/runs", json={"objective": OBJECTIVE}).json()
    manager.join(created["run_id"], timeout=2.0)

    all_events = client.get(f"/runs/{created['run_id']}/events").json()
    assert len(all_events) > 0

    cutoff = all_events[0]["seq"]
    filtered = client.get(f"/runs/{created['run_id']}/events", params={"after": cutoff}).json()

    assert all(e["seq"] > cutoff for e in filtered)


def test_create_run_without_keys_returns_503(monkeypatch):
    # Hermetic on purpose: the developer's real .env (and shell env) must never leak into
    # this test. Without this, a machine with real keys would launch a REAL run in a
    # background thread and burn provider quota on every `pytest`.
    from agentops.config import Settings, get_settings

    for name in (
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "TAVILY_API_KEY",
        "SERPER_API_KEY",
        "GEMINI_MODEL",
        "GROQ_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("api.dependencies.get_settings", lambda: Settings(_env_file=None))
    client = TestClient(app)

    response = client.post("/runs", json={"objective": OBJECTIVE})

    assert response.status_code == 503
    get_settings.cache_clear()


def test_llm_router_gets_one_provider_per_gemini_model_then_groq(monkeypatch):
    from agentops.config import Settings
    from api.dependencies import _build_llm_router

    settings = Settings(
        _env_file=None,
        gemini_api_key="k",
        gemini_model="model-a",
        gemini_extra_models="model-b, model-a, model-c",
        groq_api_key="g",
        groq_model="groq-model",
    )
    monkeypatch.setattr("api.dependencies.get_settings", lambda: settings)

    router = _build_llm_router()

    assert [p.name for p in router._providers] == [
        "gemini",
        "gemini:model-b",
        "gemini:model-c",
        "groq",
    ]
