import json

import httpx

from agentops.contracts import AgentName, BudgetLimits, EventType, RunState, RunStatus
from ui.api_client import AgentOpsClient

OBJECTIVE = "Compare three approaches for an enterprise AI platform"


def _client(handler) -> AgentOpsClient:
    http_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    return AgentOpsClient("http://test", client=http_client)


def _json_body(request: httpx.Request) -> dict:
    return json.loads(request.read())


def test_health_true_on_200():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    assert _client(handler).health() is True


def test_health_false_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    assert _client(handler).health() is False


def test_create_run_parses_response_into_a_run_state():
    expected = RunState(objective=OBJECTIVE)

    def handler(request):
        assert request.url.path == "/runs"
        return httpx.Response(201, json=expected.model_dump(mode="json"))

    run = _client(handler).create_run(OBJECTIVE)

    assert run.run_id == expected.run_id
    assert run.status == RunStatus.PENDING


def test_create_run_sends_limits_when_given():
    captured = {}

    def handler(request):
        captured["body"] = _json_body(request)
        return httpx.Response(201, json=RunState(objective=OBJECTIVE).model_dump(mode="json"))

    _client(handler).create_run(OBJECTIVE, BudgetLimits(max_subtasks=3))

    assert captured["body"]["limits"]["max_subtasks"] == 3


def test_get_run_returns_none_on_404():
    def handler(request):
        return httpx.Response(404, json={"detail": "no run"})

    assert _client(handler).get_run("nope") is None


def test_get_run_returns_parsed_state():
    expected = RunState(objective=OBJECTIVE)

    def handler(request):
        return httpx.Response(200, json=expected.model_dump(mode="json"))

    run = _client(handler).get_run(expected.run_id)

    assert run is not None
    assert run.run_id == expected.run_id


def test_list_runs_sends_limit_and_parses_states():
    a, b = RunState(objective=OBJECTIVE), RunState(objective=OBJECTIVE)

    def handler(request):
        assert request.url.params["limit"] == "10"
        return httpx.Response(200, json=[a.model_dump(mode="json"), b.model_dump(mode="json")])

    runs = _client(handler).list_runs(limit=10)

    assert [r.run_id for r in runs] == [a.run_id, b.run_id]


def test_get_events_sends_after_param_and_parses_events():
    state = RunState(objective=OBJECTIVE)
    event = state.add_event(AgentName.ORCHESTRATOR, EventType.RUN_STARTED, "started")

    def handler(request):
        assert request.url.params["after"] == "5"
        return httpx.Response(200, json=[event.model_dump(mode="json")])

    events = _client(handler).get_events(state.run_id, after=5)

    assert len(events) == 1
    assert events[0].event_type == EventType.RUN_STARTED
