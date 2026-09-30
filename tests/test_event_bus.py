import logging

from agentops.contracts import AgentName, EventType, RunState
from agentops.orchestrator import EventBus


def _state() -> RunState:
    return RunState(objective="Compare three approaches for an enterprise AI platform")


def test_emit_appends_event_and_notifies_subscribers():
    seen = []
    bus = EventBus([seen.append])
    state = _state()
    event = bus.emit(state, AgentName.PLANNER, EventType.PLAN, "Plan created", success=True)
    assert state.events == [event]
    assert seen == [event]
    assert event.seq == 1


def test_long_summary_is_truncated_not_rejected():
    bus = EventBus()
    event = bus.emit(_state(), AgentName.ORCHESTRATOR, EventType.ERROR, "x" * 1000)
    assert len(event.summary) == 300


def test_failing_subscriber_does_not_break_the_run(caplog):
    def broken(_event):
        raise RuntimeError("db down")

    good = []
    bus = EventBus([broken, good.append])
    state = _state()
    with caplog.at_level(logging.ERROR):
        bus.emit(state, AgentName.ORCHESTRATOR, EventType.RUN_STARTED, "start")
    assert len(state.events) == 1
    assert len(good) == 1  # later subscribers still run
    assert "event subscriber failed" in caplog.text
