from agentops.contracts import AgentName, Event, EventType, RunStatus
from ui.formatting import (
    badge_html,
    citation_numbers,
    event_html,
    event_line,
    format_duration,
    format_tokens,
    status_kind,
    status_label,
)


def test_format_duration_under_a_minute():
    assert format_duration(12.3) == "12.3s"


def test_format_duration_over_a_minute():
    assert format_duration(125.0) == "2m 5s"


def test_status_label_covers_every_status():
    for status in RunStatus:
        assert status_label(status)


def test_status_kind_covers_every_status():
    for status in RunStatus:
        assert status_kind(status) in {"running", "success", "warning", "danger"}


def test_badge_html_contains_label_and_kind_class():
    out = badge_html(RunStatus.COMPLETED)
    assert "success" in out
    assert "Completed" in out


def test_event_line_includes_agent_and_summary():
    event = Event(
        run_id="run_1",
        seq=1,
        agent=AgentName.PLANNER,
        event_type=EventType.PLAN,
        summary="Created 3 subtasks",
    )
    line = event_line(event)
    assert "Planner" in line
    assert "Created 3 subtasks" in line


def test_event_line_marks_failure():
    event = Event(
        run_id="run_1",
        seq=1,
        agent=AgentName.RESEARCHER,
        event_type=EventType.ERROR,
        summary="Search failed",
        success=False,
    )
    assert "⚠️" in event_line(event)


def test_event_html_escapes_html_in_summary():
    event = Event(
        run_id="run_1",
        seq=1,
        agent=AgentName.RESEARCHER,
        event_type=EventType.TOOL_RESULT,
        summary="<script>alert(1)</script>",
    )
    out = event_html(event)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_format_tokens_compacts_large_counts():
    assert format_tokens(842) == "842"
    assert format_tokens(17521) == "17.5K"


def test_citation_numbers_share_a_number_per_source_url():
    from types import SimpleNamespace as NS

    evidence = {
        "ev_a": NS(url="https://a.com/x"),
        "ev_b": NS(url="https://b.com/y"),
        "ev_c": NS(url="https://a.com/x"),  # same page as ev_a
    }
    findings = [NS(evidence_ids=["ev_a", "ev_b"]), NS(evidence_ids=["ev_c", "ev_missing"])]
    numbers = citation_numbers(evidence, findings)
    assert numbers == {"ev_a": 1, "ev_b": 2, "ev_c": 1}
    assert "ev_missing" not in numbers
