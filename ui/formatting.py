import html
from collections.abc import Iterable, Mapping
from typing import Any

from agentops.contracts import Event, RunStatus

STATUS_LABELS: dict[RunStatus, str] = {
    RunStatus.PENDING: "Pending",
    RunStatus.RUNNING: "Running",
    RunStatus.COMPLETED: "Completed",
    RunStatus.COMPLETED_WITH_LIMITATIONS: "Completed (limitations)",
    RunStatus.FAILED: "Failed",
    RunStatus.INTERRUPTED: "Interrupted",
    RunStatus.CANCELLED: "Cancelled",
}

# CSS modifier class used by .ao-badge in ui/theme.py
STATUS_KIND: dict[RunStatus, str] = {
    RunStatus.PENDING: "running",
    RunStatus.RUNNING: "running",
    RunStatus.COMPLETED: "success",
    RunStatus.COMPLETED_WITH_LIMITATIONS: "warning",
    RunStatus.FAILED: "danger",
    RunStatus.INTERRUPTED: "warning",
    RunStatus.CANCELLED: "danger",
}

EVENT_ICONS: dict[str, str] = {
    "run_started": "🚀",
    "plan": "🗺️",
    "llm_call": "🧠",
    "tool_call": "🔧",
    "tool_result": "📄",
    "retry": "🔁",
    "critique": "🔍",
    "validation": "✅",
    "budget": "⏱️",
    "error": "⛔",
    "run_finished": "🏁",
}


def status_label(status: RunStatus) -> str:
    return STATUS_LABELS.get(status, status.value)


def status_kind(status: RunStatus) -> str:
    return STATUS_KIND.get(status, "running")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.0f}s"


def format_tokens(n: int) -> str:
    """12 -> '12', 17521 -> '17.5K'. Raw 5-digit counts are hard to read at a glance."""
    return str(n) if n < 1000 else f"{n / 1000:.1f}K"


def citation_numbers(evidence: Mapping[str, Any], findings: Iterable[Any]) -> dict[str, int]:
    """Map evidence id -> reader-facing citation number ([1], [2], ...).

    Numbers are per unique source URL, in order of first citation, so three evidence items
    from one page all show as [1] -- the reader sees sources, not internal ids.
    Ids that don't resolve in `evidence` are skipped (the caller shows them as unresolved).
    """
    by_source: dict[str, int] = {}
    numbers: dict[str, int] = {}
    for finding in findings:
        for ev_id in finding.evidence_ids:
            item = evidence.get(ev_id)
            if item is None or ev_id in numbers:
                continue
            key = item.url or ev_id
            if key not in by_source:
                by_source[key] = len(by_source) + 1
            numbers[ev_id] = by_source[key]
    return numbers


def event_line(event: Event) -> str:
    """Plain markdown form -- kept for any plain-text/log context."""
    icon = EVENT_ICONS.get(event.event_type.value, "•")
    agent = event.agent.value.capitalize()
    duration = f" ({event.duration_ms}ms)" if event.duration_ms is not None else ""
    fail_marker = " ⚠️" if event.success is False else ""
    return f"{icon} **{agent}** — {event.summary}{duration}{fail_marker}"


def badge_html(status: RunStatus) -> str:
    kind = status_kind(status)
    label = status_label(status)
    return f'<span class="ao-badge {kind}"><span class="ao-dot"></span>{label}</span>'


def event_html(event: Event) -> str:
    """Styled timeline row for the custom HTML trace (see ui/theme.py .ao-timeline)."""
    icon = EVENT_ICONS.get(event.event_type.value, "•")
    agent = html.escape(event.agent.value.capitalize())
    summary = html.escape(event.summary)
    cls = "fail" if event.success is False else ("ok" if event.success else "")
    duration = (
        f'<span class="ao-event-dur">{event.duration_ms}ms</span>'
        if event.duration_ms is not None
        else ""
    )
    return (
        f'<div class="ao-event {cls}">'
        f'<div class="ao-event-head"><span>{icon}</span>'
        f'<span class="ao-event-agent">{agent}</span>{duration}</div>'
        f'<div class="ao-event-summary">{summary}</div>'
        f"</div>"
    )
