import logging
from collections.abc import Callable, Iterable
from typing import Any

from agentops.contracts import AgentName, Event, EventType, RunState

logger = logging.getLogger(__name__)

Subscriber = Callable[[Event], None]

_MAX_SUMMARY = 300  # must match Event.summary max_length


class EventBus:
    """Single place where events are created. Appends to RunState, then notifies subscribers.

    Subscribers (DB writer, logger, UI queue) are added later without touching agent code.
    """

    def __init__(self, subscribers: Iterable[Subscriber] = ()) -> None:
        self._subscribers: list[Subscriber] = list(subscribers)

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    def emit(
        self,
        state: RunState,
        agent: AgentName,
        event_type: EventType,
        summary: str,
        *,
        subtask_id: str | None = None,
        success: bool | None = None,
        duration_ms: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> Event:
        # Truncate instead of failing validation: a long summary must never crash a run.
        if len(summary) > _MAX_SUMMARY:
            summary = summary[: _MAX_SUMMARY - 1] + "…"
        event = state.add_event(
            agent,
            event_type,
            summary,
            subtask_id=subtask_id,
            success=success,
            duration_ms=duration_ms,
            data=data,
        )
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception:
                # Observability must never take the run down.
                logger.exception("event subscriber failed")
        return event
