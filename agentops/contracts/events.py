from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .enums import AgentName, EventType
from .ids import utc_now


class Event(BaseModel):
    """One line of the audit trace. Short summaries only, never raw content."""

    run_id: str
    seq: int = Field(ge=1)
    ts: datetime = Field(default_factory=utc_now)
    agent: AgentName
    event_type: EventType
    subtask_id: str | None = None
    success: bool | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    summary: str = Field(max_length=300)
    data: dict[str, Any] = Field(default_factory=dict)
