from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .analysis import Finding
from .budget import BudgetLimits, BudgetUsage
from .critique import Critique
from .enums import TERMINAL_RUN_STATUSES, AgentName, EventType, RunStatus
from .events import Event
from .evidence import Evidence
from .ids import new_id, utc_now
from .plan import Plan
from .report import Report


class RunState(BaseModel):
    """Single source of truth for one agent run."""

    run_id: str = Field(default_factory=lambda: new_id("run"))
    objective: str = Field(min_length=10, max_length=2000)
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None

    plan: Plan | None = None
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)

    limits: BudgetLimits = Field(default_factory=BudgetLimits)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    events: list[Event] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    report: Report | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_RUN_STATUSES

    def add_event(
        self,
        agent: AgentName,
        event_type: EventType,
        summary: str,
        *,
        subtask_id: str | None = None,
        success: bool | None = None,
        duration_ms: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> Event:
        event = Event(
            run_id=self.run_id,
            seq=len(self.events) + 1,
            agent=agent,
            event_type=event_type,
            subtask_id=subtask_id,
            success=success,
            duration_ms=duration_ms,
            summary=summary,
            data=data or {},
        )
        self.events.append(event)
        return event

    def add_evidence(self, evidence: Evidence) -> None:
        if evidence.id in self.evidence:
            raise ValueError(f"duplicate evidence id: {evidence.id}")
        if self.plan is not None and evidence.subtask_id not in self.plan.subtask_ids:
            raise ValueError(f"evidence refers to unknown subtask: {evidence.subtask_id}")
        self.evidence[evidence.id] = evidence

    def evidence_for(self, subtask_id: str) -> list[Evidence]:
        return [e for e in self.evidence.values() if e.subtask_id == subtask_id]

    def add_finding(self, finding: Finding) -> None:
        """Reject findings that cite evidence we never collected (hallucinated citations)."""
        unknown = [eid for eid in finding.evidence_ids if eid not in self.evidence]
        if unknown:
            raise ValueError(f"finding cites unknown evidence ids: {unknown}")
        self.findings.append(finding)
