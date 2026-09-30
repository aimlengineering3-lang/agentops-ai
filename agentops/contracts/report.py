from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator

from .ids import new_id, utc_now


class ReportFinding(BaseModel):
    """One key finding as shown in the final report, traceable to its evidence."""

    id: str = Field(default_factory=lambda: new_id("rf"))
    statement: str = Field(min_length=1, max_length=1500)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    # False once the citation verifier (Phase 5) can't resolve an evidence id.
    supported: bool = True


class ReportSource(BaseModel):
    """One deduplicated source listed in the report's sources section."""

    title: str = Field(max_length=300)
    url: str
    domain: str


class ReportMetadata(BaseModel):
    """Numbers a reader can sanity-check the run against: timings, calls, tokens, providers."""

    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    duration_s: float = Field(ge=0.0)
    llm_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    critic_cycles: int = Field(ge=0)
    providers_used: list[str] = Field(default_factory=list)


class Report(BaseModel):
    """The Finalizer's structured, source-backed output. This is what the user reads."""

    id: str = Field(default_factory=lambda: new_id("rpt"))
    objective: str = Field(min_length=10, max_length=2000)
    executive_summary: str = Field(min_length=1, max_length=2000)
    key_findings: list[ReportFinding] = Field(default_factory=list)
    comparison: str | None = None
    trade_offs: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    implementation_considerations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[ReportSource] = Field(default_factory=list)
    validation_passed: bool
    metadata: ReportMetadata

    @model_validator(mode="after")
    def _empty_findings_need_an_explanation(self) -> Self:
        if not self.key_findings and not self.limitations:
            raise ValueError("a report with no key findings must explain why via limitations")
        return self
