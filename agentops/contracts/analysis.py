from typing import Self

from pydantic import BaseModel, Field, model_validator

from .enums import FindingKind
from .ids import new_id


class Finding(BaseModel):
    """A statement produced by the Analyst, linked to the evidence behind it."""

    id: str = Field(default_factory=lambda: new_id("fd"))
    subtask_id: str
    statement: str = Field(min_length=1, max_length=1500)
    kind: FindingKind
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _evidence_findings_must_cite(self) -> Self:
        if self.kind == FindingKind.EVIDENCE and not self.evidence_ids:
            raise ValueError("a finding of kind 'evidence' must cite evidence ids")
        return self
