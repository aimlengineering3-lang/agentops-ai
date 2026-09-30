from typing import Self

from pydantic import BaseModel, Field, model_validator

from .enums import CritiqueAction, CritiqueVerdict, IssueType, Severity


class CritiqueIssue(BaseModel):
    type: IssueType
    subtask_id: str | None = None
    severity: Severity
    action: CritiqueAction
    detail: str = Field(max_length=500)


class Critique(BaseModel):
    iteration: int = Field(ge=1)
    verdict: CritiqueVerdict
    issues: list[CritiqueIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verdict_matches_issues(self) -> Self:
        if self.verdict == CritiqueVerdict.FAIL and not self.issues:
            raise ValueError("a FAIL verdict must list at least one issue")
        has_high = any(i.severity == Severity.HIGH for i in self.issues)
        if self.verdict == CritiqueVerdict.PASS and has_high:
            raise ValueError("a PASS verdict cannot contain HIGH severity issues")
        return self
