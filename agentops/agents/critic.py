from typing import Literal

from pydantic import BaseModel, Field

from agentops.contracts import (
    AgentName,
    Critique,
    CritiqueAction,
    CritiqueIssue,
    CritiqueVerdict,
    Finding,
    IssueType,
    RunState,
    Severity,
    SubtaskStatus,
)
from agentops.errors import MalformedOutputError, ProviderError
from agentops.llm import Generator, Message
from agentops.orchestrator import RunContext

from .llm_call import call_llm_structured

_LLM_SYSTEM_PROMPT = """\
You are the Critic reviewing findings from a research run. You are given a
numbered list of findings (kind, confidence, statement). Look for:
- contradictions between two or more findings
- "analysis" or "conclusion" findings stated with unwarranted confidence given
  weak or no supporting evidence

Return ONLY JSON, no prose:
{
  "issues": [
    {"finding_index": <int, index from the list below>,
     "type": "contradiction" | "unsupported_claim",
     "severity": "low" | "medium" | "high",
     "detail": "<what is wrong, at most 200 chars>"}
  ]
}
If nothing is wrong, return {"issues": []}.
"""


class _LLMIssueItem(BaseModel):
    finding_index: int = Field(ge=0)
    type: Literal["contradiction", "unsupported_claim"]
    severity: Severity
    detail: str = Field(min_length=1, max_length=200)


class _LLMIssues(BaseModel):
    issues: list[_LLMIssueItem] = Field(default_factory=list)


def _format_findings(findings: list[Finding]) -> str:
    lines = [
        f"[{i}] (kind={f.kind.value}, confidence={f.confidence}, subtask={f.subtask_id}) "
        f"{f.statement!r}"
        for i, f in enumerate(findings)
    ]
    return "\n".join(lines) or "(no findings)"


class HybridCritic:
    """Critic agent: cheap deterministic checks always run first; one LLM pass then looks
    for contradictions and unsupported claims across the findings collected so far.

    The two passes check different things -- structural sufficiency (enough evidence, enough
    distinct sources, every subtask actually produced findings) versus semantic quality
    (does a conclusion overreach its evidence, do two findings disagree) -- so a deterministic
    issue on one subtask says nothing about whether another subtask's reasoning holds up.
    Both passes run every cycle; only "no findings exist yet" skips the LLM pass outright,
    since there is nothing yet for it to review.
    """

    def __init__(
        self,
        generator: Generator,
        *,
        temperature: float = 0.2,
        min_evidence_per_subtask: int = 2,
        min_source_domains: int = 2,
    ) -> None:
        self._generator = generator
        self._temperature = temperature
        self._min_evidence = min_evidence_per_subtask
        self._min_domains = min_source_domains

    def critique(self, state: RunState, ctx: RunContext, *, iteration: int) -> Critique:
        issues = self._deterministic_checks(state)
        issues += self._llm_checks(state, ctx)
        verdict = CritiqueVerdict.FAIL if issues else CritiqueVerdict.PASS
        return Critique(iteration=iteration, verdict=verdict, issues=issues)

    # -- deterministic ---------------------------------------------------------------------

    def _deterministic_checks(self, state: RunState) -> list[CritiqueIssue]:
        assert state.plan is not None
        issues: list[CritiqueIssue] = []
        for subtask in state.plan.subtasks:
            if subtask.status != SubtaskStatus.DONE:
                issues.append(
                    CritiqueIssue(
                        type=IssueType.MISSING_SUBTASK,
                        subtask_id=subtask.id,
                        severity=Severity.HIGH,
                        action=CritiqueAction.REPLAN,
                        detail=f"Subtask {subtask.id} never completed "
                        f"(status={subtask.status.value})",
                    )
                )
                continue

            evidence = state.evidence_for(subtask.id)
            findings = [f for f in state.findings if f.subtask_id == subtask.id]

            if not evidence:
                issues.append(
                    CritiqueIssue(
                        type=IssueType.INSUFFICIENT_EVIDENCE,
                        subtask_id=subtask.id,
                        severity=Severity.HIGH,
                        action=CritiqueAction.RESEARCH_MORE,
                        detail=f"Subtask {subtask.id} has no evidence",
                    )
                )
            elif len(evidence) < self._min_evidence:
                issues.append(
                    CritiqueIssue(
                        type=IssueType.INSUFFICIENT_EVIDENCE,
                        subtask_id=subtask.id,
                        severity=Severity.MEDIUM,
                        action=CritiqueAction.RESEARCH_MORE,
                        detail=f"Subtask {subtask.id} has only {len(evidence)} evidence item(s)",
                    )
                )
            else:
                domains = {e.domain for e in evidence if e.domain}
                if len(domains) < self._min_domains:
                    issues.append(
                        CritiqueIssue(
                            type=IssueType.WEAK_SOURCE_COVERAGE,
                            subtask_id=subtask.id,
                            severity=Severity.MEDIUM,
                            action=CritiqueAction.RESEARCH_MORE,
                            detail=f"Subtask {subtask.id} evidence spans only "
                            f"{len(domains)} distinct domain(s)",
                        )
                    )

            if evidence and not findings:
                issues.append(
                    CritiqueIssue(
                        type=IssueType.INCOMPLETE_OUTPUT,
                        subtask_id=subtask.id,
                        severity=Severity.HIGH,
                        action=CritiqueAction.REANALYZE,
                        detail=f"Subtask {subtask.id} has evidence but no findings",
                    )
                )
        return issues

    # -- LLM ---------------------------------------------------------------------------------

    def _llm_checks(self, state: RunState, ctx: RunContext) -> list[CritiqueIssue]:
        if not state.findings:
            return []  # nothing to review yet

        messages = [
            Message(role="system", content=_LLM_SYSTEM_PROMPT),
            Message(role="user", content=_format_findings(state.findings)),
        ]
        try:
            parsed = call_llm_structured(
                state,
                ctx,
                AgentName.CRITIC,
                self._generator,
                messages,
                _LLMIssues,
                temperature=self._temperature,
            )
        except (MalformedOutputError, ProviderError) as exc:
            state.limitations.append(f"Critic LLM review skipped: {exc}")
            return []

        issues: list[CritiqueIssue] = []
        for item in parsed.issues:
            if not 0 <= item.finding_index < len(state.findings):
                continue  # model referenced a finding we never showed it
            finding = state.findings[item.finding_index]
            is_contradiction = item.type == "contradiction"
            issues.append(
                CritiqueIssue(
                    type=IssueType.CONTRADICTION
                    if is_contradiction
                    else IssueType.UNSUPPORTED_CLAIM,
                    subtask_id=finding.subtask_id,
                    severity=item.severity,
                    action=CritiqueAction.REANALYZE
                    if is_contradiction
                    else CritiqueAction.FIX_OUTPUT,
                    detail=item.detail,
                )
            )
        return issues
