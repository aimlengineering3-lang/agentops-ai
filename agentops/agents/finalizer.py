from pydantic import BaseModel, Field

from agentops.contracts import (
    AgentName,
    Evidence,
    Finding,
    Report,
    ReportFinding,
    ReportMetadata,
    ReportSource,
    RunState,
)
from agentops.errors import BudgetExceededError, MalformedOutputError, ProviderError
from agentops.llm import Generator, Message
from agentops.orchestrator import RunContext

from .llm_call import call_llm_structured

_SYSTEM_PROMPT = """\
You are the Finalizer in an autonomous research agent. You are given a
numbered list of Findings already produced for this run (kind, confidence,
statement). Compose a structured report by synthesizing them. Return ONLY
JSON, no prose:

{
  "executive_summary": "<2-4 sentence overview of the objective and outcome>",
  "key_findings": [
    {"statement": "<1-2 sentence key finding, in your own words>",
     "finding_indices": [<int, ...>],
     "confidence": <float 0.0-1.0>}
  ],
  "comparison": "<comparison across the options, or null if not applicable>",
  "trade_offs": ["<trade-off>", ...],
  "risks": ["<risk>", ...],
  "implementation_considerations": ["<consideration>", ...]
}

Rules:
- finding_indices MUST be indices from the list below. Never invent one.
- Every key finding should draw on at least one finding_indices entry unless
  it is a genuinely unsupported synthesis -- prefer citing over not citing.
- Keep trade_offs, risks and implementation_considerations short and concrete.
"""


class _DraftFinding(BaseModel):
    statement: str = Field(min_length=1, max_length=1500)
    finding_indices: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class _DraftReport(BaseModel):
    executive_summary: str = Field(min_length=1, max_length=2000)
    key_findings: list[_DraftFinding] = Field(default_factory=list)
    comparison: str | None = None
    trade_offs: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    implementation_considerations: list[str] = Field(default_factory=list)


def _format_findings(findings: list[Finding]) -> str:
    lines = [
        f"[{i}] (kind={f.kind.value}, confidence={f.confidence}) {f.statement!r}"
        for i, f in enumerate(findings)
    ]
    return "\n".join(lines) or "(no findings)"


def _fallback_draft(findings: list[Finding]) -> _DraftReport:
    """Used when the LLM synthesis call fails outright. One key finding per Finding,
    verbatim -- plain but honest, and it guarantees finalize() never leaves state.report
    unset just because a single LLM call had a bad day."""
    return _DraftReport(
        executive_summary="Automated synthesis was unavailable; findings are listed as-is.",
        key_findings=[
            _DraftFinding(statement=f.statement, finding_indices=[i], confidence=f.confidence)
            for i, f in enumerate(findings)
        ],
    )


def _build_sources(evidence: dict[str, Evidence]) -> list[ReportSource]:
    seen_urls: set[str] = set()
    sources: list[ReportSource] = []
    for e in evidence.values():
        if e.url is None or e.url in seen_urls:
            continue
        seen_urls.add(e.url)
        sources.append(ReportSource(title=e.title, url=e.url, domain=e.domain or ""))
    return sources


def _providers_used(state: RunState) -> list[str]:
    providers = {e.data.get("provider") for e in state.events if e.data.get("provider") is not None}
    providers |= {ev.provider for ev in state.evidence.values()}
    return sorted(providers)


def _verify_citations(state: RunState, key_findings: list[ReportFinding]) -> None:
    """Deterministic citation verifier: trusts nothing upstream. Every evidence_ids entry
    must resolve to real Evidence actually present in state -- checked here directly against
    state.evidence, not against the Finding objects that produced it. A key finding left with
    no resolvable evidence is kept (the reader still sees the claim) but marked unsupported.
    """
    for kf in key_findings:
        resolved = [eid for eid in kf.evidence_ids if eid in state.evidence]
        kf.evidence_ids = resolved
        if not resolved:
            kf.supported = False


class LLMFinalizer:
    """Finalizer agent: LLM synthesizes Findings into a narrative report; a deterministic
    citation verifier then checks every claim before it reaches the reader.

    Key findings cite Finding *indices*, not evidence directly: synthesis re-groups and
    narrates findings the Analyst already produced and grounded, so re-deriving citations
    from raw Evidence here would duplicate the Analyst's job. Each key finding's evidence_ids
    is the union of its cited findings' own evidence_ids, then re-checked by the verifier.
    """

    def __init__(self, generator: Generator, *, temperature: float = 0.2) -> None:
        self._generator = generator
        self._temperature = temperature

    def finalize(self, state: RunState, ctx: RunContext) -> None:
        draft = self._draft_report(state, ctx)

        key_findings = [
            self._to_report_finding(item, state.findings) for item in draft.key_findings
        ]
        _verify_citations(state, key_findings)
        if not key_findings and not state.limitations:
            state.limitations.append("No key findings could be synthesized")

        # Refresh the elapsed-time counter for the report; stopping here would defeat the
        # point of finalizing, so a budget hit at this point is deliberately swallowed.
        try:
            ctx.guard.check()
        except BudgetExceededError:
            pass

        state.report = Report(
            objective=state.objective,
            executive_summary=draft.executive_summary,
            key_findings=key_findings,
            comparison=draft.comparison,
            trade_offs=draft.trade_offs,
            risks=draft.risks,
            implementation_considerations=draft.implementation_considerations,
            limitations=list(state.limitations),
            sources=_build_sources(state.evidence),
            validation_passed=all(kf.supported for kf in key_findings),
            metadata=ReportMetadata(
                run_id=state.run_id,
                duration_s=state.usage.elapsed_s,
                llm_calls=state.usage.llm_calls,
                tool_calls=state.usage.tool_calls,
                tokens_in=state.usage.tokens_in,
                tokens_out=state.usage.tokens_out,
                critic_cycles=state.usage.critic_cycles,
                providers_used=_providers_used(state),
            ),
        )

    # -- steps ---------------------------------------------------------------------------

    def _draft_report(self, state: RunState, ctx: RunContext) -> _DraftReport:
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"Objective: {state.objective}\n\nFindings:\n{_format_findings(state.findings)}"
                ),
            ),
        ]
        try:
            return call_llm_structured(
                state,
                ctx,
                AgentName.FINALIZER,
                self._generator,
                messages,
                _DraftReport,
                temperature=self._temperature,
            )
        except (MalformedOutputError, ProviderError, BudgetExceededError) as exc:
            state.limitations.append(f"Report synthesis failed, using raw findings ({exc})")
            return _fallback_draft(state.findings)

    def _to_report_finding(self, item: _DraftFinding, findings: list[Finding]) -> ReportFinding:
        evidence_ids: list[str] = []
        for index in item.finding_indices:
            if not 0 <= index < len(findings):
                continue  # model referenced a finding we never showed it
            for eid in findings[index].evidence_ids:
                if eid not in evidence_ids:
                    evidence_ids.append(eid)
        return ReportFinding(
            statement=item.statement,
            evidence_ids=evidence_ids,
            confidence=item.confidence,
        )
