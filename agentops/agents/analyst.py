from pydantic import BaseModel, Field, ValidationError

from agentops.contracts import AgentName, Evidence, FindingKind, RunState, Subtask
from agentops.contracts.analysis import Finding
from agentops.errors import MalformedOutputError, ProviderError
from agentops.llm import Generator, Message
from agentops.orchestrator import RunContext

from .llm_call import call_llm_structured

_SYSTEM_PROMPT = """\
You are the Analyst in an autonomous research agent. You are given a numbered
list of Evidence collected for one subtask. Produce findings and return ONLY
JSON, no prose:

{
  "findings": [
    {"statement": "<1-2 sentence finding, in your own words>",
     "kind": "evidence" | "analysis" | "conclusion",
     "evidence_indices": [<int, ...>],
     "confidence": <float 0.0-1.0>}
  ]
}

Kinds, keep these distinct:
- "evidence": a claim taken directly from one source. Must cite at least one
  evidence_indices entry.
- "analysis": derived by comparing or combining more than one piece of
  evidence (e.g. a trade-off). Cite the evidence_indices it is derived from.
- "conclusion": your own judgment or recommendation. Cite evidence_indices
  only if it is grounded in specific evidence; otherwise leave it empty, but
  never present a conclusion as if it were a verified fact.

Rules:
- evidence_indices MUST be indices from the list below. Never invent one.
- Do not restate the same evidence twice under different findings.
- If the evidence list is empty or useless, return {"findings": []}.
"""


class _FindingItem(BaseModel):
    statement: str = Field(min_length=1, max_length=1500)
    kind: FindingKind
    evidence_indices: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class _AnalysisFindings(BaseModel):
    findings: list[_FindingItem] = Field(default_factory=list)


def _format_evidence(evidence: list[Evidence]) -> str:
    lines = [
        f"[{i}] ({e.domain or e.source_type.value}) {e.extracted_claim!r}"
        for i, e in enumerate(evidence)
    ]
    return "\n".join(lines) or "(no evidence)"


class LLMAnalyst:
    """Analyst agent: turns one subtask's Evidence into Findings, via one structured LLM call.

    Findings cite evidence by index into the list this call was given, mapped back to real
    Evidence ids in code -- same grounding principle as the Researcher's source_index: the
    model never writes an id string itself, so it cannot cite an evidence id that does not
    exist.
    """

    def __init__(self, generator: Generator, *, temperature: float = 0.2) -> None:
        self._generator = generator
        self._temperature = temperature

    def analyze(self, state: RunState, ctx: RunContext, subtask: Subtask) -> None:
        evidence = state.evidence_for(subtask.id)
        if not evidence:
            # Nothing to analyze; the Critic will flag insufficient evidence. Spending an
            # LLM call here would only ask the model to invent findings from nothing.
            state.limitations.append(f"Subtask {subtask.id}: no evidence to analyze")
            return

        items = self._propose_findings(state, ctx, subtask, evidence)
        if items is None:
            return  # failure already recorded as a limitation

        for item in items:
            self._add_finding(state, subtask, item, evidence)

    # -- steps ---------------------------------------------------------------------------

    def _propose_findings(
        self, state: RunState, ctx: RunContext, subtask: Subtask, evidence: list[Evidence]
    ) -> list[_FindingItem] | None:
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"Subtask: {subtask.description}\n"
                    f"Acceptance criteria: {subtask.acceptance_criteria}\n\n"
                    f"Evidence:\n{_format_evidence(evidence)}"
                ),
            ),
        ]
        try:
            parsed = call_llm_structured(
                state,
                ctx,
                AgentName.ANALYST,
                self._generator,
                messages,
                _AnalysisFindings,
                temperature=self._temperature,
            )
        except (MalformedOutputError, ProviderError) as exc:
            state.limitations.append(f"Subtask {subtask.id}: analysis failed ({exc})")
            return None
        return parsed.findings

    def _add_finding(
        self, state: RunState, subtask: Subtask, item: _FindingItem, evidence: list[Evidence]
    ) -> None:
        evidence_ids = []
        for index in item.evidence_indices:
            if 0 <= index < len(evidence) and evidence[index].id not in evidence_ids:
                evidence_ids.append(evidence[index].id)
        try:
            finding = Finding(
                subtask_id=subtask.id,
                statement=item.statement,
                kind=item.kind,
                evidence_ids=evidence_ids,
                confidence=item.confidence,
            )
        except ValidationError:
            # e.g. kind="evidence" with no valid indices left after mapping; drop it
            # rather than crash -- the Critic will notice the resulting gap.
            return
        state.add_finding(finding)
