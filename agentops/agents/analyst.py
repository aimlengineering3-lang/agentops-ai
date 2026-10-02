from pydantic import BaseModel, Field, ValidationError

from agentops.contracts import AgentName, Evidence, FindingKind, RunState, Subtask
from agentops.contracts.analysis import Finding
from agentops.errors import MalformedOutputError, ProviderError
from agentops.llm import Generator, Message
from agentops.orchestrator import RunContext
from agentops.tools import CalculatorError, safe_eval

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
     "confidence": <float 0.0-1.0>,
     "calculation": "<arithmetic expression, or omit this field>"}
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

If a finding states a number you got by adding, subtracting, multiplying or
dividing numbers from the evidence (a total cost, a budget range, a sum of
fees), set "calculation" to the plain arithmetic expression that produces it
(e.g. "133396 + 5300 + 44631"), using only numbers and + - * / ( ). Do not put
units, currency symbols or variable names in it. This lets the figure in your
statement be checked, not just trusted. Omit the field entirely when a finding
has no such arithmetic.

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
    calculation: str | None = Field(default=None, max_length=200)


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
        statement = self._append_verified_calculation(state, subtask, item)
        try:
            finding = Finding(
                subtask_id=subtask.id,
                statement=statement,
                kind=item.kind,
                evidence_ids=evidence_ids,
                confidence=item.confidence,
            )
        except ValidationError:
            # e.g. kind="evidence" with no valid indices left after mapping; drop it
            # rather than crash -- the Critic will notice the resulting gap.
            return
        state.add_finding(finding)

    def _append_verified_calculation(
        self, state: RunState, subtask: Subtask, item: _FindingItem
    ) -> str:
        """If the model flagged an arithmetic expression behind this finding's number,
        verify it with the deterministic calculator tool (never trust the model's own
        mental arithmetic) and make the checked result visible in the statement.
        """
        if not item.calculation:
            return item.statement
        try:
            result = safe_eval(item.calculation)
        except CalculatorError as exc:
            state.limitations.append(
                f"Subtask {subtask.id}: could not verify calculation {item.calculation!r} ({exc})"
            )
            return item.statement
        number = int(result) if result.is_integer() else round(result, 2)
        annotated = f"{item.statement} (verified: {item.calculation} = {number})"
        # Finding.statement has the same 1500-char ceiling as _FindingItem; an
        # already-long statement plus the annotation could exceed it, and losing the
        # whole finding over a cosmetic suffix would be a worse outcome than a plain one.
        return annotated if len(annotated) <= 1500 else item.statement
