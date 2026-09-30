import json

from agentops.agents.analyst import LLMAnalyst
from agentops.contracts import EventType, Evidence, FindingKind, RunState, SourceType, Subtask
from agentops.orchestrator import BudgetGuard, EventBus, RunContext
from agentops.testing.fakes import FakeLLM

SUBTASK = Subtask(
    id="s1",
    description="Research FastAPI, Django and Flask for a small team",
    acceptance_criteria=["at least two independent sources"],
)


def _context(state: RunState) -> RunContext:
    bus = EventBus()
    return RunContext(bus=bus, guard=BudgetGuard(state, bus))


def _evidence(claim: str, url: str = "https://example.com") -> Evidence:
    return Evidence(
        subtask_id="s1",
        source_type=SourceType.WEB,
        title="Doc",
        url=url,
        extracted_claim=claim,
        provider="fake-search",
    )


def _state_with_evidence(*claims: str) -> RunState:
    state = RunState(objective="Compare three approaches for an enterprise AI platform")
    for i, claim in enumerate(claims):
        state.add_evidence(_evidence(claim, url=f"https://example.com/{i}"))
    return state


def test_happy_path_maps_index_to_real_evidence_id():
    state = _state_with_evidence("FastAPI is async-first.", "Django has a large ecosystem.")
    (ev0, ev1) = state.evidence.values()
    findings_json = json.dumps(
        {
            "findings": [
                {
                    "statement": "FastAPI is async-first.",
                    "kind": "evidence",
                    "evidence_indices": [0],
                    "confidence": 0.9,
                },
                {
                    "statement": "FastAPI trades ecosystem size for performance vs Django.",
                    "kind": "analysis",
                    "evidence_indices": [0, 1],
                    "confidence": 0.6,
                },
            ]
        }
    )
    llm = FakeLLM([findings_json])
    analyst = LLMAnalyst(llm)

    analyst.analyze(state, _context(state), SUBTASK)

    assert len(state.findings) == 2
    evidence_finding = state.findings[0]
    assert evidence_finding.kind == FindingKind.EVIDENCE
    assert evidence_finding.evidence_ids == [ev0.id]
    analysis_finding = state.findings[1]
    assert analysis_finding.kind == FindingKind.ANALYSIS
    assert analysis_finding.evidence_ids == [ev0.id, ev1.id]


def test_no_evidence_skips_the_llm_call_and_records_a_limitation():
    state = RunState(objective="Compare three approaches for an enterprise AI platform")
    llm = FakeLLM([])
    analyst = LLMAnalyst(llm)

    analyst.analyze(state, _context(state), SUBTASK)

    assert state.findings == []
    assert llm.calls == []
    assert any("no evidence to analyze" in m for m in state.limitations)


def test_out_of_range_index_is_dropped_not_crashed():
    state = _state_with_evidence("FastAPI is async-first.")
    findings_json = json.dumps(
        {
            "findings": [
                {
                    "statement": "bogus",
                    "kind": "evidence",
                    "evidence_indices": [99],
                    "confidence": 0.5,
                }
            ]
        }
    )
    llm = FakeLLM([findings_json])
    analyst = LLMAnalyst(llm)

    analyst.analyze(state, _context(state), SUBTASK)  # must not raise

    # evidence-kind finding with no valid citations left fails its own validator -> dropped
    assert state.findings == []


def test_conclusion_with_no_citation_is_allowed():
    state = _state_with_evidence("FastAPI is async-first.")
    findings_json = json.dumps(
        {
            "findings": [
                {
                    "statement": "FastAPI is the better fit for this team.",
                    "kind": "conclusion",
                    "evidence_indices": [],
                    "confidence": 0.4,
                }
            ]
        }
    )
    llm = FakeLLM([findings_json])
    analyst = LLMAnalyst(llm)

    analyst.analyze(state, _context(state), SUBTASK)

    assert len(state.findings) == 1
    assert state.findings[0].kind == FindingKind.CONCLUSION
    assert state.findings[0].evidence_ids == []


def test_analysis_failure_is_a_limitation_not_a_crash():
    state = _state_with_evidence("FastAPI is async-first.")
    llm = FakeLLM(["not json", "still not json"])
    analyst = LLMAnalyst(llm)

    analyst.analyze(state, _context(state), SUBTASK)

    assert state.findings == []
    assert any("analysis failed" in m for m in state.limitations)


def test_llm_call_event_is_emitted():
    state = _state_with_evidence("FastAPI is async-first.")
    llm = FakeLLM([json.dumps({"findings": []})])
    analyst = LLMAnalyst(llm)

    analyst.analyze(state, _context(state), SUBTASK)

    assert any(e.event_type == EventType.LLM_CALL for e in state.events)
    assert state.usage.llm_calls == 1
