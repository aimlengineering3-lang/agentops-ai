import json

from agentops.agents.finalizer import LLMFinalizer
from agentops.contracts import Evidence, Finding, FindingKind, RunState, SourceType
from agentops.orchestrator import BudgetGuard, EventBus, RunContext
from agentops.testing.fakes import FakeLLM

OBJECTIVE = "Compare three approaches for an enterprise AI platform"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _context(state: RunState, clock: FakeClock | None = None) -> RunContext:
    bus = EventBus()
    return RunContext(bus=bus, guard=BudgetGuard(state, bus, clock or FakeClock()))


def _evidence(url: str) -> Evidence:
    return Evidence(
        subtask_id="s1",
        source_type=SourceType.WEB,
        title=f"Doc {url}",
        url=url,
        extracted_claim="claim",
        provider="tavily",
    )


def _finding(statement: str, evidence_ids: list[str] | None = None) -> Finding:
    return Finding(
        subtask_id="s1",
        statement=statement,
        kind=FindingKind.EVIDENCE if evidence_ids else FindingKind.CONCLUSION,
        evidence_ids=evidence_ids or [],
        confidence=0.7,
    )


def _state_with_finding() -> tuple[RunState, Finding]:
    state = RunState(objective=OBJECTIVE)
    state.plan = None
    ev = _evidence("https://a.example")
    state.add_evidence(ev)
    finding = _finding("FastAPI is async-first.", [ev.id])
    state.add_finding(finding)
    return state, finding


def test_happy_path_resolves_citations_and_metadata():
    state, finding = _state_with_finding()
    draft = json.dumps(
        {
            "executive_summary": "FastAPI looks like the strongest fit.",
            "key_findings": [
                {"statement": "FastAPI is async-first.", "finding_indices": [0], "confidence": 0.8}
            ],
            "comparison": None,
            "trade_offs": ["Smaller ecosystem than Django"],
            "risks": [],
            "implementation_considerations": [],
        }
    )
    llm = FakeLLM([draft])
    finalizer = LLMFinalizer(llm)
    state.usage.llm_calls = 3
    state.usage.tool_calls = 2

    finalizer.finalize(state, _context(state))

    assert state.report is not None
    assert len(state.report.key_findings) == 1
    kf = state.report.key_findings[0]
    assert kf.evidence_ids == [finding.evidence_ids[0]]
    assert kf.supported is True
    assert state.report.validation_passed is True
    assert state.report.metadata.run_id == state.run_id
    assert state.report.metadata.llm_calls == 4  # 3 pre-set + this finalizer's own call
    assert state.report.metadata.tool_calls == 2
    assert len(state.report.sources) == 1
    assert state.report.sources[0].url == "https://a.example"
    assert "tavily" in state.report.metadata.providers_used


def test_llm_failure_falls_back_to_raw_findings():
    state, finding = _state_with_finding()
    llm = FakeLLM(["not json", "still not json"])
    finalizer = LLMFinalizer(llm)

    finalizer.finalize(state, _context(state))

    assert state.report is not None
    assert len(state.report.key_findings) == 1
    assert state.report.key_findings[0].statement == finding.statement
    assert state.report.key_findings[0].evidence_ids == finding.evidence_ids
    assert any("Report synthesis failed" in m for m in state.report.limitations)


def test_out_of_range_finding_index_leaves_finding_unsupported():
    state, _ = _state_with_finding()
    draft = json.dumps(
        {
            "executive_summary": "Summary.",
            "key_findings": [{"statement": "bogus", "finding_indices": [99], "confidence": 0.5}],
            "trade_offs": [],
            "risks": [],
            "implementation_considerations": [],
        }
    )
    llm = FakeLLM([draft])
    finalizer = LLMFinalizer(llm)

    finalizer.finalize(state, _context(state))  # must not raise

    kf = state.report.key_findings[0]
    assert kf.evidence_ids == []
    assert kf.supported is False
    assert state.report.validation_passed is False


def test_conclusion_finding_with_no_evidence_is_marked_unsupported():
    state = RunState(objective=OBJECTIVE)
    state.plan = None
    conclusion = _finding("This looks like the better overall choice.")
    state.add_finding(conclusion)
    draft = json.dumps(
        {
            "executive_summary": "Summary.",
            "key_findings": [
                {"statement": conclusion.statement, "finding_indices": [0], "confidence": 0.4}
            ],
            "trade_offs": [],
            "risks": [],
            "implementation_considerations": [],
        }
    )
    llm = FakeLLM([draft])
    finalizer = LLMFinalizer(llm)

    finalizer.finalize(state, _context(state))

    kf = state.report.key_findings[0]
    assert kf.evidence_ids == []
    assert kf.supported is False


def test_no_findings_adds_a_limitation_so_the_report_still_validates():
    state = RunState(objective=OBJECTIVE)
    state.plan = None
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "executive_summary": "No evidence was collected.",
                    "key_findings": [],
                    "trade_offs": [],
                    "risks": [],
                    "implementation_considerations": [],
                }
            )
        ]
    )
    finalizer = LLMFinalizer(llm)

    finalizer.finalize(state, _context(state))  # must not raise Report's own validator

    assert state.report.key_findings == []
    assert any("No key findings" in m for m in state.report.limitations)


def test_duplicate_source_urls_are_deduplicated():
    state = RunState(objective=OBJECTIVE)
    state.plan = None
    state.add_evidence(_evidence("https://same.example"))
    ev2 = _evidence("https://same.example")
    ev2.id = ev2.id + "-2"  # avoid the duplicate-id guard so we get two rows with one url
    state.evidence[ev2.id] = ev2
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "executive_summary": "s",
                    "key_findings": [],
                    "trade_offs": [],
                    "risks": [],
                    "implementation_considerations": [],
                }
            )
        ]
    )
    state.limitations.append("no findings for this test")
    finalizer = LLMFinalizer(llm)

    finalizer.finalize(state, _context(state))

    assert len(state.report.sources) == 1


def test_duration_reflects_elapsed_time():
    state, _ = _state_with_finding()
    clock = FakeClock()
    ctx = _context(state, clock)
    clock.now += 12.5
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "executive_summary": "s",
                    "key_findings": [],
                    "trade_offs": [],
                    "risks": [],
                    "implementation_considerations": [],
                }
            )
        ]
    )
    state.limitations.append("no findings for this test")
    finalizer = LLMFinalizer(llm)

    finalizer.finalize(state, ctx)

    assert state.report.metadata.duration_s >= 12.5
