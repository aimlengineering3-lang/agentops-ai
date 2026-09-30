import json

from agentops.agents.critic import HybridCritic
from agentops.contracts import (
    CritiqueAction,
    CritiqueVerdict,
    Evidence,
    Finding,
    FindingKind,
    IssueType,
    Plan,
    RunState,
    Severity,
    SourceType,
    Subtask,
    SubtaskStatus,
)
from agentops.orchestrator import BudgetGuard, EventBus, RunContext
from agentops.testing.fakes import FakeLLM

OBJECTIVE = "Compare three approaches for an enterprise AI platform"


def _context(state: RunState) -> RunContext:
    bus = EventBus()
    return RunContext(bus=bus, guard=BudgetGuard(state, bus))


def _subtask(status: SubtaskStatus = SubtaskStatus.DONE) -> Subtask:
    return Subtask(
        id="s1",
        description="Research FastAPI, Django and Flask for a small team",
        acceptance_criteria=["at least two independent sources"],
        status=status,
    )


def _evidence(url: str, claim: str = "claim") -> Evidence:
    return Evidence(
        subtask_id="s1",
        source_type=SourceType.WEB,
        title="Doc",
        url=url,
        extracted_claim=claim,
        provider="fake-search",
    )


def _finding(statement: str = "FastAPI is async-first.", evidence_ids=None) -> Finding:
    return Finding(
        subtask_id="s1",
        statement=statement,
        kind=FindingKind.EVIDENCE if evidence_ids else FindingKind.CONCLUSION,
        evidence_ids=evidence_ids or [],
        confidence=0.8,
    )


def _state(subtask_status: SubtaskStatus = SubtaskStatus.DONE) -> RunState:
    state = RunState(objective=OBJECTIVE)
    state.plan = Plan(subtasks=[_subtask(subtask_status)])
    return state


def test_pass_when_evidence_findings_and_llm_pass_are_all_clean():
    state = _state()
    state.add_evidence(_evidence("https://a.example"))
    state.add_evidence(_evidence("https://b.example"))
    state.add_finding(_finding(evidence_ids=[list(state.evidence)[0]]))
    llm = FakeLLM([json.dumps({"issues": []})])
    critic = HybridCritic(llm)

    critique = critic.critique(state, _context(state), iteration=1)

    assert critique.verdict == CritiqueVerdict.PASS
    assert critique.issues == []
    assert len(llm.calls) == 1  # findings exist, so the LLM pass does run


def test_no_evidence_gives_one_high_severity_research_more_issue():
    state = _state()
    llm = FakeLLM([])  # must not be called: no findings exist
    critic = HybridCritic(llm)

    critique = critic.critique(state, _context(state), iteration=1)

    assert critique.verdict == CritiqueVerdict.FAIL
    assert len(critique.issues) == 1
    issue = critique.issues[0]
    assert issue.type == IssueType.INSUFFICIENT_EVIDENCE
    assert issue.severity == Severity.HIGH
    assert issue.action == CritiqueAction.RESEARCH_MORE
    assert llm.calls == []


def test_low_domain_diversity_flags_weak_source_coverage():
    state = _state()
    state.add_evidence(_evidence("https://a.example/1"))
    state.add_evidence(_evidence("https://a.example/2"))  # same domain
    state.add_finding(_finding(evidence_ids=[list(state.evidence)[0]]))
    llm = FakeLLM([json.dumps({"issues": []})])
    critic = HybridCritic(llm)

    critique = critic.critique(state, _context(state), iteration=1)

    types = [i.type for i in critique.issues]
    assert IssueType.WEAK_SOURCE_COVERAGE in types
    assert IssueType.INSUFFICIENT_EVIDENCE not in types  # had >= min_evidence items


def test_evidence_without_findings_flags_incomplete_output():
    state = _state()
    state.add_evidence(_evidence("https://a.example"))
    state.add_evidence(_evidence("https://b.example"))
    llm = FakeLLM([])  # no findings yet, so the LLM pass is skipped
    critic = HybridCritic(llm)

    critique = critic.critique(state, _context(state), iteration=1)

    assert len(critique.issues) == 1
    assert critique.issues[0].type == IssueType.INCOMPLETE_OUTPUT
    assert critique.issues[0].action == CritiqueAction.REANALYZE


def test_llm_contradiction_maps_to_correct_subtask_and_action():
    state = _state()
    state.add_evidence(_evidence("https://a.example"))
    state.add_evidence(_evidence("https://b.example"))
    state.add_finding(_finding("FastAPI is fastest.", [list(state.evidence)[0]]))
    state.add_finding(_finding("FastAPI is the slowest option.", [list(state.evidence)[1]]))
    llm_json = json.dumps(
        {
            "issues": [
                {
                    "finding_index": 1,
                    "type": "contradiction",
                    "severity": "high",
                    "detail": "Contradicts finding 0 about speed",
                }
            ]
        }
    )
    llm = FakeLLM([llm_json])
    critic = HybridCritic(llm)

    critique = critic.critique(state, _context(state), iteration=1)

    contradiction = next(i for i in critique.issues if i.type == IssueType.CONTRADICTION)
    assert contradiction.subtask_id == "s1"
    assert contradiction.action == CritiqueAction.REANALYZE
    assert contradiction.severity == Severity.HIGH


def test_out_of_range_finding_index_from_llm_is_dropped():
    state = _state()
    state.add_evidence(_evidence("https://a.example"))
    state.add_evidence(_evidence("https://b.example"))
    state.add_finding(_finding(evidence_ids=[list(state.evidence)[0]]))
    llm_json = json.dumps(
        {
            "issues": [
                {"finding_index": 99, "type": "contradiction", "severity": "low", "detail": "x"}
            ]
        }
    )
    llm = FakeLLM([llm_json])
    critic = HybridCritic(llm)

    critique = critic.critique(state, _context(state), iteration=1)  # must not raise

    assert critique.verdict == CritiqueVerdict.PASS
    assert critique.issues == []


def test_llm_pass_failure_degrades_to_deterministic_only():
    state = _state()
    state.add_evidence(_evidence("https://a.example"))
    state.add_evidence(_evidence("https://b.example"))
    state.add_finding(_finding(evidence_ids=[list(state.evidence)[0]]))
    llm = FakeLLM(["not json", "still not json"])
    critic = HybridCritic(llm)

    critique = critic.critique(state, _context(state), iteration=1)  # must not raise

    assert critique.verdict == CritiqueVerdict.PASS  # deterministic checks were clean
    assert any("Critic LLM review skipped" in m for m in state.limitations)


def test_subtask_not_done_flags_missing_subtask():
    state = _state(subtask_status=SubtaskStatus.PENDING)
    llm = FakeLLM([])
    critic = HybridCritic(llm)

    critique = critic.critique(state, _context(state), iteration=1)

    assert len(critique.issues) == 1
    assert critique.issues[0].type == IssueType.MISSING_SUBTASK
    assert critique.issues[0].action == CritiqueAction.REPLAN
